"""Exercise the real SDK runner against a fake HTTP model, without paid calls."""

import asyncio
import importlib.util
import json
import os
import unittest
from urllib.error import HTTPError
from unittest.mock import Mock, patch

from federated_agent.errors import LLMResponseError
from federated_agent.planner import LLMResearchPlanner
from federated_agent.app import build_agent

HAS_SDK = importlib.util.find_spec("agents") is not None
if HAS_SDK:
    import httpx2 as httpx
    from federated_agent.sdk_runtime import AgentsSDKProvider


PLAN = {"analysis_type": "descriptive_statistics", "variables": ["age at diagnosis"],
        "sites": ["hospital 1", "hospital 2"], "database_label": "default", "purpose": "Describe age."}


def completion(plan=None, *, finish="tool_calls", name="propose_research_plan", content=None, calls=1):
    return {"id": "test-completion", "object": "chat.completion", "created": 1, "model": "test-local",
            "choices": [{"index": 0, "finish_reason": finish, "message": {
                "role": "assistant", "content": content,
                "tool_calls": [{"id": "call_" + str(i), "type": "function", "function": {
                    "name": name, "arguments": json.dumps({"proposal_json": json.dumps(PLAN if plan is None else plan)})}}
                    for i in range(calls)] if finish == "tool_calls" else None}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}}


def sse(delta, finish=None):
    return ("data: " + json.dumps({"id": "test-completion", "object": "chat.completion.chunk", "created": 1,
        "model": "test-local", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}) + "\n\n").encode()


def streamed_plan(plan=None, *, finish="tool_calls", done=True):
    arguments = json.dumps({"proposal_json": json.dumps(PLAN if plan is None else plan)})
    return [sse({"role": "assistant", "reasoning_content": "HIDDEN_REASONING", "tool_calls": [
                {"index": 0, "id": "call_0", "type": "function", "function": {
                    "name": "propose_research_plan", "arguments": ""}}]}),
            sse({"tool_calls": [{"index": 0, "function": {"arguments": arguments[:20]}}]}),
            sse({"tool_calls": [{"index": 0, "function": {"arguments": arguments[20:]}}]}),
            sse({}, finish), *([b"data: [DONE]\n\n"] if done else [])]


@unittest.skipUnless(HAS_SDK, "Install the [sdk] extra to exercise the Agents SDK runtime")
class AgentsSDKTests(unittest.TestCase):
    def provider(self, *, reply=None, chunks=None, status=200, delay=0, timeout=5):
        self.requests = []
        self.closed = []
        owner = self

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                for chunk in chunks:
                    if delay:
                        await asyncio.sleep(delay)
                    yield chunk

            async def aclose(self):
                owner.closed.append(True)

        async def handle(request):
            self.requests.append((str(request.url), dict(request.headers), json.loads(request.content)))
            if chunks is not None:
                return httpx.Response(status, headers={"content-type": "text/event-stream"}, stream=Stream())
            return httpx.Response(status, json=completion() if reply is None else reply)

        return AgentsSDKProvider(endpoint="http://localhost:9000/v1/chat/completions", model="test-local",
            api_key="test-secret", extra_body={"thinking": {"type": "disabled"}},
            transport=httpx.MockTransport(handle), timeout_seconds=timeout)

    def test_actual_sdk_runner_executes_only_proposal_tool_at_configured_endpoint(self):
        provider = self.provider()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "never-use-openai"}):
            self.assertEqual(provider.generate_json("research instructions", "research request"), PLAN)
        self.assertEqual(len(self.requests), 1)
        url, headers, body = self.requests[0]
        self.assertEqual(url, "http://localhost:9000/v1/chat/completions")
        self.assertEqual(body["model"], "test-local")
        self.assertEqual(headers["authorization"], "Bearer test-secret")
        self.assertNotIn("test-secret", json.dumps(body))
        self.assertEqual([t["function"]["name"] for t in body["tools"]], ["propose_research_plan"])
        self.assertEqual(body["tool_choice"], "required")
        self.assertFalse(body["parallel_tool_calls"])
        self.assertEqual(body["thinking"], {"type": "disabled"})

    def test_streamed_tool_arguments_validate_before_return(self):
        provider = self.provider(chunks=streamed_plan())
        seen = []
        self.assertEqual(provider.generate_json_stream("s", "u", seen.append), PLAN)
        self.assertTrue(seen)
        self.assertNotIn("HIDDEN_REASONING", "".join(seen))
        self.assertTrue(self.closed)
        self.assertEqual(len(self.requests), 1)

    def test_keyless_local_endpoint_does_not_inherit_openai_credentials(self):
        provider = self.provider()
        provider.api_key = None
        with patch.dict(os.environ, {"OPENAI_API_KEY": "private-openai-key", "OPENAI_ADMIN_KEY": "private-admin",
                                   "OPENAI_ORG_ID": "private-org", "OPENAI_PROJECT_ID": "private-project"}):
            provider.generate_json("s", "u")
        headers = self.requests[0][1]
        self.assertNotIn("authorization", headers)
        self.assertNotIn("private-", json.dumps(self.requests))

    def test_feedback_stream_is_real_and_has_no_tools(self):
        provider = self.provider(chunks=[sse({"content": "First "}), sse({"content": "second"}),
            sse({}, "stop"), b"data: [DONE]\n\n"])
        stream = provider.stream_text("s", "u")
        self.assertEqual(next(stream), "First ")
        self.assertEqual(list(stream), ["second"])
        self.assertFalse(self.requests[0][2].get("tools"))
        self.assertTrue(self.closed)

    def test_closing_feedback_stream_closes_transport(self):
        provider = self.provider(chunks=[sse({"content": "First "}), sse({}, "stop"), b"data: [DONE]\n\n"])
        stream = provider.stream_text("s", "u")
        next(stream)
        stream.close()
        self.assertTrue(self.closed)

    def test_truncated_refused_missing_done_and_invalid_streams_fail_closed(self):
        for chunks in [streamed_plan(done=False), streamed_plan(finish="length"),
                       streamed_plan(finish="content_filter"), [b"data: not-json\n\n"],
                       [b"data: \xff\n\n"], [b"data: [DONE]\n\n"]]:
            with self.subTest(chunks=len(chunks)):
                provider = self.provider(chunks=chunks)
                with self.assertRaises(LLMResponseError):
                    provider.generate_json_stream("s", "u", lambda _: None)
                self.assertTrue(self.closed)
                self.assertEqual(len(self.requests), 1)

    def test_bounded_response_and_timeout(self):
        # Allow client/runner startup, then force a timeout inside the open stream.
        # A 20 ms total deadline could expire before any stream existed to close.
        for chunks, delay, timeout in [([b"x" * 1_000_001], 0, 5), (streamed_plan(), 2, 1)]:
            provider = self.provider(chunks=chunks, delay=delay, timeout=timeout)
            with self.assertRaises(LLMResponseError):
                provider.generate_json_stream("s", "u", lambda _: None)
            self.assertEqual(len(self.requests), 1)
            self.assertTrue(self.closed)

    def test_auth_rate_limit_and_billing_errors_keep_existing_classification(self):
        for code in [401, 402, 429]:
            provider = self.provider(status=code, reply={"error": {"message": "test-secret", "type": "test"}})
            with self.assertRaises(HTTPError) as caught:
                provider.generate_json("s", "u")
            self.assertEqual(caught.exception.code, code)
            self.assertNotIn("test-secret", str(caught.exception))
            self.assertEqual(len(self.requests), 1)

    def test_unknown_tool_multiple_calls_and_plain_text_cannot_create_plan(self):
        for reply in [completion(name="submit_task"), completion(calls=2),
                      completion(finish="stop", content=json.dumps(PLAN)), completion(plan=[]),
                      completion(finish="length", content=json.dumps(PLAN))]:
            provider = self.provider(reply=reply)
            with self.assertRaises(LLMResponseError):
                provider.generate_json("s", "u")

    def test_existing_domain_pipeline_still_validates_and_requires_confirmation(self):
        agent = build_agent()
        agent.executor.submit = Mock(side_effect=AssertionError("Must not execute at planning time"))
        planner = LLMResearchPlanner(self.provider(), retries=0)
        response = agent.respond("Describe age in hospital 1 and hospital 2.", planner)
        self.assertEqual(response.status, "ready")
        self.assertIsNotNone(response.prepared_task)
        agent.executor.submit.assert_not_called()

    def test_unknown_sites_and_unsupported_filters_not_silently_replaced(self):
        for plan in [PLAN | {"sites": ["imaginary hospital"]},
                     PLAN | {"unsupported_requirements": ["Only women under 25"]}]:
            planner = LLMResearchPlanner(self.provider(reply=completion(plan)), retries=0)
            response = build_agent().respond("request", planner)
            self.assertIsNone(response.prepared_task)
            self.assertIn(response.status, {"needs_clarification", "unsupported"})

    def test_multiturn_partial_update_preserves_other_requirements(self):
        from federated_agent.models import ResearchPlan
        previous = ResearchPlan.model_validate(PLAN | {"sites": [], "unsupported_requirements": ["Only women"]})
        provider = self.provider(reply=completion({"changes": {"sites": ["hospital 1", "hospital 2"]}}))
        planner = LLMResearchPlanner(provider, retries=0)
        plan = planner.revise_plan("hospital 1 and hospital 2", previous, build_agent().algorithm_catalog)
        self.assertEqual(plan.variables, previous.variables)
        self.assertEqual(plan.unsupported_requirements, ["Only women"])
        self.assertEqual(plan.sites, ["hospital 1", "hospital 2"])

    def test_dataset_missing_uses_clarification_after_actual_sdk_proposal(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                proposal = PLAN | {"database_label": None}
                provider = self.provider(chunks=streamed_plan(proposal)) if streaming else self.provider(reply=completion(proposal))
                planner = LLMResearchPlanner(provider, retries=0)
                if streaming:
                    planner.on_delta = lambda _: None
                agent = build_agent()
                agent.executor.submit = Mock(side_effect=AssertionError("No implicit execution"))
                response = agent.respond("Describe age in hospital 1 and hospital 2", planner)
                self.assertEqual(response.status, "needs_clarification")
                self.assertIsNone(response.prepared_task)
                self.assertEqual(response.clarification.questions[0].field, "database_label")
                self.assertNotIn("default_database_label", planner.context)
                agent.executor.submit.assert_not_called()

    def test_dataset_omission_and_explicit_null_are_different_sdk_updates(self):
        from federated_agent.models import ResearchPlan
        previous = ResearchPlan.model_validate(PLAN)
        for changes, expected in [({"variables": ["survival status"]}, "ready"),
                                  ({"database_label": None}, "needs_clarification")]:
            with self.subTest(changes=changes):
                provider = self.provider(reply=completion({"changes": changes}))
                response = build_agent().respond("Follow-up", LLMResearchPlanner(provider, retries=0), previous_plan=previous)
                self.assertEqual(response.status, expected)
                if expected == "ready":
                    self.assertEqual(response.prepared_task.task_spec.databases[0].label, "default")
                    self.assertIn("dataset_retained", [item.code for item in response.issues])
                else:
                    self.assertIsNone(response.clarification.research_plan.database_label)

    def test_extra_script_fields_cannot_bypass_researchplan_schema(self):
        provider = self.provider(reply=completion(PLAN | {"script": "run_anything()"}))
        response = build_agent().respond("request", LLMResearchPlanner(provider, retries=0))
        self.assertEqual(response.status, "planner_error")
        self.assertIsNone(response.prepared_task)

    def test_context_exposes_categorical_datatype_not_patient_data(self):
        context = build_agent().planner_context()
        status = next(v for v in context["semantic_variables_not_data_availability"] if v["name"] == "survival_status")
        self.assertEqual(status["datatype"], "categorical")
        self.assertNotIn("server_url", context)
        self.assertNotIn("results", context)

    def test_tracing_is_disabled_even_with_environment_key(self):
        from agents.tracing import get_trace_provider
        provider = self.provider()
        self.assertTrue(get_trace_provider()._disabled)
        with patch("federated_agent.sdk_runtime.Runner.run", wraps=__import__("agents").Runner.run) as run:
            provider.generate_json("s", "u")
        self.assertTrue(run.call_args.kwargs["run_config"].tracing_disabled)
        self.assertFalse(run.call_args.kwargs["run_config"].trace_include_sensitive_data)

    def test_invalid_endpoint_is_rejected_without_contacting_provider(self):
        for endpoint in ["https://host/v1", "https://key@host/chat/completions", "https://host/chat/completions?secret=yes"]:
            with self.assertRaises(ValueError):
                AgentsSDKProvider(endpoint=endpoint, model="test")


if __name__ == "__main__":
    unittest.main()
