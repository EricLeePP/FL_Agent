"""Phase 6 guarantees: read-only lookups shared by an MCP server and an optional SDK tool loop."""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from federated_agent.app import build_agent, build_planner
from federated_agent.bootstrap import ConfigurationError
from federated_agent.config import LLMConfig
from federated_agent.errors import LLMResponseError
from federated_agent.lookups import ResearchLookups
from federated_agent.models import ResearchPlan
from federated_agent.resources import RESOURCE_ROOT

HAS_MCP = importlib.util.find_spec("mcp") is not None
HAS_SDK = importlib.util.find_spec("agents") is not None
TOOLS = {"list_algorithms", "list_sites", "resolve_variable", "resolve_site", "check_plan"}
PLAN = {"analysis_type": "descriptive_statistics", "variables": ["age at diagnosis"],
        "sites": ["hospital 1", "hospital 2"], "database_label": "default", "purpose": "Describe age."}


def plan(**changes):
    return ResearchPlan(**(PLAN | changes))


def without_private_identifiers(test, value):
    text = json.dumps(value)
    for private in ("organisation_id", "server_url", "server_port", "collaboration_id", "replace-with-real"):
        test.assertNotIn(private, text)


class LookupTests(unittest.TestCase):
    def setUp(self):
        self.agent = build_agent()
        self.lookups = ResearchLookups(self.agent.refresh_metadata())

    def test_lookups_resolve_or_return_candidates_without_guessing(self):
        self.assertEqual(self.lookups.resolve_variable("age at diagnosis")["status"], "resolved")
        self.assertEqual(self.lookups.resolve_variable("age at diagnosis")["datatype"], "numerical")
        self.assertEqual(self.lookups.resolve_site("centre 1"),
                         {"status": "resolved", "requested_name": "centre 1", "name": "hospital 1", "database_labels": ["default"]})
        ambiguous = self.lookups.resolve_site("hosp")
        self.assertEqual(ambiguous["status"], "needs_confirmation")
        self.assertIn("hospital 2", ambiguous["candidates"])

    def test_lookups_return_only_public_planning_facts(self):
        for value in (self.lookups.list_sites(), self.lookups.list_algorithms(),
                      self.lookups.resolve_site("hospital 1"), self.lookups.resolve_variable("gender")):
            without_private_identifiers(self, value)

    def test_lookups_stay_on_the_snapshot_they_were_given(self):
        with tempfile.TemporaryDirectory() as temporary:
            deployment = Path(temporary) / "deployment.json"
            document = json.loads((RESOURCE_ROOT / "config/demo_deployment.json").read_text(encoding="utf-8"))
            deployment.write_text(json.dumps(document), encoding="utf-8")
            agent = build_agent(deployment)
            lookups = ResearchLookups(agent.refresh_metadata())
            document["sites"][0]["aliases"].append("north clinic")
            deployment.write_text(json.dumps(document), encoding="utf-8")
            self.assertEqual(lookups.resolve_site("north clinic")["status"], "needs_confirmation")
            self.assertEqual(ResearchLookups(agent.refresh_metadata()).resolve_site("north clinic")["name"], "hospital 1")

    def test_check_plan_checks_a_plan_and_never_treats_it_as_conversation(self):
        self.assertEqual(self.agent.check_plan(plan(purpose="hi")).status, "ready")
        response = self.agent.check_plan(plan(database_label=None))
        self.assertEqual(response.status, "needs_clarification")
        self.assertEqual(response.clarification.questions[0].field, "database_label")

    def test_server_explains_a_missing_mcp_extra(self):
        import io
        from contextlib import redirect_stderr

        from federated_agent import mcp_server
        error = io.StringIO()
        with patch.object(mcp_server, "MCPServer", None), redirect_stderr(error), self.assertRaises(SystemExit) as raised:
            mcp_server.main([])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("[mcp]", error.getvalue())

    def test_tool_lookups_need_the_sdk_runtime(self):
        with self.assertRaisesRegex(ConfigurationError, "sdk runtime"):
            build_planner(LLMConfig(provider="local", model="m", local_endpoint="http://127.0.0.1:9/v1/chat/completions",
                                    tool_lookups=True))


@unittest.skipUnless(HAS_MCP, "Install the [mcp] extra to run the MCP server")
class MCPServerTests(unittest.TestCase):
    def setUp(self):
        from federated_agent.mcp_server import build_server
        self.agent = build_agent()
        self.server = build_server(self.agent)

    def call(self, *calls):
        import anyio
        from mcp import Client

        async def run():
            async with Client(self.server) as client:
                listed = await client.list_tools()
                results = [await client.call_tool(name, arguments) for name, arguments in calls]
                return listed.tools, results

        return anyio.run(run)

    def test_every_tool_is_declared_read_only(self):
        tools, _ = self.call()
        self.assertEqual({tool.name for tool in tools}, TOOLS)
        for tool in tools:
            with self.subTest(tool=tool.name):
                self.assertTrue(tool.annotations.read_only_hint)
                self.assertFalse(tool.annotations.destructive_hint)

    def test_tools_answer_with_checked_public_facts(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        before = os.getcwd()
        os.chdir(directory.name)
        self.addCleanup(os.chdir, before)
        _, results = self.call(("resolve_variable", {"name": "age at diagnosis"}), ("list_sites", {}),
                               ("check_plan", {"variables": ["age at diagnosis"], "sites": ["hospital 1", "hospital 2"],
                                               "database_label": "default"}),
                               ("check_plan", {"variables": ["age at diagnosis"], "sites": ["hospital 1", "hospital 2"]}))
        resolved, sites, ready, clarification = [result.structured_content for result in results]
        self.assertEqual(resolved["canonical_name"], "age_at_initial_diagnosis")
        self.assertEqual(ready["status"], "ready")
        self.assertEqual(ready["prepared"]["database"], "default")
        self.assertEqual(clarification["status"], "needs_clarification")
        for value in (sites, ready):
            without_private_identifiers(self, value)
        self.assertEqual(os.listdir("."), [], "MCP tools must not write files")

    def test_invalid_input_and_internal_failures_are_reported_without_details(self):
        _, (invalid,) = self.call(("check_plan", {"variables": "age at diagnosis", "sites": ["hospital 1"]}))
        self.assertTrue(invalid.is_error)
        with patch.object(self.agent, "refresh_metadata", side_effect=RuntimeError("secret /home/operator/path")), \
                self.assertLogs("federated_agent.mcp_server", "ERROR"):
            _, (failed,) = self.call(("list_sites", {}))
        self.assertEqual(failed.structured_content["status"], "internal_error")
        self.assertRegex(failed.structured_content["incident_id"], r"^[0-9a-f]{12}$")
        self.assertNotIn("secret", json.dumps(failed.structured_content))

    def test_stdio_server_starts_as_a_module_and_keeps_stdout_for_the_protocol(self):
        import anyio
        from mcp import Client, StdioServerParameters

        environment = {key: value for key, value in os.environ.items() if key not in {"OPENAI_API_KEY", "DEEPSEEK_API_KEY"}}
        parameters = StdioServerParameters(command=sys.executable, args=["-m", "federated_agent.mcp_server", "-v"],
                                           env=environment)

        async def run():
            async with Client(parameters) as client:
                return await client.call_tool("resolve_site", {"name": "hospital 2"})

        result = anyio.run(run)
        self.assertEqual(result.structured_content["name"], "hospital 2")


@unittest.skipUnless(HAS_SDK, "Install the [sdk] extra to exercise the Agents SDK lookup loop")
class SDKLookupLoopTests(unittest.TestCase):
    def scripted(self, *calls):
        """A fake chat-completions endpoint that answers each request with the next tool call."""
        import httpx2 as httpx

        replies = iter(calls)
        self.requests = []

        async def handle(request):
            self.requests.append(json.loads(request.content))
            name, arguments = next(replies)
            return httpx.Response(200, json={
                "id": "scripted", "object": "chat.completion", "created": 1, "model": "test-local",
                "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {"role": "assistant", "content": None,
                             "tool_calls": [{"id": f"call_{len(self.requests)}", "type": "function",
                                             "function": {"name": name, "arguments": json.dumps(arguments)}}]}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

        return httpx.MockTransport(handle)

    def planner(self, transport, max_turns=4):
        planner = build_planner(LLMConfig(provider="local", runtime="sdk", model="test-local", tool_lookups=True,
                                          max_turns=max_turns, local_endpoint="http://127.0.0.1:9/v1/chat/completions"))
        planner.provider.transport = transport
        return planner

    def tool_outputs(self, request):
        return [message["content"] for message in request["messages"] if message.get("role") == "tool"]

    def test_the_model_looks_names_up_before_its_proposal_is_validated(self):
        transport = self.scripted(("resolve_variable", {"name": "age at diagnosis"}), ("resolve_site", {"name": "centre 2"}),
                                  ("propose_research_plan", {"proposal_json": json.dumps(PLAN)}))
        agent = build_agent()
        response = agent.respond("Describe age at diagnosis in hospital 1 and centre 2 using default.", self.planner(transport))
        self.assertEqual(response.status, "ready")
        self.assertEqual(len(self.requests), 3)
        tool_names = {tool["function"]["name"] for tool in self.requests[0]["tools"]}
        self.assertEqual(tool_names, {"propose_research_plan", *TOOLS - {"check_plan"}})
        self.assertTrue(all(request["tool_choice"] == "required" for request in self.requests))
        outputs = self.tool_outputs(self.requests[2])
        self.assertEqual(json.loads(outputs[0])["canonical_name"], "age_at_initial_diagnosis")
        self.assertEqual(json.loads(outputs[1])["name"], "hospital 2")
        for output in outputs:
            without_private_identifiers(self, json.loads(output))
        self.assertIn("read-only lookup tools", self.requests[0]["messages"][0]["content"])

    def test_a_model_that_never_proposes_stops_at_the_turn_limit(self):
        transport = self.scripted(*[("list_sites", {})] * 3)
        planner = self.planner(transport, max_turns=3)
        planner.retries = 0
        agent = build_agent()
        planner.lookups = ResearchLookups(agent.refresh_metadata())
        with self.assertRaises(LLMResponseError):
            planner.create_plan("Describe age", agent.algorithm_catalog)
        self.assertEqual(len(self.requests), 3)

    def test_without_lookups_the_runtime_offers_only_the_proposal_tool(self):
        transport = self.scripted(("propose_research_plan", {"proposal_json": json.dumps(PLAN)}))
        planner = self.planner(transport)
        planner.tool_lookups = False
        self.assertEqual(build_agent().respond("Describe age", planner).status, "ready")
        self.assertEqual([tool["function"]["name"] for tool in self.requests[0]["tools"]], ["propose_research_plan"])


if __name__ == "__main__":
    unittest.main()
