"""Local system questions must not become research tasks or mutate a study."""

import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rich.console import Console

from federated_agent.cli import main, run_chat
from federated_agent.conversation import conversation_response, is_local_conversation
from federated_agent.models import ResearchPlan
from federated_agent.planner import DeepSeekJSONProvider, DemoResearchPlanner, LLMResearchPlanner, OpenAICompatibleJSONProvider
from federated_agent.terminal_ui import TerminalUI
from federated_agent.app import build_agent
from federated_agent.resources import SOURCE_CHECKOUT


class SystemConversationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.agent = build_agent()
        self.provider = DeepSeekJSONProvider("SECRET_TEST_KEY", model="configured-model-override")
        self.planner = LLMResearchPlanner(self.provider)
        self.now = datetime(2026, 9, 17, 0, 30, tzinfo=timezone(timedelta(hours=2)))

    def test_model_question_variants_use_configured_identity(self):
        for request in ("which llm model are u using now", "Which model are you running currently?",
                        "what is your model?", "What's your LLM?", "what's your agent framework?",
                        "which llm are you using?", "could you tell me which model are you using?",
                        "what model are you using, please?", "\u4f60\u73b0\u5728\u7528\u7684\u662f\u4ec0\u4e48\u6a21\u578b\uff1f",
                        "Hi, which model are you using?", "what model are you?",
                        "could you tell me which model you are using?", "what's your current llm model?",
                        "\u4f60\u7528\u54ea\u4e2a\u5927\u6a21\u578b", "\u4f60\u662f\u4ec0\u4e48\u6a21\u578b", "/model"):
            with self.subTest(request=request):
                response = conversation_response(request, self.planner)
                self.assertIsNotNone(response)
                self.assertEqual(response.status, "conversation")
                self.assertIn("configured-model-override", response.message)
                self.assertIn("DeepSeek", response.message)
                self.assertNotIn("SECRET_TEST_KEY", response.model_dump_json())

    def test_local_model_name_does_not_imply_a_remote_provider(self):
        planner = LLMResearchPlanner(OpenAICompatibleJSONProvider(
            endpoint="http://private-host:9000/v1/chat/completions", model="my-local-model", api_key="PRIVATE_KEY"))
        response = conversation_response("what model are you using", planner)
        self.assertIn("my-local-model", response.message)
        self.assertIn("OpenAI-compatible", response.message)
        self.assertNotIn("DeepSeek", response.message)
        self.assertNotIn("private-host", response.message)
        self.assertNotIn("PRIVATE_KEY", response.message)

    def test_demo_and_unknown_planner_do_not_invent_a_model(self):
        response = conversation_response("/model", DemoResearchPlanner())
        self.assertIn("not an LLM", response.message)
        response = conversation_response("/model", Mock())
        self.assertIn("not available", response.message)
        self.assertNotIn("DeepSeek", response.message)

    def test_dates_use_local_clock_and_offset_not_utc_calendar_day(self):
        # At this instant UTC is still September 16; the local date is September 17.
        self.assertEqual(self.now.astimezone(timezone.utc).day, 16)
        for request in ("what date is today", "what's today's date?", "What day is it?",
                        "What's the current date?", "Hello, what's today's date?",
                        "what is the date today?", "today's date", "/date",
                        "\u4eca\u5929\u662f\u51e0\u53f7\uff1f", "\u4eca\u5929\u661f\u671f\u51e0"):
            with self.subTest(request=request):
                response = conversation_response(request, now=self.now)
                self.assertEqual(response.status, "conversation")
                self.assertIn("2026-09-17", response.message)
                self.assertIn("UTC+02:00", response.message)
                self.assertNotIn("2026-09-16", response.message)

    def test_time_is_aware_and_date_is_not_cached(self):
        time = conversation_response("what time is it now?", now=self.now)
        self.assertIn("2026-09-17T00:30:00+02:00", time.message)
        next_day = conversation_response("/date", now=self.now + timedelta(days=1))
        self.assertIn("2026-09-18", next_day.message)

    def test_english_and_chinese_are_selected_from_researcher_input(self):
        self.assertTrue(conversation_response("what date is today", now=self.now).message.startswith("Today"))
        self.assertTrue(conversation_response("\u4eca\u5929\u662f\u51e0\u53f7", now=self.now).message.startswith("\u4eca\u5929"))
        self.assertTrue(conversation_response("/model", self.planner, language="zh").message.startswith("\u5f53\u524d"))

    def test_queries_bypass_catalog_model_and_execution_even_when_catalog_is_down(self):
        self.agent.executor.submit = Mock()
        with patch.object(self.agent, "refresh_metadata", side_effect=AssertionError("No registry request")), \
                patch.object(self.agent.algorithm_catalog, "snapshot", side_effect=AssertionError("No catalog request")), \
                patch.object(self.provider, "generate_json", side_effect=AssertionError("No model request")), \
                patch.object(self.provider, "stream_text", side_effect=AssertionError("No narration")):
            for request in ("which llm model are u using now", "what date is today", "/model", "/date", "/time"):
                response = self.agent.respond(request, self.planner)
                self.assertEqual(response.status, "conversation")
                self.assertIsNone(response.prepared_task)
                self.assertEqual(response.submission_state, "not_submitted")
                self.assertIsNone(response.catalog)
        self.agent.executor.submit.assert_not_called()

    def test_research_and_mixed_requests_are_not_swallowed(self):
        for request in ("What models can analyze survival?", "Which algorithms are available?",
                        "Calculate statistics for diagnosis date", "what date is today? Then download patient records",
                        "Which model are you using; train a Cox model on hospital 1",
                        "Hi, describe age at diagnosis in hospital 1 and hospital 2.",
                        "\u4eca\u5929\u662f\u51e0\u53f7\uff1f\u7136\u540e\u4e0b\u8f7d\u60a3\u8005\u6570\u636e",
                        "/model now submit", '{"reply": "what date is today"}'):
            with self.subTest(request=request):
                self.assertIsNone(conversation_response(request, self.planner))
                self.assertFalse(is_local_conversation(request))
        response = self.agent.respond("Describe age at diagnosis in hospital 1 and hospital 2 using default.", DemoResearchPlanner())
        self.assertEqual(response.status, "ready")

    def test_ui_does_not_narrate_or_show_catalog_for_system_queries(self):
        output = io.StringIO()
        ui = TerminalUI(plain=True, console=Console(file=output))
        with patch.object(self.provider, "stream_text", side_effect=AssertionError("No model narration")):
            ui.present(conversation_response("/model", self.planner), self.planner, self.root)
        self.assertIn("configured-model-override", output.getvalue())
        self.assertNotIn("Registered algorithms", output.getvalue())
        self.assertNotIn("not_submitted", output.getvalue())
        self.assertFalse((self.root / "stream.json").exists())

    def test_cli_exact_reported_sequence_is_local(self):
        args = SimpleNamespace(output=self.root, provider="deepseek", backend="mock", stream=True)
        replies = iter(["hi", "which llm model are u using now", "what date is today", "/model", "/date", "/exit"])
        output = io.StringIO()
        with redirect_stdout(output), patch.object(self.agent, "refresh_metadata", side_effect=AssertionError("No metadata")), \
                patch.object(self.provider, "generate_json_stream", side_effect=AssertionError("No LLM")), \
                patch.object(self.provider, "stream_text", side_effect=AssertionError("No LLM")):
            run_chat(self.agent, self.planner, args, read=lambda _: next(replies))
        self.assertNotIn("Reading intent", output.getvalue())
        self.assertNotIn("Registered algorithms", output.getvalue())
        self.assertIn("configured-model-override", output.getvalue())
        self.assertIn("Today is", output.getvalue())
        responses = list(self.root.glob("session-*/turn-*/response.json"))
        self.assertEqual(len(responses), 5)
        self.assertTrue(all(json.loads(p.read_text())["status"] == "conversation" for p in responses))
        self.assertFalse(list(self.root.rglob("prepared_task.json")))
        self.assertFalse(list(self.root.rglob("submission.json")))

    def test_queries_preserve_clarification_and_plan_across_restart(self):
        provider = Mock()
        provider.model = "local-test"
        initial = ResearchPlan(analysis_type="descriptive_statistics", variables=["age at diagnosis"], purpose="test")
        provider.generate_json.side_effect = [
            initial.model_dump(), {"changes": {"sites": ["hospital 1", "hospital 2"]}}]
        planner = LLMResearchPlanner(provider)
        args = SimpleNamespace(output=self.root, provider="local", backend="mock", stream=False)
        replies = iter(["Describe age", "which llm model are u using now", "what date is today", "/exit"])
        with redirect_stdout(io.StringIO()):
            run_chat(self.agent, planner, args, read=lambda _: next(replies))
        session = next(self.root.glob("session-*"))
        saved = json.loads((session / "session.json").read_text())
        self.assertEqual(saved["current_plan"]["variables"], ["age at diagnosis"])
        self.assertIsNotNone(saved["pending"])
        self.assertEqual(saved["revision"], 1)
        self.assertEqual(saved["active_turn"], "turn-001")
        replies = iter(["hospital 1 and hospital 2", "", "/exit"])
        args.resume = session.name
        with redirect_stdout(io.StringIO()):
            run_chat(self.agent, planner, args, read=lambda _: next(replies))
        self.assertEqual(provider.generate_json.call_count, 2)
        continuation = json.loads(provider.generate_json.call_args.args[1])
        self.assertEqual(continuation["previous_plan"]["variables"], ["age at diagnosis"])
        self.assertEqual(continuation["researcher_reply"], "hospital 1 and hospital 2")
        self.assertFalse(list(session.rglob("submission.json")))

    def test_interactive_free_text_interjection_is_not_wrapped_as_a_field_answer(self):
        provider = Mock()
        provider.generate_json.return_value = ResearchPlan(
            analysis_type="descriptive_statistics", variables=["age at diagnosis"], purpose="test").model_dump()
        pending = self.agent.respond("Describe age", LLMResearchPlanner(provider))
        ui = TerminalUI(read=lambda _: "what date is today", console=Console(file=io.StringIO()))
        ui.interactive = True
        with patch("federated_agent.terminal_ui.choice_application") as menu:
            menu.return_value.run.return_value = ["write"]
            self.assertEqual(ui.ask(pending), "what date is today")

    def test_nonchat_system_query_does_not_refresh_catalog(self):
        argv = ["agent", "--provider", "demo", "--request", "what date is today", "--output", str(self.root)]
        with patch("sys.argv", argv), \
                patch("federated_agent.service.FederatedResearchAgent.refresh_metadata", side_effect=AssertionError("No registry")), \
                redirect_stdout(io.StringIO()):
            main()
        self.assertEqual(json.loads((self.root / "response.json").read_text())["status"], "conversation")


@unittest.skipUnless(importlib.util.find_spec("agents"), "SDK extra required")
class SDKConversationTests(unittest.TestCase):
    def test_sdk_preserves_deepseek_provider_and_reports_model_override(self):
        from federated_agent.sdk_runtime import AgentsSDKProvider
        provider = AgentsSDKProvider.from_provider(DeepSeekJSONProvider("FAKE_KEY", model="override-test"))
        planner = LLMResearchPlanner(provider)
        with patch.object(provider, "_run", side_effect=AssertionError("No SDK model run")):
            response = conversation_response("which llm model are u using now", planner)
        self.assertIn("override-test (DeepSeek)", response.message)
        self.assertIn("OpenAI Agents SDK", response.message)
        self.assertIn(provider.runtime_version, response.message)

    def test_real_sdk_cli_subprocess_never_calls_the_unused_model_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = subprocess.run([sys.executable, str(SOURCE_CHECKOUT / "start_agent.py"),
                "--provider", "local", "--llm-endpoint", "http://127.0.0.1:1/v1/chat/completions",
                "--model", "actual-configured-cli-model", "--backend", "mock", "--plain", "--output", str(output)],
                input="hi\nwhich llm model are u using now\nwhat date is today\n/model\n/date\n/exit\n",
                capture_output=True, encoding="utf-8", timeout=40,
                env=os.environ | {"PYTHONUTF8": "1"})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("actual-configured-cli-model", result.stdout)
            self.assertIn("Agent runtime: OpenAI Agents SDK", result.stdout)
            self.assertIn("Today is", result.stdout)
            self.assertNotIn("Reading intent", result.stdout)
            self.assertNotIn("Registered algorithms", result.stdout)
            self.assertNotIn("Traceback", result.stderr)
            self.assertFalse(list(output.rglob("prepared_task.json")))


if __name__ == "__main__":
    unittest.main()
