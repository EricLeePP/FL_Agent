import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rich.console import Console

from federated_agent.cli import run_chat
from federated_agent.conversation import response_language, social_response
from federated_agent.models import AgentResponse, ResearchPlan
from federated_agent.narration import explain
from federated_agent.planner import DeepSeekJSONProvider, DemoResearchPlanner, LLMResearchPlanner
from federated_agent.terminal_ui import TerminalUI
from federated_agent.app import build_agent


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.agent = build_agent()
        self.text = io.StringIO()
        self.ui = TerminalUI(plain=True, console=Console(file=self.text, width=100))

    def test_pure_greetings_do_not_call_planner_catalog_or_executor(self):
        planner = Mock(side_effect=AssertionError("must not call LLM"))
        self.agent.executor.submit = Mock()
        with patch.object(self.agent.algorithm_catalog, "snapshot", side_effect=AssertionError("must not query catalog")):
            for request in ("hi", " Hi!! ", "Hello there.", "Good morning", "\u4f60\u597d\uff01", "thanks", "Thank you!"):
                with self.subTest(request=request):
                    response = self.agent.respond(request, planner)
                    self.assertEqual(response.status, "conversation")
                    self.assertEqual(response.submission_state, "not_submitted")
                    self.assertIsNone(response.prepared_task)
                    self.assertEqual(response.issues, [])
                    self.assertEqual(response.suggested_algorithms, [])
        planner.create_plan.assert_not_called()
        self.agent.executor.submit.assert_not_called()

    def test_mixed_greeting_task_is_not_swallowed(self):
        for request in ("Hi, describe age at diagnosis in hospital 1 and hospital 2.",
                        "Hello, train a Cox model", "Thanks, now download records", "hi\nrun an algorithm",
                        "\u4f60\u597d\uff0c\u8bf7\u8ba1\u7b97\u5e74\u9f84"):
            self.assertIsNone(social_response(request))
        request = "Hi, describe age at diagnosis in hospital 1 and hospital 2 using default."
        self.assertEqual(self.agent.respond(request, DemoResearchPlanner()).status, "ready")

    def test_english_greeting_and_chinese_greeting_follow_input(self):
        self.assertTrue(social_response("hi").message.startswith("Hi!"))
        self.assertTrue(social_response("\u4f60\u597d").message.startswith("\u4f60\u597d"))

    def test_language_defaults_switches_and_neutral_input(self):
        self.assertEqual(response_language("Hi"), "en")
        self.assertEqual(response_language("\u8bf7\u5206\u6790\u5e74\u9f84"), "zh")
        self.assertEqual(response_language("Describe age", "zh"), "en")
        self.assertEqual(response_language("1, 2", "zh"), "zh")
        self.assertEqual(response_language("/algorithms", "zh"), "zh")

    def test_prompt_tracks_user_input_not_confirmation_or_menus(self):
        values = iter(["\u8bf7\u5206\u6790\u5e74\u9f84", "confirm", "/menu", "Describe age"])
        ui = TerminalUI(read=lambda _: next(values), console=self.ui.console)
        ui.ask()
        self.assertEqual(ui.language, "zh")
        ui.prompt("Task confirmation: ")
        self.assertEqual(ui.language, "zh")
        ui.ask()
        self.assertEqual(ui.language, "zh")
        ui.ask()
        self.assertEqual(ui.language, "en")

    def test_hospital_selection_keeps_chinese_despite_english_catalog_names(self):
        planner = Mock()
        planner.create_plan.return_value = ResearchPlan(analysis_type="descriptive_statistics", variables=["age at diagnosis"], sites=[], database_label="default", purpose="test")
        pending = self.agent.respond("Describe age", planner)
        self.ui.language = "zh"
        self.ui.interactive = True
        self.ui.read = Mock(side_effect=AssertionError("A fully specified variable must not prompt for text"))
        with patch("federated_agent.terminal_ui.choice_application") as menu:
            menu.return_value.run.return_value = ["0", "1"]
            reply = json.loads(self.ui.ask(pending))
        self.assertEqual(self.ui.language, "zh")
        self.assertEqual(reply["confirmed_choices"][0]["selected_values"], ["hospital 1", "hospital 2"])

    def test_social_ui_has_no_error_status_or_model_narration(self):
        provider = DeepSeekJSONProvider("sk-fake-key")
        with patch.object(provider, "stream_text") as stream:
            self.ui.present(social_response("hi"), LLMResearchPlanner(provider), self.root)
        stream.assert_not_called()
        text = self.text.getvalue()
        self.assertIn("Hi!", text)
        self.assertNotIn("Analysis not supported", text)
        self.assertNotIn("not_submitted", text)
        self.assertNotIn("Catalog 0.", text)

    def test_narration_language_comes_from_user_not_chinese_facts(self):
        provider = DeepSeekJSONProvider("sk-fake-key")
        for language, expected in [("en", "English"), ("zh", "Chinese")]:
            self.ui.language = language
            chinese_facts = AgentResponse(status="unsupported", message="\u4e2d\u6587\u76ee\u5f55")
            with patch.object(provider, "stream_text", return_value=iter(["test reply"])) as stream:
                self.ui._stream_reply(explain(provider, chinese_facts, self.ui.language), self.root)
            self.assertTrue(stream.call_args.args[0].endswith(f"Response language: {expected}."))
            self.assertEqual(json.loads((self.root / "stream.json").read_text())["language"], language)

    def test_planner_diagnostics_default_english_and_accept_explicit_chinese(self):
        provider = Mock()
        provider.generate_json.return_value = dict(analysis_type="descriptive_statistics", purpose="test")
        planner = LLMResearchPlanner(provider)
        planner.create_plan("Describe age", self.agent.algorithm_catalog)
        self.assertIn("Write diagnostic text in English.", provider.generate_json.call_args.args[0])
        planner.response_language = "zh"
        planner.create_plan('{"confirmed_choices": ["hospital 1"]}', self.agent.algorithm_catalog)
        self.assertIn("Write diagnostic text in Chinese.", provider.generate_json.call_args.args[0])

    def test_hi_cli_is_immediate_and_does_not_generate_script(self):
        args = SimpleNamespace(output=self.root, provider="demo", backend="mock", stream=False)
        replies = iter(["hi", "/exit"])
        capture = io.StringIO()
        with redirect_stdout(capture):
            run_chat(self.agent, DemoResearchPlanner(), args, read=lambda _: next(replies))
        self.assertIn("Hi!", capture.getvalue())
        self.assertNotIn("Reading intent", capture.getvalue())
        self.assertNotIn("Analysis not supported", capture.getvalue())
        self.assertEqual(list(self.root.glob("session-*/turn-*/prepared_task.json")), [])

    def test_social_turn_does_not_clear_pending_clarification(self):
        provider = Mock()
        plan = ResearchPlan(analysis_type="descriptive_statistics", variables=["age at diagnosis"], purpose="test")
        provider.generate_json.side_effect = [plan.model_dump(), {"changes": {"sites": ["hospital 1", "hospital 2"]}}]
        planner = LLMResearchPlanner(provider)
        args = SimpleNamespace(output=self.root, provider="local", backend="mock", stream=False)
        replies = iter(["Describe age", "hi", "hospital 1 and hospital 2", "", "/exit"])
        with redirect_stdout(io.StringIO()):
            run_chat(self.agent, planner, args, read=lambda _: next(replies))
        self.assertEqual(provider.generate_json.call_count, 2)
        continuation = json.loads(provider.generate_json.call_args.args[1])
        self.assertEqual(continuation["previous_plan"]["variables"], ["age at diagnosis"])
        self.assertEqual(continuation["researcher_reply"], "hospital 1 and hospital 2")


if __name__ == "__main__":
    unittest.main()
