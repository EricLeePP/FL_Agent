"""Phase 4 guarantees: one set of prompts, one place that builds planners, narration
outside the terminal and deterministic menu selections."""

import importlib.util
import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

from federated_agent.app import build_agent, build_planner
from federated_agent.bootstrap import ConfigurationError
from federated_agent.clarification import apply_choices, selected_choices
from federated_agent.config import LLMConfig
from federated_agent.models import AgentResponse, ResearchPlan
from federated_agent.narration import explain, explanation_prompt
from federated_agent.planner import (
    DeepSeekJSONProvider,
    DemoResearchPlanner,
    LLMResearchPlanner,
    OpenAICompatibleJSONProvider,
)
from federated_agent.prompts import NARRATION_PROMPT, PLAN_PROMPT, TOOL_DELIVERY_PROMPT
from federated_agent.services.study import StudyChat, StudySettings

CATEGORICAL_RULE = "frequencies of a categorical variable are categorical"
KEY_VARIABLES = ("DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY_FILE", "DEEPSEEK_MODEL", "LOCAL_LLM_MODEL", "LOCAL_LLM_API_KEY")


def plan(**changes):
    return ResearchPlan(**(dict(analysis_type="descriptive_statistics", variables=["age at diagnosis"],
                                sites=["hospital 1", "hospital 2"], database_label=None,
                                purpose="selection test") | changes))


def choices(*answers):
    return json.dumps({"confirmed_choices": list(answers)})


class PromptTests(unittest.TestCase):
    def test_both_runtimes_receive_the_same_research_rules(self):
        self.assertIn(CATEGORICAL_RULE, PLAN_PROMPT.replace("\n", " "))
        self.assertNotIn(CATEGORICAL_RULE, TOOL_DELIVERY_PROMPT.replace("\n", " "))
        self.assertIn("propose_research_plan", TOOL_DELIVERY_PROMPT)
        provider = Mock()
        provider.generate_json.return_value = plan().model_dump()
        LLMResearchPlanner(provider).create_plan("Describe age", build_agent().algorithm_catalog)
        self.assertTrue(provider.generate_json.call_args.args[0].startswith(PLAN_PROMPT))

    @unittest.skipUnless(importlib.util.find_spec("agents"), "SDK extra required")
    def test_sdk_runtime_adds_only_its_delivery_format(self):
        from federated_agent.sdk_runtime import AgentsSDKProvider
        provider = AgentsSDKProvider(endpoint="http://127.0.0.1:9/v1/chat/completions", model="test-model")
        self.assertEqual(len(provider.runtime_prompt_sha256), 64)

    def test_narration_prompt_only_waives_policy_approval_in_development_mode(self):
        self.assertIn("When policy_mode is development_allow_all", NARRATION_PROMPT)
        self.assertNotIn("Policy is allow-all development mode", NARRATION_PROMPT)


class PlannerFactoryTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {})
        environment.start()
        self.addCleanup(environment.stop)
        for name in KEY_VARIABLES:
            os.environ.pop(name, None)

    def test_demo_needs_no_model(self):
        self.assertIsInstance(build_planner(LLMConfig()), DemoResearchPlanner)

    def test_deepseek_without_a_key_explains_how_to_supply_one(self):
        with self.assertRaisesRegex(ConfigurationError, "DEEPSEEK_API_KEY"):
            build_planner(LLMConfig(provider="deepseek"))

    def test_deepseek_key_is_asked_only_in_an_interactive_session(self):
        ask = Mock(return_value="sk-test-secret1")
        planner = build_planner(LLMConfig(provider="deepseek", model="test-model"), interactive=True, ask_secret=ask)
        ask.assert_called_once()
        self.assertIsInstance(planner.provider, DeepSeekJSONProvider)
        self.assertEqual(planner.provider.model, "test-model")
        with self.assertRaises(ConfigurationError):
            build_planner(LLMConfig(provider="deepseek"), interactive=False, ask_secret=ask)

    def test_local_provider_requires_endpoint_and_model(self):
        with self.assertRaisesRegex(ConfigurationError, "--llm-endpoint"):
            build_planner(LLMConfig(provider="local"))
        with self.assertRaisesRegex(ConfigurationError, "--model"):
            build_planner(LLMConfig(provider="local", local_endpoint="http://127.0.0.1:9/v1/chat/completions"))
        planner = build_planner(LLMConfig(provider="local", model="m",
                                          local_endpoint="http://127.0.0.1:9/v1/chat/completions"))
        self.assertIs(type(planner.provider), OpenAICompatibleJSONProvider)

    @unittest.skipUnless(importlib.util.find_spec("agents"), "SDK extra required")
    def test_sdk_runtime_wraps_the_configured_provider(self):
        from federated_agent.sdk_runtime import AgentsSDKProvider
        planner = build_planner(LLMConfig(provider="local", runtime="sdk", model="m",
                                          local_endpoint="http://127.0.0.1:9/v1/chat/completions"))
        self.assertIsInstance(planner.provider, AgentsSDKProvider)
        with self.assertRaisesRegex(ConfigurationError, "chat/completions"):
            build_planner(LLMConfig(provider="local", runtime="sdk", model="m", local_endpoint="http://127.0.0.1:9/v1"))


class NarrationTests(unittest.TestCase):
    def test_only_checked_preparation_facts_are_explained(self):
        provider = DeepSeekJSONProvider("sk-fake-key")
        provider.stream_text = Mock(return_value=iter(["ok"]))
        self.assertIsNone(explain(provider, AgentResponse(status="conversation", message="Hi"), "en"))
        self.assertIsNone(explain(provider, AgentResponse(status="completed", message="Done"), "en"))
        self.assertIsNone(explain(Mock(), AgentResponse(status="unsupported", message="No"), "en"))
        self.assertEqual(list(explain(provider, AgentResponse(status="unsupported", message="No"), "zh")), ["ok"])
        prompt, facts = provider.stream_text.call_args.args
        self.assertEqual(prompt, explanation_prompt("zh"))
        self.assertTrue(prompt.endswith("Response language: Chinese."))
        self.assertEqual(json.loads(facts)["status"], "unsupported")


class SelectionTests(unittest.TestCase):
    def test_dataset_and_algorithm_selections_are_applied_exactly(self):
        revised = apply_choices(plan(), choices(
            {"field": "database_label", "requested_value": None, "selected_values": ["default"]},
            {"field": "algorithm_id", "requested_value": None, "selected_values": ["descriptive_statistics_v1"]}))
        self.assertEqual((revised.database_label, revised.algorithm_id), ("default", "descriptive_statistics_v1"))
        self.assertEqual(revised.purpose, "selection test")

    def test_a_selected_candidate_replaces_only_the_unresolved_name(self):
        original = plan(sites=["hospital 1", "hosptial two"])
        revised = apply_choices(original, choices(
            {"field": "sites", "requested_value": "hosptial two", "selected_values": ["hospital 2"]}))
        self.assertEqual(revised.sites, ["hospital 1", "hospital 2"])
        self.assertEqual(original.sites, ["hospital 1", "hosptial two"])

    def test_missing_lists_are_filled_from_the_selection(self):
        revised = apply_choices(plan(sites=[]), choices(
            {"field": "sites", "requested_value": None, "selected_values": ["hospital 1", "hospital 2", "hospital 1"]}))
        self.assertEqual(revised.sites, ["hospital 1", "hospital 2"])

    def test_anything_that_is_not_a_plain_selection_goes_to_the_planner(self):
        for reply in ("use the clean dataset",
                      choices({"field": "database_label", "requested_value": None, "reply": "the clean one"}),
                      choices({"field": "parameters.bins", "requested_value": None, "selected_values": ["10"]}),
                      choices({"field": "sites", "requested_value": "unknown", "selected_values": ["hospital 2"]}),
                      choices({"field": "database_label", "requested_value": None, "selected_values": ["a", "b"]}),
                      json.dumps({"confirmed_choices": []})):
            with self.subTest(reply=reply):
                self.assertIsNone(apply_choices(plan(), reply))
        self.assertIsNone(selected_choices("not json"))


class SelectionUI:
    def __init__(self, inputs):
        self.inputs = iter(inputs)
        self.language = "en"
        self.last_response = None
        self.presented = []
        self.messages = []

    def header(self, *args):
        pass

    def say(self, text="", style=None):
        self.messages.append(text)

    def ask(self, pending=None):
        return next(self.inputs)

    @contextmanager
    def planning(self, planner):
        yield

    def present(self, response, planner, output, *, narrate=True):
        self.last_response = response
        self.presented.append(response)

    def review(self, prepared, output):
        return False

    def progress(self, text):
        pass


class SelectionStudyTests(unittest.TestCase):
    def test_a_menu_selection_completes_the_study_without_a_model_request(self):
        provider = Mock()
        provider.generate_json.side_effect = [plan().model_dump(), AssertionError("A selection needs no model")]
        selection = choices({"field": "database_label", "requested_value": None, "selected_values": ["default"]})
        ui = SelectionUI(["Describe age at diagnosis in hospital 1 and hospital 2", selection, "/exit"])
        with tempfile.TemporaryDirectory() as temporary:
            StudyChat(build_agent(), LLMResearchPlanner(provider), ui, StudySettings(output=Path(temporary))).run()
        clarification, ready = ui.presented
        self.assertEqual(clarification.status, "needs_clarification")
        self.assertEqual(ready.status, "ready")
        self.assertEqual(ready.prepared_task.task_spec.databases[0].label, "default")
        self.assertEqual(provider.generate_json.call_count, 1)


if __name__ == "__main__":
    unittest.main()
