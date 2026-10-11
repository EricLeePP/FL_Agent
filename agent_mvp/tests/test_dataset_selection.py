"""A dataset is a researcher choice, never an implicit deployment default."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rich.console import Console

from federated_agent.resources import RESOURCE_ROOT
from federated_agent.cli import run_chat
from federated_agent.errors import ClarificationRequiredError, TaskValidationError
from federated_agent.models import ResearchPlan
from federated_agent.planner import DemoResearchPlanner, LLMResearchPlanner
from federated_agent.sessions import SessionStore
from federated_agent.terminal_ui import TerminalUI
from federated_agent.app import build_agent
from federated_agent.storage import read_json, write_json


REQUEST = "Describe age at diagnosis and survival status in hospital 1 and hospital 2"


def plan(label=None, **changes):
    return ResearchPlan(**(dict(analysis_type="descriptive_statistics",
        variables=["age at diagnosis", "survival status"], sites=["hospital 1", "hospital 2"],
        database_label=label, purpose=REQUEST) | changes))


class DatasetSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.deployment = self.root / "deployment.json"
        self.profile = read_json(RESOURCE_ROOT / "config/ubuntu_sandbox.json")
        write_json(self.deployment, self.profile)
        self.agent = build_agent(self.deployment)
        self.provider = Mock()
        self.provider.model = "test-local"
        self.planner = LLMResearchPlanner(self.provider, retries=0)

    def response(self, value, previous=None):
        self.provider.generate_json.return_value = value
        return self.agent.respond("Researcher request", self.planner, previous_plan=previous)

    def dataset_question(self, response):
        self.assertEqual(response.status, "needs_clarification")
        self.assertIsNone(response.prepared_task)
        return next(q for q in response.clarification.questions if q.field == "database_label")

    def chat(self, replies, resume=None):
        args = SimpleNamespace(output=self.root / "runs", provider="local", backend="mock",
                               stream=False, plain=True, resume=resume)
        entries = iter(replies)
        output = io.StringIO()
        with redirect_stdout(output):
            run_chat(self.agent, self.planner, args, read=lambda _: next(entries))
        return output.getvalue()

    def test_missing_empty_and_blank_dataset_stop_before_compile_policy_or_execution(self):
        self.agent.task_compiler.compile = Mock(side_effect=AssertionError("No compile"))
        self.agent.policy_checker.evaluate = Mock(side_effect=AssertionError("No policy-ready task"))
        self.agent.executor.submit = Mock(side_effect=AssertionError("No submission"))
        for label in (None, "", "   "):
            with self.subTest(label=label):
                response = self.response(plan(label).model_dump())
                question = self.dataset_question(response)
                self.assertIsNone(response.clarification.research_plan.database_label)
                self.assertIn("synthetic_clean", question.candidates)
                self.assertIn("missing_database", [item.code for item in response.issues])
        self.agent.executor.submit.assert_not_called()

    def test_single_candidate_and_deployment_default_still_need_selection(self):
        agent = build_agent()
        response = agent.respond(REQUEST, DemoResearchPlanner())
        self.assertEqual(self.dataset_question(response).candidates, ["default"])
        self.assertIsNone(response.clarification.research_plan.database_label)

    def test_offline_demo_preserves_explicit_generic_and_unknown_labels(self):
        response = build_agent().respond(REQUEST + " using default", DemoResearchPlanner())
        self.assertEqual(response.status, "ready")
        response = self.agent.respond(REQUEST + " using database synthetic_nonexistent", DemoResearchPlanner())
        self.assertEqual(self.dataset_question(response).requested_value, "synthetic_nonexistent")

    def test_interactive_dataset_choice_is_single_select_and_requires_an_answer(self):
        pending = self.response(plan().model_dump())
        ui = TerminalUI(plain=True, console=Console(file=io.StringIO()))
        ui.interactive = True
        with patch("federated_agent.terminal_ui.choice_application") as menu:
            menu.return_value.run.return_value = "1"
            answer = json.loads(ui.ask(pending))
            self.assertFalse(menu.call_args.kwargs["multiple"])
            self.assertEqual(answer["confirmed_choices"][0]["selected_values"], ["synthetic_missing"])
            menu.return_value.run.return_value = None
            self.assertEqual(ui.ask(pending), "/cancel")

    def test_compiler_itself_cannot_fall_back_to_profile_or_resolved_site_label(self):
        prepared = self.agent.prepare(plan("synthetic_clean"))
        algorithm = self.agent.algorithm_catalog.snapshot().select("descriptive_statistics")
        for label in (None, "", " "):
            resolved = prepared.resolved_plan.model_copy(update={"database_label": label})
            with self.assertRaises(TaskValidationError) as caught:
                self.agent.task_compiler.compile(resolved, algorithm)
            self.assertEqual(caught.exception.code, "missing_database")

    def test_direct_prepare_requires_selection(self):
        with self.assertRaises(ClarificationRequiredError) as caught:
            self.agent.prepare(plan())
        self.assertIn("database_label", [q.field for q in caught.exception.request.questions])

    def test_site_resolution_and_model_context_do_not_offer_an_implicit_default(self):
        self.assertIsNone(self.agent.deployment_registry.resolve("hospital 1").database_label)
        context = self.agent.planner_context()
        self.assertNotIn("default_database_label", context)
        self.assertTrue(context["dataset_selection_required"])

    def test_candidates_intersect_selected_sites_central_site_and_algorithm(self):
        self.profile["sites"][0]["database_labels"].remove("synthetic_rare")
        self.profile["sites"][1]["database_labels"].remove("synthetic_missing")
        write_json(self.deployment, self.profile)
        raw = read_json(RESOURCE_ROOT / "catalogs/algorithms.json")
        for scope in raw["algorithms"][0]["availability"]:
            if scope["deployment"] == self.profile["name"]:
                scope["database_labels"].remove("synthetic_small")
        catalog = self.root / "algorithms.json"
        write_json(catalog, raw)
        self.agent = build_agent(self.deployment, algorithm_catalog_path=catalog)
        response = self.response(plan(sites=["hospital 2", "hospital 3"]).model_dump())
        self.assertEqual(self.dataset_question(response).candidates, ["synthetic_clean"])

    def test_no_shared_candidate_explains_constraint_without_selecting_default(self):
        raw = read_json(RESOURCE_ROOT / "catalogs/algorithms.json")
        for scope in raw["algorithms"][0]["availability"]:
            if scope["deployment"] == self.profile["name"]:
                scope["database_labels"] = ["not_registered_at_any_site"]
        catalog = self.root / "algorithms.json"
        write_json(catalog, raw)
        self.agent = build_agent(self.deployment, algorithm_catalog_path=catalog)
        question = self.dataset_question(self.response(plan().model_dump()))
        self.assertEqual(question.candidates, [])
        self.assertIn("No common dataset", question.question)

    def test_unknown_sites_do_not_produce_misleading_dataset_options(self):
        question = self.dataset_question(self.response(plan(sites=["unknown hospital"]).model_dump()))
        self.assertEqual(question.candidates, [])

    def test_explicit_selection_binds_payload_and_provenance_without_autoexecution(self):
        self.agent.executor.submit = Mock()
        response = self.response(plan("synthetic_missing").model_dump())
        self.assertEqual(response.status, "ready")
        self.assertEqual(response.prepared_task.task_spec.databases[0].label, "synthetic_missing")
        self.assertTrue(all(site.database_label == "synthetic_missing" for site in response.prepared_task.resolved_plan.sites))
        source = next(item.source for item in response.prepared_task.task_spec.provenance if item.field == "databases")
        self.assertEqual(source.uri, "research_plan.database_label")
        self.agent.executor.submit.assert_not_called()

    def test_omitted_followup_retains_selection_and_displays_its_source(self):
        response = self.response({"changes": {"variables": ["age at diagnosis"]}}, plan("synthetic_missing"))
        self.assertEqual(response.status, "ready")
        self.assertEqual(response.prepared_task.task_spec.databases[0].label, "synthetic_missing")
        self.assertIn("dataset_retained", [item.code for item in response.issues])
        output = io.StringIO()
        ui = TerminalUI(plain=True, stream=False, console=Console(file=output))
        ui.present(response, self.planner, self.root)
        self.assertIn("Dataset retained from active study: synthetic_missing", output.getvalue())

    def test_explicit_null_clears_selection_and_preserves_other_requirements(self):
        previous = plan("synthetic_clean")
        response = self.response({"changes": {"database_label": None}}, previous)
        self.dataset_question(response)
        updated = response.clarification.research_plan
        self.assertIsNone(updated.database_label)
        self.assertEqual(updated.sites, previous.sites)
        self.assertEqual(updated.variables, previous.variables)
        self.assertEqual(updated.unsupported_requirements, previous.unsupported_requirements)

    def test_clear_does_not_erase_or_hide_known_unsupported_requirements(self):
        previous = plan("synthetic_clean", unsupported_requirements=["Only patients under 25"])
        response = self.response({"changes": {"database_label": None}}, previous)
        self.assertEqual(response.status, "unsupported")
        self.assertIsNone(response.research_plan.database_label)
        self.assertEqual(response.research_plan.unsupported_requirements, previous.unsupported_requirements)
        self.assertIsNone(response.prepared_task)

    def test_old_prepared_default_cannot_be_newly_submitted_without_selection(self):
        prepared = self.agent.prepare(plan("synthetic_clean"))
        prepared.research_plan.database_label = None
        self.agent.executor.submit = Mock(side_effect=AssertionError("No submission"))
        response = self.agent.execute_response(prepared, prepared.confirmation_id)
        self.assertEqual(response.issues[0].code, "missing_database")
        self.assertEqual(response.submission_state, "not_submitted")
        self.agent.executor.submit.assert_not_called()

    def test_unknown_dataset_never_falls_back(self):
        response = self.response(plan("nonexistent_dataset").model_dump())
        self.assertEqual(self.dataset_question(response).requested_value, "nonexistent_dataset")
        self.assertEqual(response.clarification.research_plan.database_label, "nonexistent_dataset")

    def test_cli_new_clears_previous_selection_and_writes_no_task_for_missing_dataset(self):
        self.provider.generate_json.side_effect = [plan("synthetic_clean").model_dump(), plan().model_dump()]
        self.agent.executor.submit = Mock()
        text = self.chat([REQUEST + " using synthetic_clean", "", "/new", REQUEST, "/exit"])
        session = next((self.root / "runs").glob("session-*"))
        self.assertEqual(read_json(session / "turn-002/response.json")["status"], "needs_clarification")
        self.assertFalse((session / "turn-002/run_vantage6_task.py").exists())
        self.assertFalse((session / "turn-002/prepared_task.json").exists())
        self.assertIsNone(read_json(session / "session.json")["current_plan"]["database_label"])
        self.assertIn("Which dataset", text)
        self.agent.executor.submit.assert_not_called()

    def test_clarification_resume_select_clear_and_reselect_then_confirm_mock(self):
        self.provider.generate_json.return_value = plan().model_dump()
        self.chat([REQUEST, "/exit"])
        session = next((self.root / "runs").glob("session-*"))
        self.provider.generate_json.side_effect = [
            {"changes": {"database_label": "synthetic_clean"}},
            {"changes": {"database_label": None}},
            {"changes": {"database_label": "synthetic_missing"}},
        ]
        self.chat(["Use synthetic_clean", "", "Clear the dataset selection", "synthetic_missing", "confirm", "/exit"], resume=session.name)
        self.assertEqual(read_json(session / "turn-003/response.json")["status"], "needs_clarification")
        self.assertFalse((session / "turn-003/run_vantage6_task.py").exists())
        self.assertEqual(read_json(session / "turn-004/prepared_task.json")["task_spec"]["databases"][0]["label"], "synthetic_missing")
        self.assertEqual(read_json(session / "turn-004/response.json")["status"], "completed")
        self.assertEqual(len(list(session.glob("turn-*/researcher_confirmation.json"))), 1)

    def test_selected_dataset_survives_new_checkpoint_and_changed_deployment_default(self):
        self.provider.generate_json.return_value = plan("synthetic_missing").model_dump()
        self.chat([REQUEST + " using synthetic_missing", "", "/exit"])
        session = next((self.root / "runs").glob("session-*"))
        self.profile["database_label"] = "synthetic_small"
        write_json(self.deployment, self.profile)
        with_store = SessionStore.open(session.parent, session.name, "mock")
        try:
            self.assertEqual(with_store.state.current_plan.database_label, "synthetic_missing")
        finally:
            with_store.close()
        self.chat(["/review", "", "/exit"], resume=session.name)
        self.assertEqual(read_json(session / "turn-002/prepared_task.json")["task_spec"]["databases"][0]["label"], "synthetic_missing")



if __name__ == "__main__":
    unittest.main()
