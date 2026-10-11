"""Regression cases for intent, result, report and metadata boundaries."""

import io
import json
import os
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import federated_agent.cli as cli
from federated_agent.services.study import StudySettings, report_turn
from federated_agent.resources import RESOURCE_ROOT
from federated_agent.models import ExecutionRecord, ExecutionStatus, ResearchPlan
from federated_agent.planner import LLMResearchPlanner
from federated_agent.app import build_agent


def plan(**changes):
    return ResearchPlan(**(dict(analysis_type="descriptive_statistics", variables=["age at diagnosis"],
        sites=["hospital 1", "hospital 2"], database_label="default", purpose="Describe age.") | changes))


def planner_for(value):
    provider = Mock()
    provider.model = "offline-regression-fixture"
    provider.generate_json.return_value = value
    return LLMResearchPlanner(provider, retries=0)


class RequirementProtectionTests(unittest.TestCase):
    def test_omitted_filter_cannot_prepare(self):
        for request in (
            "Describe age in hospital 1 and hospital 2 using default, only for patients younger than 25.",
            "Describe age in hospital 1 and hospital 2 using default where age < 25.",
            "Describe age in hospital 1 and hospital 2 using default for patients diagnosed between 2010 and 2020.",
            "\u5206\u6790\u533b\u9662 1 \u548c\u533b\u9662 2 \u7684\u5e74\u9f84\uff0c\u4ec5\u9650\u5c0f\u4e8e 25 \u5c81\u7684\u60a3\u8005\u3002",
        ):
            with self.subTest(request=request):
                agent = build_agent()
                response = agent.respond(request, planner_for(plan().model_dump()))
                self.assertEqual(response.status, "unsupported")
                self.assertIsNone(response.prepared_task)
                self.assertTrue(response.research_plan.unsupported_requirements)

    def test_site_change_cannot_clear_previous_condition(self):
        previous = plan(unsupported_requirements=["age < 25"])
        response = build_agent().respond(
            "Change to hospital 2 and hospital 3; keep everything else.",
            planner_for({"changes": {"sites": ["hospital 2", "hospital 3"], "unsupported_requirements": []}}),
            previous_plan=previous,
        )
        self.assertEqual(response.status, "unsupported")
        self.assertIn("age < 25", response.research_plan.unsupported_requirements)

    def test_plain_request_still_prepares(self):
        response = build_agent().respond(
            "Describe age at diagnosis in hospital 1 and hospital 2 using default.", planner_for(plan().model_dump()))
        self.assertEqual(response.status, "ready")

    def test_hospital_only_is_not_a_patient_filter(self):
        request = "Describe age in only hospital 1 and hospital 2 using default."
        response = build_agent().respond(request, planner_for(plan().model_dump()))
        self.assertEqual(response.status, "ready")

    def test_direct_prepare_checks_stated_conditions(self):
        from federated_agent.errors import UnsupportedAnalysisError
        with self.assertRaises(UnsupportedAnalysisError):
            build_agent().prepare(plan(purpose="Describe age for patients younger than 25."))

    def test_model_cannot_hide_condition_by_rewriting_purpose(self):
        request = "Describe age in hospital 1 and hospital 2 using default for patients under 25."
        response = build_agent().respond(request, planner_for(plan(purpose="Describe everyone").model_dump()))
        self.assertEqual(response.status, "unsupported")

    def test_protected_condition_survives_restart_and_new_can_reset(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            agent = build_agent()
            planner = planner_for(plan().model_dump())
            args = SimpleNamespace(output=root, provider="local", backend="mock", timeout_seconds=5,
                                   confirm_synthetic_unencrypted_sandbox=False, resume=None, stream=False, plain=True)
            answers = iter(["Describe age in hospital 1 and hospital 2 using default, only patients under 25.", "/exit"])
            with redirect_stdout(io.StringIO()):
                cli.run_chat(agent, planner, args, read=lambda _: next(answers))
            session = next(root.glob("session-*"))
            args.resume = session.name
            planner.provider.generate_json.return_value = {"changes": {"sites": ["hospital 2", "hospital 3"], "unsupported_requirements": []}}
            answers = iter(["Change hospitals to hospital 2 and hospital 3.", "/exit"])
            with redirect_stdout(io.StringIO()):
                cli.run_chat(agent, planner, args, read=lambda _: next(answers))
            self.assertEqual(json.loads((session / "turn-002/response.json").read_text())["status"], "unsupported")
            planner.provider.generate_json.return_value = plan().model_dump()
            answers = iter(["/new", "Describe age in hospital 1 and hospital 2 using default.", "", "/exit"])
            with redirect_stdout(io.StringIO()):
                cli.run_chat(agent, planner, args, read=lambda _: next(answers))
            self.assertTrue((session / "turn-003/prepared_task.json").exists())
            self.assertFalse(list(session.glob("turn-*/mock_result.json")))


class ResultCompletenessTests(unittest.TestCase):
    def setUp(self):
        self.agent = build_agent()
        self.prepared = self.agent.prepare(plan(variables=["survival status"]))

    def test_absent_requested_table_is_not_suppression(self):
        parsed = self.agent.result_parser.parse({"numerical_general_statistics": {}})
        with self.assertRaises(ValueError):
            self.agent.result_parser.validate_requested(parsed, self.prepared.resolved_plan)

    def test_explicit_empty_categorical_table_remains_valid(self):
        parsed = self.agent.result_parser.parse({"categorical_general_statistics": {}})
        self.agent.result_parser.validate_requested(parsed, self.prepared.resolved_plan)

    def test_null_and_list_tables_are_invalid(self):
        for table in (None, [], "null", "[]"):
            with self.subTest(table=table), self.assertRaises(ValueError):
                self.agent.result_parser.parse({"categorical_general_statistics": table})

    def test_service_does_not_mark_missing_table_completed(self):
        self.agent.executor = object()
        record = ExecutionRecord(task_id="fixture-only", status=ExecutionStatus.completed,
                                 result={"numerical_general_statistics": {}})
        from federated_agent.errors import BackendExecutionError
        with self.assertRaises(BackendExecutionError):
            self.agent._completed_response(self.prepared, record, self.prepared.task_spec.catalog)

    def test_missing_numerical_table_also_fails(self):
        prepared = self.agent.prepare(plan())
        parsed = self.agent.result_parser.parse({"categorical_general_statistics": {}})
        with self.assertRaises(ValueError):
            self.agent.result_parser.validate_requested(parsed, prepared.resolved_plan)

    def test_categorical_rows_require_a_category_field(self):
        parsed = self.agent.result_parser.parse({"categorical_general_statistics": {
            "variable": {"0": "ncit:C25717"}, "count": {"0": 12}}})
        with self.assertRaises(ValueError):
            self.agent.result_parser.validate_requested(parsed, self.prepared.resolved_plan)


class ReportRecoveryTests(unittest.TestCase):
    def test_report_failure_keeps_cli_alive_and_can_rebuild_without_submission(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            agent = build_agent()
            planner = planner_for(plan().model_dump())
            args = SimpleNamespace(output=root, provider="local", backend="mock", timeout_seconds=5,
                                   confirm_synthetic_unencrypted_sandbox=False, resume=None, stream=False, plain=True)
            answers = iter(["Describe age in hospital 1 and hospital 2 using default.", "confirm", "/exit"])
            with patch("federated_agent.result_parser.ResultParser.write_html", side_effect=OSError("fixture write failure")), redirect_stdout(io.StringIO()):
                cli.run_chat(agent, planner, args, read=lambda _: next(answers))
            session = next(root.glob("session-*"))
            turn = session / "turn-001"
            response = json.loads((turn / "response.json").read_text())
            self.assertEqual(response["status"], "report_failed")
            self.assertEqual(response["submission_state"], "completed")
            self.assertEqual(response["report_status"], "failed")
            self.assertTrue(response["execution"]["result"])
            self.assertEqual(json.loads((turn / "run_manifest.json").read_text())["report_status"], "failed")
            args.resume = session.name
            answers = iter(["/report turn-001", "/exit"])
            planner.provider.generate_json.reset_mock()
            with patch.object(agent.executor, "submit", side_effect=AssertionError("No resubmission")), redirect_stdout(io.StringIO()):
                cli.run_chat(agent, planner, args, read=lambda _: next(answers))
            planner.provider.generate_json.assert_not_called()
            self.assertTrue((turn / "result.html").exists())
            self.assertEqual(json.loads((turn / "response.json").read_text())["report_status"], "completed")

    def test_local_regeneration_rechecks_policy_and_rejects_paths(self):
        from federated_agent.models import PolicyDecision
        from federated_agent.sessions import SessionError, SessionStore
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            agent = build_agent()
            planner = planner_for(plan().model_dump())
            args = SimpleNamespace(output=root, provider="local", backend="mock", timeout_seconds=5,
                                   confirm_synthetic_unencrypted_sandbox=False, resume=None, stream=False, plain=True)
            answers = iter(["Describe age in hospital 1 and hospital 2 using default.", "confirm", "/exit"])
            with redirect_stdout(io.StringIO()):
                cli.run_chat(agent, planner, args, read=lambda _: next(answers))
            session = next(root.glob("session-*"))
            store = SessionStore.open(root, session.name, "mock")
            try:
                before = (session / "turn-001/result.html").read_bytes()
                agent.policy_checker = Mock()
                agent.policy_checker.evaluate.return_value = PolicyDecision(allowed=False)
                with self.assertRaises(SessionError):
                    report_turn(agent, planner, StudySettings.from_args(args), store, "turn-001", Mock())
                self.assertEqual((session / "turn-001/result.html").read_bytes(), before)
                with self.assertRaises(SessionError):
                    report_turn(agent, planner, StudySettings.from_args(args), store, "../other", Mock())
            finally:
                store.close()


class LocalMetadataRefreshTests(unittest.TestCase):
    def with_schema(self, root):
        (root / "catalogs").mkdir()
        schema_path = root / "catalogs/AYA_cancer_schema.jsonld"
        schema_path.write_bytes((RESOURCE_ROOT / "catalogs/AYA_cancer_schema.jsonld").read_bytes())
        agent = build_agent(semantic_schema_path=schema_path)
        return agent, schema_path

    def test_local_semantic_update_is_visible_without_restart(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "catalogs").mkdir()
            path = root / "catalogs/AYA_cancer_schema.jsonld"
            document = {"version": "test-1", "schema": {"variables": {
                "fixture_age": {"class": "fixture:Age", "dataType": "continuous"}}}}
            path.write_text(json.dumps(document), encoding="utf-8")
            agent = build_agent(semantic_schema_path=path)
            updated = deepcopy(document)
            updated["schema"]["variables"]["fixture_marker"] = {"class": "fixture:Marker", "dataType": "continuous"}
            path.write_text(json.dumps(updated), encoding="utf-8")
            agent.refresh_metadata()
            self.assertEqual(agent.semantic_catalog.resolve("fixture marker").class_id, "fixture:Marker")

    def test_semantic_update_invalidates_confirmation_even_with_same_version(self):
        with TemporaryDirectory() as temporary:
            agent, path = self.with_schema(Path(temporary))
            prepared = agent.prepare(plan())
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["schema"]["variables"]["fixture_marker"] = {"class": "fixture:Marker", "dataType": "continuous"}
            path.write_text(json.dumps(raw), encoding="utf-8")
            agent.executor.submit = Mock(side_effect=AssertionError("Stale confirmation cannot submit"))
            response = agent.execute_response(prepared, prepared.confirmation_id)
            self.assertEqual(response.status, "invalid_request")
            self.assertEqual(response.issues[0].code, "confirmation_invalid")

    def test_same_local_snapshot_is_used_during_planning(self):
        with TemporaryDirectory() as temporary:
            agent, path = self.with_schema(Path(temporary))
            original = json.loads(path.read_text(encoding="utf-8"))
            before = agent.semantic_catalog.resolve("age at diagnosis").class_id
            planner = planner_for(plan().model_dump())
            def change_during_model(*_):
                changed = deepcopy(original)
                changed["schema"]["variables"]["age_at_initial_diagnosis"]["class"] = "fixture:UpdatedAge"
                path.write_text(json.dumps(changed), encoding="utf-8")
                return plan().model_dump()
            planner.provider.generate_json.side_effect = change_during_model
            response = agent.respond("Describe age in hospital 1 and hospital 2 using default.", planner)
            self.assertEqual(response.prepared_task.resolved_plan.variables[0].class_id, before)
            agent.refresh_metadata()
            self.assertEqual(agent.semantic_catalog.resolve("age at diagnosis").class_id, "fixture:UpdatedAge")

    def test_invalid_local_schema_blocks_without_calling_model(self):
        with TemporaryDirectory() as temporary:
            agent, path = self.with_schema(Path(temporary))
            path.write_text("{}", encoding="utf-8")
            planner = planner_for(plan().model_dump())
            response = agent.respond("Describe age in hospital 1 and hospital 2 using default.", planner)
            self.assertEqual(response.status, "preflight_error")
            planner.provider.generate_json.assert_not_called()


class CheckpointTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows sharing conflicts")
    def test_short_local_replace_conflict_is_retried_without_losing_checkpoint(self):
        from federated_agent.storage import write_json
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.json"
            write_json(path, {"revision": 1})
            replace = Path.replace
            attempts = []
            def busy_then_success(source, destination):
                attempts.append(1)
                if len(attempts) < 3:
                    error = PermissionError("fixture sharing conflict")
                    error.winerror = 5
                    raise error
                return replace(source, destination)
            with patch.object(Path, "replace", busy_then_success), patch("federated_agent.storage.time.sleep"):
                write_json(path, {"revision": 2})
            self.assertEqual(json.loads(path.read_text()), {"revision": 2})
            self.assertEqual(len(attempts), 3)
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    @unittest.skipUnless(os.name == "nt", "Windows sharing conflicts")
    def test_permanent_denial_is_bounded_and_keeps_old_checkpoint(self):
        from federated_agent.storage import write_json
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.json"
            write_json(path, {"revision": 1})
            error = PermissionError("fixture permanent denial")
            error.winerror = 5
            with patch.object(Path, "replace", side_effect=error) as replace, patch("federated_agent.storage.time.sleep"):
                with self.assertRaises(PermissionError):
                    write_json(path, {"revision": 2})
            self.assertEqual(replace.call_count, 6)
            self.assertEqual(json.loads(path.read_text()), {"revision": 1})
            self.assertEqual(list(path.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
