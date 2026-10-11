import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock
from urllib.error import HTTPError

from federated_agent.data_availability import FileDataAvailabilityChecker
from federated_agent.executor import ClassicVantage6Executor
from federated_agent.feedback import render_researcher_response
from federated_agent.models import ExecutionRecord, ExecutionStatus, PolicyDecision, ResearchPlan
from federated_agent.script_renderer import render_vantage6_script
from federated_agent.app import build_agent
from federated_agent.resources import ALGORITHM_CATALOG


ROOT = Path(__file__).resolve().parents[1]


def plan(**changes):
    values = dict(analysis_type="descriptive_statistics", variables=["age at diagnosis"],
                  sites=["hospital 1", "hospital 2"], database_label="default", purpose="Describe the synthetic population")
    values.update(changes)
    return ResearchPlan(**values)


class FixedPlanner:
    def __init__(self, research_plan, callback=None):
        self.plan, self.callback = research_plan, callback

    def create_plan(self, request, catalog):
        if self.callback:
            self.callback()
        return self.plan


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.catalog_path = self.directory / "algorithms.json"
        self.catalog_raw = json.loads(ALGORITHM_CATALOG.read_text(encoding="utf-8"))
        self.write_catalog()
        self.agent = build_agent(algorithm_catalog_path=self.catalog_path)

    def write_catalog(self):
        self.catalog_path.write_text(json.dumps(self.catalog_raw), encoding="utf-8")

    def respond(self, **changes):
        return self.agent.respond("Test request", FixedPlanner(plan(**changes)))

    def metadata(self, *, state="available", expired=False):
        now = datetime.now(timezone.utc)
        raw = dict(schema_version="1.0", version="test-only-1", deployment="local-demo-sandbox",
                   observed_at=(now - timedelta(days=2)).isoformat(),
                   expires_at=(now + timedelta(days=-1 if expired else 1)).isoformat(),
                   sites=[dict(organisation_id=i, database_label="default", variables={"ncit:C156420": state}) for i in (1, 2)])
        path = self.directory / "data.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        self.agent.data_checker = FileDataAvailabilityChecker(path)
        return path, raw

    def test_unknown_variable_is_actionable_and_does_not_prepare(self):
        response = self.respond(variables=["imaginary_marker_xyz"])
        self.assertEqual(response.status, "needs_clarification")
        self.assertIsNone(response.prepared_task)
        self.assertEqual(response.issues[0].requested_value, "imaginary_marker_xyz")
        self.assertEqual(response.issues[0].code, "unknown_variable")
        self.assertTrue(response.issues[0].next_steps)

    def test_fuzzy_match_requires_confirmation_even_for_one_close_candidate(self):
        response = self.respond(variables=["age at diagnozis"])
        self.assertEqual(response.status, "needs_clarification")
        self.assertIn("age_at_initial_diagnosis", response.issues[0].candidates)
        self.assertIsNone(response.prepared_task)

    def test_unknown_hospital_never_proposes_a_different_hospital_implicitly(self):
        response = self.respond(sites=["hospital 99"])
        self.assertEqual(response.issues[0].code, "unknown_site")
        self.assertEqual(response.suggested_algorithms, [])
        self.assertIn("hospital 1", response.issues[0].candidates)

    def test_missing_sites_options_require_site_selection(self):
        response = self.respond(sites=[])
        self.assertEqual(response.issues[0].code, "missing_sites")
        self.assertEqual(response.suggested_algorithms[0].scope, "select_sites_first")

    def test_unsupported_analysis_proposes_only_catalog_backed_non_equivalent_options(self):
        response = self.respond(analysis_type="cox", algorithm_id="invented_cox")
        self.assertEqual(response.status, "unsupported")
        option = response.suggested_algorithms[0]
        self.assertEqual(option.algorithm_id, "strongaya_descriptive_statistics")
        self.assertFalse(option.replaces_original_request)
        self.assertEqual(option.authorization, "not_verified")
        self.assertEqual(option.catalog.sha256, response.catalog.sha256)
        self.assertEqual(option.capabilities, self.catalog_raw["algorithms"][0]["capabilities"])
        self.assertIsNone(response.prepared_task)

    def test_unsupported_filter_preserved_in_feedback(self):
        response = self.respond(unsupported_requirements=["only patients under 25"])
        self.assertEqual(response.status, "unsupported")
        self.assertEqual(response.issues[0].code, "unmet_requirements")
        self.assertEqual(response.issues[0].requested_value, "only patients under 25")
        self.assertIn("not drop filters", response.issues[0].next_steps[0])

    def test_disabled_algorithm_not_recommended_on_next_turn(self):
        self.assertTrue(self.respond(analysis_type="cox").suggested_algorithms)
        self.catalog_raw["algorithms"][0]["status"] = "disabled"
        self.write_catalog()
        self.assertEqual(self.respond(analysis_type="cox").suggested_algorithms, [])

    def test_suggestions_share_planner_snapshot(self):
        def edit_catalog():
            self.catalog_raw["algorithms"][0]["status"] = "disabled"
            self.write_catalog()
        response = self.agent.respond("Cox", FixedPlanner(plan(analysis_type="cox"), callback=edit_catalog))
        self.assertTrue(response.suggested_algorithms)
        self.assertEqual(response.catalog.sha256, response.suggested_algorithms[0].catalog.sha256)
        self.assertFalse(self.respond(analysis_type="cox").suggested_algorithms)

    def test_scope_version_adapter_and_database_filter_suggestions(self):
        original = json.loads(json.dumps(self.catalog_raw))
        changes = [
            lambda item: item.update(vantage6_version="5.0"),
            lambda item: item["execution"].update(adapter="unknown"),
            lambda item: item["availability"][0].update(organisation_ids=[1]),
            lambda item: item["availability"][0].update(database_labels=["other"]),
            lambda item: item["availability"][0].update(deployment="other"),
        ]
        for change in changes:
            self.catalog_raw = json.loads(json.dumps(original))
            change(self.catalog_raw["algorithms"][0])
            self.write_catalog()
            self.assertEqual(self.respond(analysis_type="cox").suggested_algorithms, [])

    def test_declared_scope_failure_has_specific_reason(self):
        self.catalog_raw["algorithms"][0]["availability"][0]["organisation_ids"] = [1]
        self.write_catalog()
        response = self.respond()
        self.assertEqual(response.status, "invalid_request")
        self.assertEqual(response.issues[0].code, "algorithm_site_unavailable")

    def test_schema_validation_explains_permitted_parameters(self):
        response = self.respond(parameters={"arbitrary_parameter": "secret-test-value"})
        self.assertEqual(response.issues[0].code, "invalid_parameters")
        self.assertEqual(response.issues[0].constraint, "Allowed parameter names: none")
        self.assertNotIn("secret-test-value", response.model_dump_json())

    def test_missing_catalog_returns_no_cached_suggestions_or_raw_errors(self):
        self.catalog_path.unlink()
        response = self.respond()
        self.assertEqual(response.status, "catalog_error")
        self.assertEqual(response.suggested_algorithms, [])
        self.assertNotIn(str(self.directory), response.message)

    def test_ready_does_not_claim_data_or_permission_verified(self):
        response = self.respond()
        codes = {item.code for item in response.issues}
        self.assertTrue({"data_unverified", "authorization_unverified", "runtime_unverified"} <= codes)
        self.assertEqual(response.submission_state, "not_submitted")

    def test_metadata_unavailable_names_site_and_variable_but_stops(self):
        self.metadata(state="unavailable")
        response = self.respond()
        self.assertEqual(response.status, "data_unavailable")
        self.assertIsNone(response.prepared_task)
        self.assertEqual(response.issues[0].field, "hospital 1.age_at_initial_diagnosis")
        self.assertEqual(response.issues[0].source.kind, "data_availability_snapshot")
        self.assertEqual(response.suggested_algorithms, [])

    def test_metadata_omission_means_unknown_not_unavailable(self):
        self.metadata()
        response = self.respond(variables=["survival status"])
        self.assertEqual(response.status, "ready")
        self.assertIn("data_not_listed", {item.code for item in response.issues})

    def test_expired_metadata_cannot_make_known_availability_claim(self):
        self.metadata(state="unavailable", expired=True)
        response = self.respond()
        self.assertEqual(response.status, "ready")
        self.assertIn("data_metadata_expired", {item.code for item in response.issues})
        self.assertTrue(all(check.state == "unknown" for check in response.prepared_task.data_checks))

    def test_expired_managed_evidence_blocks_execution(self):
        self.metadata(expired=True)
        prepared = self.respond().prepared_task
        response = self.agent.execute_response(prepared, prepared.confirmation_id)
        self.assertEqual(response.issues[0].code, "data_verification_required")
        self.assertFalse(self.agent.executor._tasks)

    def test_valid_managed_metadata_can_execute_mock(self):
        self.metadata()
        prepared = self.respond().prepared_task
        response = self.agent.execute_response(prepared, prepared.confirmation_id)
        self.assertEqual(response.status, "completed")
        self.assertIn("no real", response.message)

    def test_wrong_deployment_or_duplicate_declarations_are_rejected(self):
        path, raw = self.metadata()
        raw["deployment"] = "other"
        path.write_text(json.dumps(raw), encoding="utf-8")
        self.assertEqual(self.respond().status, "preflight_error")
        raw["deployment"] = "local-demo-sandbox"
        raw["sites"].append(raw["sites"][0])
        path.write_text(json.dumps(raw), encoding="utf-8")
        self.assertEqual(self.respond().status, "preflight_error")

    def test_invalid_metadata_blocks_instead_of_using_cached_copy(self):
        path, _ = self.metadata()
        self.assertEqual(self.respond().status, "ready")
        path.write_text("not json", encoding="utf-8")
        response = self.respond()
        self.assertEqual(response.status, "preflight_error")
        self.assertIsNone(response.prepared_task)

    def test_changed_data_availability_rechecked_before_submit(self):
        path, raw = self.metadata()
        prepared = self.respond().prepared_task
        raw["sites"][1]["variables"]["ncit:C156420"] = "unavailable"
        path.write_text(json.dumps(raw), encoding="utf-8")
        response = self.agent.execute_response(prepared, prepared.confirmation_id)
        self.assertEqual(response.status, "data_unavailable")
        self.assertFalse(self.agent.executor._tasks)

    def test_deny_is_sanitized_and_happens_before_data_checks(self):
        self.agent.policy_checker = Mock()
        self.agent.policy_checker.evaluate.return_value = PolicyDecision(allowed=False, reasons=["secret cohort and internal URL"])
        self.agent.data_checker = Mock()
        response = self.respond()
        self.assertEqual(response.status, "permission_denied")
        self.assertIsNone(response.prepared_task)
        self.assertIsNone(response.catalog_document)
        self.assertFalse(response.suggested_algorithms)
        self.assertNotIn("secret cohort", response.model_dump_json())
        self.agent.data_checker.check.assert_not_called()

    def test_authorization_failure_is_not_permission_denied(self):
        self.agent.policy_checker = Mock()
        self.agent.policy_checker.evaluate.side_effect = RuntimeError("sensitive URL")
        response = self.respond()
        self.assertEqual(response.status, "preflight_error")
        self.assertNotIn("sensitive URL", response.model_dump_json())

    def test_revoked_permission_blocks_prepared_task(self):
        prepared = self.respond().prepared_task
        self.agent.policy_checker = Mock()
        self.agent.policy_checker.evaluate.return_value = PolicyDecision(allowed=False)
        response = self.agent.execute_response(prepared, prepared.confirmation_id)
        self.assertEqual(response.status, "permission_denied")
        self.assertFalse(self.agent.executor._tasks)

    def test_managed_export_cannot_bypass_configured_checks(self):
        self.metadata()
        prepared = self.respond().prepared_task
        self.assertTrue(prepared.task_spec.requires_managed_execution)
        printed = io.StringIO()
        with self.assertRaisesRegex(SystemExit, "requires fresh data/authorization"), redirect_stdout(printed):
            exec(compile(render_vantage6_script(prepared.task_spec), "generated.py", "exec"), {})
        self.assertEqual(json.loads(printed.getvalue()), json.loads(json.dumps(prepared.task_spec.to_client_payload())))

    def test_runtime_http_denial_is_specific_but_does_not_leak_body(self):
        prepared = self.respond().prepared_task
        executor = Mock()
        executor.submit.side_effect = HTTPError("https://secret.invalid/token", 403, "private resource", {}, None)
        self.agent.executor = executor
        response = self.agent.execute_response(prepared, prepared.confirmation_id)
        self.assertEqual(response.issues[0].code, "backend_forbidden")
        self.assertEqual(response.submission_state, "not_submitted")
        self.assertNotIn("secret.invalid", response.model_dump_json())
        self.assertFalse(response.suggested_algorithms)
        executor.wait_for_results.assert_not_called()

    def test_submit_timeout_is_uncertain_and_not_retried(self):
        prepared = self.respond().prepared_task
        executor = Mock()
        executor.submit.side_effect = TimeoutError("internal endpoint")
        self.agent.executor = executor
        response = self.agent.execute_response(prepared, prepared.confirmation_id)
        self.assertEqual(response.submission_state, "unknown")
        self.assertEqual(response.issues[0].code, "backend_timeout")
        self.assertIn("before retrying", response.issues[0].next_steps[0])
        self.assertEqual(executor.submit.call_count, 1)

    def test_result_timeout_retains_task_id_without_claiming_remote_failure(self):
        prepared = self.respond().prepared_task
        executor = Mock()
        executor.submit.return_value = ExecutionRecord(task_id="77", status=ExecutionStatus.submitted)
        executor.wait_for_results.side_effect = TimeoutError()
        self.agent.executor = executor
        response = self.agent.execute_response(prepared, prepared.confirmation_id)
        self.assertEqual(response.submission_state, "submitted")
        self.assertEqual(response.execution.task_id, "77")
        self.assertEqual(response.execution.status, ExecutionStatus.submitted)

    def test_failed_execution_is_not_presented_as_completed(self):
        prepared = self.respond().prepared_task
        executor = Mock()
        executor.submit.return_value = ExecutionRecord(task_id="77", status=ExecutionStatus.submitted)
        executor.wait_for_results.return_value = ExecutionRecord(task_id="77", status=ExecutionStatus.failed, error="private node log")
        self.agent.executor = executor
        response = self.agent.execute_response(prepared, prepared.confirmation_id)
        self.assertEqual(response.submission_state, "failed")
        self.assertNotIn("private node log", response.model_dump_json())
        self.assertEqual(response.issues[0].code, "backend_task_failed")

    def test_classic_adapter_recognizes_nested_failure_and_drops_error_body(self):
        client = Mock()
        client.task.create.return_value = {"id": 42}
        client.result.from_task.return_value = {"data": [{"result": '{"error":"private node log"}'}]}
        executor = ClassicVantage6Executor(client)
        prepared = self.respond().prepared_task
        submitted = executor.submit(prepared.task_spec)
        result = executor.wait_for_results(submitted.task_id)
        self.assertEqual(result.status, ExecutionStatus.failed)
        self.assertIsNone(result.result)

    def test_changed_catalog_response_has_current_source(self):
        prepared = self.respond().prepared_task
        self.catalog_raw["catalog_version"] = "changed"
        self.write_catalog()
        response = self.agent.execute_response(prepared, prepared.confirmation_id)
        self.assertEqual(response.issues[0].code, "catalog_changed")
        self.assertEqual(response.catalog.catalog_version, "changed")

    def test_mock_success_explicitly_says_not_real(self):
        prepared = self.respond().prepared_task
        response = self.agent.execute_response(prepared, prepared.confirmation_id)
        self.assertEqual(response.status, "completed")
        self.assertIn("no real Vantage6", response.message)

    def test_readable_response_and_cli_artifact(self):
        response = self.respond(analysis_type="cox")
        text = render_researcher_response(response)
        self.assertIn("Catalog Options", text)
        self.assertIn("not automatic replacements", text)
        self.assertIn(response.catalog.sha256, text)
        output = self.directory / "cli"
        result = subprocess.run([sys.executable, "-m", "federated_agent", "--provider", "demo",
                                 "--request", "Run a Cox model in hospital 1 and hospital 2",
                                 "--output", str(output)], cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertTrue((output / "researcher_response.md").exists())
        self.assertFalse((output / "run_vantage6_task.py").exists())
        artifact = json.loads((output / "response.json").read_text(encoding="utf-8"))
        self.assertTrue(artifact["suggested_algorithms"])
        self.assertIn("Cox", artifact["issues"][0]["requested_value"])


if __name__ == "__main__":
    unittest.main()
