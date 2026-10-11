import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from federated_agent.resources import RESOURCE_ROOT, SOURCE_CHECKOUT
from federated_agent.cli import run_chat
from federated_agent.errors import BackendExecutionError
from federated_agent.models import ResearchPlan
from federated_agent.planner import LLMResearchPlanner
from federated_agent.adapters.vantage6 import create_client
from federated_agent.app import build_agent
from federated_agent.runs.executor import SandboxVantage6Executor
from federated_agent.runs.worker import run_worker
from federated_agent.storage import read_json, write_json


RAW_FIXTURE = Path(__file__).parent / "fixtures/vantage6_s02_result.json"
try:
    HAS_SDK = version("vantage6-client") == "4.15.0"
except PackageNotFoundError:
    HAS_SDK = False


def plan(**changes):
    values = dict(analysis_type="descriptive_statistics", variables=["age at diagnosis"],
                  sites=["hospital 1", "hospital 2"], database_label="synthetic_clean", purpose="Describe the synthetic data")
    values.update(changes)
    return ResearchPlan(**values)


class FixedPlanner:
    def __init__(self, value):
        self.value = value

    def create_plan(self, request, catalog):
        return self.value


class LiveCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        clock = patch("federated_agent.data_availability.datetime")
        mocked_clock = clock.start()
        self.addCleanup(clock.stop)
        mocked_clock.now.return_value = datetime(2026, 9, 7, tzinfo=timezone.utc)
        self.agent = build_agent(RESOURCE_ROOT / "config/ubuntu_sandbox.json",
            data_availability_path=SOURCE_CHECKOUT / "tests/fixtures/ubuntu_data_availability_20260907.json")

    def respond(self, **changes):
        return self.agent.respond("test request", FixedPlanner(plan(**changes)))

    def test_valid_real_payload_matches_smoke_payload(self):
        prepared = self.respond().prepared_task
        payload = prepared.task_spec.to_client_payload()
        self.assertEqual(payload["input_"]["kwargs"]["organisation_ids"], [1, 2])
        self.assertEqual(payload["databases"], [{"label": "synthetic_clean"}])
        self.assertEqual(prepared.task_spec.connection.server_port, 17601)
        self.assertTrue(prepared.task_spec.requires_managed_execution)
        self.assertTrue(all(check.state == "available" for check in prepared.data_checks))

    def test_single_site_numerical_blocked_without_adding_hospitals(self):
        response = self.respond(sites=["hospital 1"])
        self.assertEqual(response.status, "invalid_request")
        self.assertEqual(response.issues[0].code, "site_count_unsupported")
        self.assertIsNone(response.prepared_task)

    def test_single_site_categorical_allowed(self):
        response = self.respond(sites=["hospital 1"], variables=["survival status"])
        self.assertEqual(response.status, "ready")

    def test_unknown_site_does_not_fall_back(self):
        response = self.respond(sites=["hospital 99"])
        self.assertEqual(response.status, "needs_clarification")
        self.assertIn("hospital 1", response.clarification.questions[0].candidates)
        self.assertIsNone(response.prepared_task)

    def test_unknown_variable_in_mixed_request_not_dropped(self):
        response = self.respond(variables=["age at diagnosis", "unregistered_test_variable"])
        self.assertEqual(response.status, "needs_clarification")
        self.assertIsNone(response.prepared_task)

    def test_unknown_reference_gets_targeted_question_before_generic_llm_diagnostic(self):
        response = self.respond(variables=["age at diagnosis", "unregistered_test_variable"],
                                unsupported_requirements=["Requested variable could not be verified"])
        self.assertEqual(response.status, "needs_clarification")
        self.assertEqual(response.clarification.questions[0].requested_value, "unregistered_test_variable")
        self.assertEqual(response.clarification.research_plan.unsupported_requirements,
                         ["Requested variable could not be verified"])
        self.assertIsNone(response.prepared_task)

    def test_reference_clarification_does_not_discard_a_cohort_filter(self):
        response = self.respond(sites=["hospital 99"], unsupported_requirements=["age < 25"])
        self.assertEqual(response.status, "needs_clarification")
        previous = response.clarification.research_plan
        self.assertEqual(previous.unsupported_requirements, ["age < 25"])
        corrected = previous.model_copy(update={"sites": ["hospital 1", "hospital 2"]})
        outcome = self.agent.respond("corrected hospitals", FixedPlanner(corrected))
        self.assertEqual(outcome.status, "unsupported")
        self.assertEqual(outcome.issues[0].requested_value, "age < 25")
        self.assertIsNone(outcome.prepared_task)

    def test_semantic_variable_absent_from_fixture_is_unavailable(self):
        response = self.respond(variables=["gender"])
        self.assertEqual(response.status, "data_unavailable")
        self.assertEqual(response.submission_state, "not_submitted")

    def test_missing_inventory_is_not_false_absence(self):
        path = self.root / "data.json"
        document = read_json(SOURCE_CHECKOUT / "tests/fixtures/ubuntu_data_availability_20260907.json")
        for site in document["sites"]:
            site["complete_inventory"] = False
        write_json(path, document)
        self.agent.data_checker.path = path
        response = self.respond(variables=["gender"])
        self.assertEqual(response.status, "ready")
        outcome = self.agent.execute_response(response.prepared_task, response.prepared_task.confirmation_id)
        self.assertEqual(outcome.issues[0].code, "data_verification_required")

    def test_unknown_database_clarifies(self):
        response = self.respond(database_label="missing_dataset")
        self.assertEqual(response.status, "needs_clarification")
        self.assertEqual(response.issues[0].code, "unknown_database")
        self.assertIn("synthetic_missing", response.issues[0].candidates)

    def test_requested_database_survives_compilation(self):
        response = self.respond(database_label="synthetic_missing")
        self.assertEqual(response.prepared_task.task_spec.databases[0].label, "synthetic_missing")
        self.assertTrue(all(s.database_label == "synthetic_missing" for s in response.prepared_task.resolved_plan.sites))

    def test_explicit_filter_not_dropped(self):
        response = self.respond(unsupported_requirements=["age < 25"])
        self.assertEqual(response.status, "unsupported")
        self.assertEqual(response.issues[0].requested_value, "age < 25")

    def test_unknown_algorithm_suggests_catalog_options(self):
        response = self.respond(analysis_type="cox", algorithm_id="unregistered_cox")
        self.assertEqual(response.status, "unsupported")
        self.assertEqual(response.suggested_algorithms[0].algorithm_id, "strongaya_descriptive_statistics")

    def test_changed_data_rechecked_before_any_credentials(self):
        prepared = self.respond().prepared_task
        document = read_json(SOURCE_CHECKOUT / "tests/fixtures/ubuntu_data_availability_20260907.json")
        document["expires_at"] = "2026-09-06T22:00:00Z"
        path = self.root / "expired.json"
        write_json(path, document)
        self.agent.data_checker.path = path
        provider = Mock(side_effect=AssertionError("must not ask for credentials"))
        self.agent.executor = SandboxVantage6Executor(self.root / "backend", confirmed=True, credentials_provider=provider)
        outcome = self.agent.execute_response(prepared, prepared.confirmation_id)
        self.assertEqual(outcome.status, "preflight_error")
        provider.assert_not_called()

    def test_changed_deployment_invalidates_confirmation(self):
        path = self.root / "deployment.json"
        profile = read_json(RESOURCE_ROOT / "config/ubuntu_sandbox.json")
        write_json(path, profile)
        agent = build_agent(path)
        prepared = agent.prepare(plan())
        profile["server_port"] = 17602
        write_json(path, profile)
        outcome = agent.execute_response(prepared, prepared.confirmation_id)
        self.assertEqual(outcome.issues[0].code, "confirmation_invalid")

    def test_bad_confirmation_never_asks_for_credentials(self):
        prepared = self.respond().prepared_task
        provider = Mock()
        self.agent.executor = SandboxVantage6Executor(self.root / "backend", confirmed=True, credentials_provider=provider)
        outcome = self.agent.execute_response(prepared, "wrong")
        self.assertEqual(outcome.status, "invalid_request")
        provider.assert_not_called()

    def test_real_executor_requires_synthetic_opt_in(self):
        provider = Mock()
        executor = SandboxVantage6Executor(self.root / "backend", credentials_provider=provider)
        with self.assertRaises(BackendExecutionError):
            executor.submit(self.respond().prepared_task.task_spec)
        provider.assert_not_called()

    def test_duplicate_artifact_never_submits_or_reads_credentials(self):
        provider = Mock()
        executor = SandboxVantage6Executor(self.root, confirmed=True, credentials_provider=provider)
        with self.assertRaises(BackendExecutionError) as caught:
            executor.submit(self.respond().prepared_task.task_spec)
        self.assertEqual(caught.exception.code, "duplicate_execution")
        provider.assert_not_called()

    def make_worker(self, *, failure=None, status="completed", encrypted=False):
        task = self.respond().prepared_task.task_spec
        write_json(self.root / "task_spec.json", task.model_dump(mode="json"))
        client = Mock()
        client.util.get_server_version.return_value = {"version": "4.15.0"}
        client.collaboration.get.return_value = {"encrypted": encrypted}
        client.task.create.return_value = {"id": 9001}
        client.task.get.return_value = {"status": status}
        client.result.from_task.return_value = read_json(RAW_FIXTURE)
        if failure:
            client.task.create.side_effect = BackendExecutionError(failure)
        return client

    def test_worker_submits_once_and_retains_raw_result(self):
        client = self.make_worker()
        run_worker(self.root, {"username": "fake", "password": "TEST_SECRET"}, lambda _: client)
        self.assertEqual(read_json(self.root / "state.json")["status"], "completed")
        self.assertEqual(read_json(self.root / "submission.json")["task_id"], "9001")
        client.task.create.assert_called_once()
        self.assertNotIn("TEST_SECRET", "".join(p.read_text() for p in self.root.glob("*.json")))

    def test_worker_denial_is_not_retried_or_reported_as_success(self):
        client = self.make_worker(failure="backend_forbidden")
        run_worker(self.root, {"username": "fake", "password": "TEST_SECRET"}, lambda _: client)
        state = read_json(self.root / "state.json")
        self.assertEqual(state["code"], "backend_forbidden")
        self.assertFalse((self.root / "submission.json").exists())
        client.task.create.assert_called_once()
        client.result.from_task.assert_not_called()

    def test_encrypted_collaboration_rejected_before_submission(self):
        client = self.make_worker(encrypted=True)
        run_worker(self.root, {"username": "fake", "password": "TEST_SECRET"}, lambda _: client)
        self.assertEqual(read_json(self.root / "state.json")["code"], "backend_configuration")
        client.task.create.assert_not_called()

    def test_crashed_task_keeps_id_but_no_normal_report(self):
        client = self.make_worker(status="crashed")
        run_worker(self.root, {"username": "fake", "password": "TEST_SECRET"}, lambda _: client)
        self.assertEqual(read_json(self.root / "state.json")["code"], "backend_task_failed")
        self.assertTrue((self.root / "submission.json").exists())
        self.assertFalse((self.root / "result.raw.json").exists())

    @unittest.skipUnless(HAS_SDK, "Install the live extra for actual SDK transport tests")
    def test_transport_exposes_status_without_sdk_write_retry(self):
        client = create_client(self.respond().prepared_task.task_spec.connection)
        for status, code in ((401, "backend_authentication"), (403, "backend_forbidden"), (404, "backend_not_found")):
            with self.subTest(status=status), patch("requests.request", return_value=SimpleNamespace(status_code=status)) as send:
                with self.assertRaises(BackendExecutionError) as caught:
                    client.request("task", json={}, method="post")
                self.assertEqual(caught.exception.code, code)
                send.assert_called_once()
                self.assertFalse(send.call_args.kwargs["allow_redirects"])

    @unittest.skipUnless(HAS_SDK, "Install the live extra for actual SDK transport tests")
    def test_actual_sdk_auth_serialization_and_result_decode_without_network(self):
        import jwt
        from urllib.parse import urlparse
        from vantage6.common.encryption import DummyCryptor
        task = self.respond().prepared_task.task_spec
        write_json(self.root / "task_spec.json", task.model_dump(mode="json"))
        token = jwt.encode({"sub": "6"}, "test-signing-key-not-a-real-secret" * 2)
        raw = read_json(RAW_FIXTURE)
        raw["data"][0]["result"] = DummyCryptor().encrypt_bytes_to_str(raw["data"][0]["result"].encode(), None)
        submitted = []

        def respond(method, url, **kwargs):
            endpoint = urlparse(url).path.removeprefix("/api/")
            if endpoint == "task" and method.lower() == "post":
                submitted.append(kwargs["json"])
                payload = {"id": 9002}
            else:
                payload = {"version": {"version": "4.15.0"},
                    "user/6": {"firstname": "test", "organization": {"id": 1}},
                    "organization/1": {"name": "test", "public_key": None},
                    "collaboration/1": {"id": 1, "encrypted": False},
                    "task/9002": {"status": "completed"}, "result": deepcopy(raw)}[endpoint]
            return SimpleNamespace(status_code=200, json=lambda: payload)

        auth = SimpleNamespace(status_code=200, json=lambda: {"access_token": token, "refresh_token": "test"})
        blob = SimpleNamespace(ok=True, status_code=200, json=lambda: {"blob_store_enabled": False})
        with patch("requests.request", side_effect=respond), patch("requests.post", return_value=auth), patch("requests.get", return_value=blob):
            run_worker(self.root, {"username": "fake", "password": "TEST_SECRET"})
        self.assertEqual(read_json(self.root / "state.json")["status"], "completed")
        self.assertEqual(len(submitted), 1)
        self.assertEqual(submitted[0]["collaboration_id"], 1)
        self.assertEqual([entry["id"] for entry in submitted[0]["organizations"]], [1])
        parsed = self.agent.result_parser.parse(read_json(self.root / "result.raw.json"))
        self.agent.result_parser.validate_requested(parsed, self.respond().prepared_task.resolved_plan)

    def test_bounded_wait_keeps_task_id_and_does_not_cancel_remote(self):
        executor = SandboxVantage6Executor(self.root, confirmed=True, timeout_seconds=1)
        executor.process = Mock()
        executor.process.poll.return_value = None
        executor.started = 0
        write_json(self.root / "submission.json", {"task_id": 77})
        with patch("time.monotonic", return_value=10):
            with self.assertRaises(BackendExecutionError) as caught:
                executor._poll()
        self.assertEqual(caught.exception.task_id, "77")
        self.assertFalse(read_json(self.root / "timeout.json")["remote_task_cancelled"])
        executor.process.terminate.assert_called_once()

    def test_real_returned_wrapper_parses_and_validates(self):
        raw = read_json(RAW_FIXTURE)
        parsed = self.agent.result_parser.parse(raw)
        self.agent.result_parser.validate_requested(parsed, self.respond().prepared_task.resolved_plan)
        self.assertEqual(next(row["value"] for row in parsed.numerical if row["statistic"] == "count"), 300)

    def test_empty_numerical_result_is_not_a_success(self):
        parsed = self.agent.result_parser.parse({"numerical_general_statistics": {}, "categorical_general_statistics": {}})
        with self.assertRaises(ValueError):
            self.agent.result_parser.validate_requested(parsed, self.respond().prepared_task.resolved_plan)

    def test_chat_clarification_preserves_plan_and_requires_confirmation(self):
        provider = Mock()
        provider.generate_json.side_effect = [plan(sites=[]).model_dump(), {"changes": {"sites": ["hospital 1", "hospital 2"]}}]
        planner = LLMResearchPlanner(provider)
        args = SimpleNamespace(output=self.root, provider="local", backend="mock", timeout_seconds=5,
                               confirm_synthetic_unencrypted_sandbox=False)
        answers = iter(["Describe age at diagnosis", "hospital 1 and hospital 2", "confirm", "/exit"])
        with redirect_stdout(io.StringIO()):
            run_chat(self.agent, planner, args, read=lambda _: next(answers))
        clarification_input = json.loads(provider.generate_json.call_args_list[1].args[1])
        self.assertEqual(clarification_input["previous_plan"]["variables"], ["age at diagnosis"])
        self.assertEqual(clarification_input["researcher_reply"], "hospital 1 and hospital 2")
        records = list(self.root.glob("session-*/turn-*/mock_result.json"))
        self.assertEqual(len(records), 1)

    def test_chat_cancel_never_submits(self):
        args = SimpleNamespace(output=self.root, provider="local", backend="mock", timeout_seconds=5,
                               confirm_synthetic_unencrypted_sandbox=False)
        answers = iter(["Describe age", "", "/exit"])
        self.agent.executor.submit = Mock(side_effect=AssertionError("must not submit"))
        with redirect_stdout(io.StringIO()):
            run_chat(self.agent, FixedPlanner(plan()), args, read=lambda _: next(answers))
        self.agent.executor.submit.assert_not_called()

    def test_provider_billing_has_actionable_feedback(self):
        provider = Mock()
        provider.generate_json.side_effect = HTTPError("https://example.invalid", 402, "private response", {}, None)
        response = self.agent.respond("Describe age", LLMResearchPlanner(provider))
        self.assertEqual(response.issues[0].code, "planner_billing")
        self.assertEqual(response.submission_state, "not_submitted")
        self.assertNotIn("private response", response.model_dump_json())


if __name__ == "__main__":
    unittest.main()
