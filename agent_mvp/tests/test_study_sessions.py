import io
import json
import subprocess
import sys
import tempfile
import unittest
import threading
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from federated_agent.resources import RESOURCE_ROOT, SOURCE_CHECKOUT
from federated_agent.cli import run_chat
from federated_agent.confirmation import task_confirmation_id
from federated_agent.errors import BackendExecutionError
from federated_agent.models import ExecutionRecord, ResearchPlan
from federated_agent.planner import LLMResearchPlanner
from federated_agent.sessions import FileLease, SessionError, SessionStore, StudySession
from federated_agent.app import build_agent
from federated_agent.runs.executor import SandboxVantage6Executor
from federated_agent.runs.worker import run_worker
from federated_agent.services.artifacts import write_prepared
from federated_agent.storage import read_json, write_json

try:
    HAS_SDK = version("vantage6-client") == "4.15.0"
except PackageNotFoundError:
    HAS_SDK = False


def plan(**changes):
    return ResearchPlan(**(dict(analysis_type="descriptive_statistics", variables=["age at diagnosis"],
        sites=["hospital 1", "hospital 2"], database_label="synthetic_clean",
        purpose="Describe age in the synthetic study") | changes))


class StudySessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.agent = build_agent(RESOURCE_ROOT / "config/ubuntu_sandbox.json",
                                     data_availability_path=SOURCE_CHECKOUT / "tests/fixtures/ubuntu_data_availability_20260907.json")
        clock = patch("federated_agent.data_availability.datetime")
        clock.start().now.return_value = datetime(2026, 9, 7, tzinfo=timezone.utc)
        self.addCleanup(clock.stop)
        self.provider = Mock()
        self.provider.model = "test-model"
        self.planner = LLMResearchPlanner(self.provider, retries=0)

    def chat(self, replies, *, resume=None, backend="mock"):
        args = SimpleNamespace(output=self.root, provider="local", backend=backend, timeout_seconds=5,
                               confirm_synthetic_unencrypted_sandbox=True, resume=resume, stream=False, plain=True)
        replies = iter(replies)
        output = io.StringIO()
        with redirect_stdout(output):
            run_chat(self.agent, self.planner, args, read=lambda _: next(replies))
        return output.getvalue()

    def session(self):
        return next(self.root.glob("session-*"))

    def test_clarification_survives_a_restart_without_auto_execution(self):
        self.provider.generate_json.return_value = plan(sites=[]).model_dump()
        self.chat(["Describe age", "/exit"])
        name = self.session().name
        self.provider.generate_json.reset_mock()
        self.agent.executor.submit = Mock(side_effect=AssertionError("No implicit submission"))
        text = self.chat(["/plan", "/history", "/exit"], resume=name)
        self.assertIn("Resumed", text)
        self.assertIn("needs_clarification", text)
        self.provider.generate_json.assert_not_called()
        self.agent.executor.submit.assert_not_called()
        self.provider.generate_json.return_value = {"changes": {"sites": ["hospital 1", "hospital 2"]}}
        self.chat(["hospital 1 and hospital 2", "", "/exit"], resume=name)
        state = read_json(self.session() / "session.json")
        self.assertEqual(state["current_plan"]["variables"], ["age at diagnosis"])
        self.assertEqual(state["current_plan"]["sites"], ["hospital 1", "hospital 2"])
        self.assertIsNone(state["pending"])
        self.assertTrue((self.session() / "turn-002/prepared_task.json").exists())

    def test_followup_after_completion_keeps_context_but_requires_new_confirmation(self):
        self.provider.generate_json.side_effect = [plan().model_dump(),
            {"changes": {"variables": ["age at diagnosis", "survival status"]}}]
        self.chat(["Describe age", "confirm", "Add survival status; keep everything else", "", "/exit"])
        self.assertEqual(len(list(self.session().glob("turn-*/mock_result.json"))), 1)
        revised = read_json(self.session() / "turn-002/prepared_task.json")["research_plan"]
        self.assertEqual(revised["sites"], ["hospital 1", "hospital 2"])
        self.assertEqual(revised["database_label"], "synthetic_clean")
        self.assertEqual(revised["variables"], ["age at diagnosis", "survival status"])
        prompt = json.loads(self.provider.generate_json.call_args.args[1])
        self.assertEqual(set(prompt), {"previous_plan", "researcher_reply", "questions"})
        self.assertNotIn("result", prompt)
        revision = read_json(self.session() / "turn-002/plan_revision.json")
        self.assertEqual(set(revision["changes"]), {"variables", "purpose"})
        self.assertIn("Add survival status", revised["purpose"])

    def test_unsupported_filter_persists_across_restart_and_hospital_change(self):
        self.provider.generate_json.return_value = plan(unsupported_requirements=["age < 25"]).model_dump()
        self.chat(["Describe age only under 25", "/exit"])
        self.provider.generate_json.return_value = {"changes": {"sites": ["hospital 2", "hospital 3"]}}
        self.chat(["Change hospitals to 2 and 3", "/exit"], resume=self.session().name)
        response = read_json(self.session() / "turn-002/response.json")
        self.assertEqual(response["status"], "unsupported")
        self.assertEqual(read_json(self.session() / "session.json")["current_plan"]["unsupported_requirements"], ["age < 25"])
        self.assertFalse(list(self.session().glob("turn-*/prepared_task.json")))

    def test_empty_or_unknown_patch_cannot_execute(self):
        for changes in ({}, {"image": "untrusted"}, {"workflow": "exec(...)"}):
            with self.subTest(changes=changes):
                self.provider.generate_json.return_value = {"changes": changes}
                response = self.agent.respond("change it", self.planner, previous_plan=plan())
                self.assertEqual(response.status, "planner_error")
                self.assertIsNone(response.prepared_task)

    def test_new_clears_context_without_removing_history(self):
        self.provider.generate_json.side_effect = [plan(sites=[]).model_dump(), plan().model_dump()]
        self.chat(["Describe age", "/new", "Describe age at both hospitals", "", "/exit"])
        prompt = self.provider.generate_json.call_args.args[1]
        self.assertEqual(prompt, "Describe age at both hospitals")
        self.assertTrue((self.session() / "turn-001/response.json").exists())

    def test_bad_followup_keeps_last_valid_study(self):
        self.provider.generate_json.side_effect = [plan().model_dump(), {"changes": {"image": "untrusted"}}]
        self.chat(["Describe age", "", "Change image", "/exit"])
        self.assertEqual(read_json(self.session() / "session.json")["current_plan"],
                         plan(algorithm_id="strongaya_descriptive_statistics").model_dump())

    def test_unavailable_variable_is_not_forgotten_on_next_followup(self):
        self.provider.generate_json.side_effect = [plan().model_dump(),
            {"changes": {"variables": ["age at diagnosis", "gender"]}},
            {"changes": {"sites": ["hospital 2", "hospital 3"]}}]
        self.chat(["Describe age", "", "Add gender", "Use hospitals 2 and 3", "/exit"])
        state = read_json(self.session() / "session.json")
        self.assertEqual(state["current_plan"]["variables"], ["age at diagnosis", "gender"])
        self.assertEqual(state["last_response"]["status"], "data_unavailable")
        self.assertFalse((self.session() / "turn-003/prepared_task.json").exists())

    def test_recovery_command_never_calls_a_model_or_submits(self):
        path = self.root / "session-recovery"
        turn = path / "turn-001"
        backend = turn / "backend"
        backend.mkdir(parents=True)
        prepared = self.agent.prepare(plan())
        write_prepared(prepared, turn)
        write_json(path / "session.json", StudySession(
            session_id=path.name, backend="vantage6", created_at="2026-10-01T00:00:00+00:00",
            updated_at="2026-10-01T00:00:00+00:00", current_plan=prepared.research_plan,
            active_turn="turn-001").model_dump(mode="json"))
        write_json(backend / "task_spec.json", prepared.task_spec.model_dump(mode="json"))
        write_json(backend / "submission.json", {"task_id": "8001"})
        executor = SandboxVantage6Executor(backend, confirmed=True)
        raw = read_json(Path(__file__).parent / "fixtures/vantage6_s02_result.json")
        executor.recover = Mock(return_value=ExecutionRecord(task_id="8001", status="completed", result=raw))
        executor.submit = Mock(side_effect=AssertionError("Recovery must never submit"))

        def configure(agent, settings, output, ui):
            agent.executor = executor

        with patch("federated_agent.services.study.configure_executor", side_effect=configure):
            self.chat(["/recover turn-001", "/exit"], resume=path.name, backend="vantage6")
        self.provider.generate_json.assert_not_called()
        executor.submit.assert_not_called()
        executor.recover.assert_called_once()
        self.assertEqual(read_json(turn / "execution.json")["task_id"], "8001")
        self.assertTrue((turn / "result.html").exists())

    def test_re_review_of_existing_attempt_is_blocked(self):
        self.provider.generate_json.return_value = plan().model_dump()
        text = self.chat(["Describe age", "confirm", "/review", "/exit"])
        self.assertIn("already has an execution attempt", text)
        self.assertEqual(len(list(self.session().glob("turn-*/mock_result.json"))), 1)

    def test_revised_request_cannot_bypass_an_uncertain_submission_in_session(self):
        self.provider.generate_json.return_value = plan().model_dump()
        self.chat(["Describe age", "", "/exit"], backend="vantage6")
        backend = self.session() / "turn-001/backend"
        backend.mkdir()
        write_json(backend / "submission_intent.json", {"automatic_retry_allowed": False})
        write_json(backend / "state.json", {"status": "submitting"})
        self.provider.generate_json.return_value = {"changes": {"variables": ["survival status"]}}
        text = self.chat(["Try survival status instead", "/exit"], resume=self.session().name, backend="vantage6")
        self.assertIn("unresolved earlier attempts", text)
        self.assertFalse((self.session() / "turn-002/backend").exists())

    def test_saved_state_excludes_results_and_records_manifest(self):
        self.provider.generate_json.return_value = plan().model_dump()
        self.chat(["Describe age", "confirm", "/exit"])
        state = read_json(self.session() / "session.json")
        self.assertIsNone(state["last_response"]["execution"]["result"])
        manifest = read_json(self.session() / "turn-001/run_manifest.json")
        self.assertEqual(manifest["template"]["version"], "1.1.0")
        self.assertEqual(manifest["planner"]["model"], "test-model")
        self.assertEqual(manifest["status"], "completed")
        self.assertTrue(manifest["task_id"])
        self.assertNotIn("api_key", json.dumps(manifest))
        self.assertTrue((self.session() / "turn-001/researcher_confirmation.json").exists())

    def test_resumed_study_preserves_explicit_dataset_when_default_changes(self):
        self.provider.generate_json.return_value = plan(database_label="synthetic_clean").model_dump()
        self.chat(["Describe age", "", "/exit"])
        profile = read_json(RESOURCE_ROOT / "config/ubuntu_sandbox.json")
        profile["database_label"] = "synthetic_missing"
        path = self.root / "deployment.json"
        write_json(path, profile)
        self.agent = build_agent(path, data_availability_path=SOURCE_CHECKOUT / "tests/fixtures/ubuntu_data_availability_20260907.json")
        self.chat(["/review", "", "/exit"], resume=self.session().name)
        revised = read_json(self.session() / "turn-002/prepared_task.json")
        self.assertEqual(revised["task_spec"]["databases"], [{"label": "synthetic_clean", "query": None, "sheet_name": None}])

    def test_session_lock_and_paths_reject_unsafe_or_concurrent_resume(self):
        store = SessionStore.create(self.root, "mock")
        try:
            for name in (store.path.name, "../outside", "session-../../outside"):
                with self.subTest(name=name), self.assertRaises(SessionError):
                    SessionStore.open(self.root, name, "mock")
        finally:
            store.close()
        other = SessionStore.open(self.root, store.path.name, "mock")
        other.close()
        with self.assertRaises(SessionError):
            SessionStore.open(self.root, store.path.name, "vantage6")

    def test_process_exit_releases_os_lease(self):
        path = self.root / ".lease"
        code = "from federated_agent.sessions import FileLease; import os, sys; lock=FileLease(sys.argv[1]); os._exit(0)"
        subprocess.run([sys.executable, "-c", code, str(path)], check=True, timeout=15)
        lease = FileLease(path)
        lease.close()

    def test_startup_calls_out_expired_metadata(self):
        text = self.chat(["/exit"])
        self.assertIn("EXPIRED", text)
        self.assertIn("Do not just extend", " ".join(text.split()))


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.agent = build_agent(RESOURCE_ROOT / "config/ubuntu_sandbox.json")
        self.prepared = self.agent.prepare(plan())
        write_json(self.root / "task_spec.json", self.prepared.task_spec.model_dump(mode="json"))
        write_json(self.root / "submission.json", {"task_id": "8001", "confirmation_id": self.prepared.confirmation_id})
        self.client = Mock()
        self.client.util.get_server_version.return_value = {"version": "4.15.0"}
        self.client.collaboration.get.return_value = {"encrypted": False}
        self.client.task.get.return_value = {"status": "completed"}
        self.raw = read_json(Path(__file__).parent / "fixtures/vantage6_s02_result.json")
        self.client.result.from_task.return_value = self.raw

    def run_recovery_worker(self):
        run_worker(self.root, {"username": "fake", "password": "LOCAL_TEST_SECRET"}, lambda _: self.client, sleep=lambda _: None, resume=True)

    def test_worker_recovers_running_task_without_creating_a_task(self):
        self.client.task.get.side_effect = [{"status": "running"}, {"status": "completed"}]
        self.run_recovery_worker()
        self.client.task.create.assert_not_called()
        self.client.task.get.assert_called_with(8001)
        self.client.result.from_task.assert_called_once_with(task_id=8001)
        self.assertEqual(read_json(self.root / "state.json")["status"], "completed")
        self.assertNotIn("LOCAL_TEST_SECRET", "".join(p.read_text() for p in self.root.glob("*.json")))

    def test_missing_id_never_authenticates_or_creates(self):
        (self.root / "submission.json").unlink()
        self.run_recovery_worker()
        self.assertEqual(read_json(self.root / "state.json")["code"], "submission_unresolved")
        self.client.authenticate.assert_not_called()
        self.client.task.create.assert_not_called()

    def test_worker_cannot_resubmit_an_attempt_after_losing_submit_response(self):
        (self.root / "submission.json").unlink()
        write_json(self.root / "submission_intent.json", {"automatic_retry_allowed": False})
        run_worker(self.root, {"username": "fake", "password": "secret"}, lambda _: self.client)
        self.assertEqual(read_json(self.root / "state.json")["code"], "submission_unresolved")
        self.client.authenticate.assert_not_called()
        self.client.task.create.assert_not_called()

    def test_existing_worker_lock_blocks_recovery_before_credentials(self):
        lease = FileLease(self.root / ".worker.lock")
        credentials = Mock()
        executor = SandboxVantage6Executor(self.root, confirmed=True, credentials_provider=credentials)
        try:
            with self.assertRaises(BackendExecutionError) as caught:
                executor.recover(self.prepared.task_spec)
            self.assertEqual(caught.exception.code, "execution_busy")
            credentials.assert_not_called()
        finally:
            lease.close()

    def test_failed_remote_task_stays_failed(self):
        self.client.task.get.return_value = {"status": "failed"}
        self.run_recovery_worker()
        self.assertEqual(read_json(self.root / "state.json")["code"], "backend_task_failed")
        self.client.task.create.assert_not_called()
        self.client.result.from_task.assert_not_called()

    def test_unauthorized_recovery_never_retries_or_returns_a_result(self):
        self.client.task.get.side_effect = BackendExecutionError("backend_forbidden")
        self.run_recovery_worker()
        self.assertEqual(read_json(self.root / "state.json")["code"], "backend_forbidden")
        self.client.task.get.assert_called_once()
        self.client.task.create.assert_not_called()
        self.client.result.from_task.assert_not_called()

    def test_recovery_uses_policy_hook_and_does_not_require_new_inventory(self):
        self.agent.executor = SandboxVantage6Executor(self.root, confirmed=True)
        self.agent.executor.recover = Mock(return_value=ExecutionRecord(task_id="8001", status="completed", result=self.raw))
        self.agent.data_checker.check = Mock(side_effect=AssertionError("Existing task recovery must not start data preflight"))
        response = self.agent.recover_response(self.prepared)
        self.assertEqual(response.status, "completed")
        self.agent.policy_checker = Mock()
        self.agent.policy_checker.evaluate.return_value = {"allowed": False}
        self.agent.executor.recover.reset_mock()
        response = self.agent.recover_response(self.prepared)
        self.assertEqual(response.issues[0].code, "permission_denied")
        self.agent.executor.recover.assert_not_called()

    def test_tampered_task_and_changed_server_block_recovery(self):
        self.agent.executor = SandboxVantage6Executor(self.root, confirmed=True)
        self.agent.executor.recover = Mock()
        changed = self.prepared.model_copy(deep=True)
        changed.task_spec.connection.server_port = 17602
        response = self.agent.recover_response(changed)
        self.assertEqual(response.issues[0].code, "recovery_mismatch")
        changed.confirmation_id = task_confirmation_id(changed.task_spec)
        response = self.agent.recover_response(changed)
        self.assertEqual(response.issues[0].code, "recovery_mismatch")
        self.agent.executor.recover.assert_not_called()

    def test_missing_task_id_blocks_before_credential_prompt(self):
        (self.root / "submission.json").unlink()
        credentials = Mock()
        executor = SandboxVantage6Executor(self.root, confirmed=True, credentials_provider=credentials)
        with self.assertRaises(BackendExecutionError) as caught:
            executor.recover(self.prepared.task_spec)
        self.assertEqual(caught.exception.code, "submission_unresolved")
        credentials.assert_not_called()

    def test_known_completed_state_is_reauthenticated_not_blindly_trusted(self):
        write_json(self.root / "state.json", {"status": "completed", "task_id": "8001"})
        write_json(self.root / "result.raw.json", {"stale": "result"})
        self.run_recovery_worker()
        self.client.authenticate.assert_called_once()
        self.client.task.get.assert_called_once_with(8001)
        self.assertEqual(read_json(self.root / "result.raw.json"), self.raw)

    @unittest.skipUnless(HAS_SDK, "Requires the pinned Vantage6 SDK")
    def test_real_sdk_subprocess_reconnects_over_loopback_without_task_post(self):
        import jwt
        from vantage6.common.encryption import DummyCryptor
        calls = []
        token = jwt.encode({"sub": "6"}, "unit-test-signing-key-not-a-real-secret" * 2)
        raw = deepcopy(self.raw)
        raw["data"][0]["result"] = DummyCryptor().encrypt_bytes_to_str(raw["data"][0]["result"].encode(), None)

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, value, status=200):
                body = json.dumps(value).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                calls.append(("POST", self.path))
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                if self.path == "/api/token/user":
                    self.reply({"access_token": token, "refresh_token": "unit-test-token"})
                else:
                    self.reply({"error": "No new tasks allowed"}, 403)

            def do_GET(self):
                calls.append(("GET", self.path))
                endpoint = self.path.split("?", 1)[0]
                values = {"/api/version": {"version": "4.15.0"},
                    "/api/user/6": {"firstname": "test", "organization": {"id": 1}},
                    "/api/organization/1": {"name": "test", "public_key": None},
                    "/api/collaboration/1": {"id": 1, "encrypted": False},
                    "/api/task/8001": {"status": "completed"},
                    "/api/blobstream/status": {"blob_store_enabled": False},
                    "/api/result": raw}
                self.reply(values.get(endpoint, {"error": "not found"}), 200 if endpoint in values else 404)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        task = self.prepared.task_spec.model_copy(deep=True)
        task.connection.server_port = server.server_port
        write_json(self.root / "task_spec.json", task.model_dump(mode="json"))
        write_json(self.root / "submission.json", {"task_id": "8001", "confirmation_id": task_confirmation_id(task)})
        executor = SandboxVantage6Executor(self.root, confirmed=True, timeout_seconds=20,
            credentials_provider=lambda: {"username": "fake", "password": "LOCAL_TEST_SECRET"}, progress=lambda _: None)
        try:
            record = executor.recover(task)
            self.assertEqual(record.task_id, "8001")
            self.assertEqual(record.status.value, "completed")
            self.assertEqual(record.result, self.raw)
            self.assertEqual([call for call in calls if call[0] == "POST"], [("POST", "/api/token/user")])
            self.assertIn(("GET", "/api/task/8001"), calls)
        finally:
            executor.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
