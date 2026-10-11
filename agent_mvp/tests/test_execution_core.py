"""Execution core guarantees: portable fingerprints, worker upgrades, one run-state owner,
worker incidents and worker-scoped code identity."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import Mock

from federated_agent.confirmation import task_confirmation_id, task_confirmation_material
from federated_agent.errors import BackendExecutionError
from federated_agent.execution_templates import WORKER_SOURCES, adapter_digest
from federated_agent.hashing import files_sha256
from federated_agent.models import ResearchPlan
from federated_agent.resources import RESOURCE_ROOT
from federated_agent.runs.executor import SandboxVantage6Executor
from federated_agent.runs.store import Attempt, RunStore
from federated_agent.runs.worker import run_worker
from federated_agent.storage import write_json
from federated_agent.app import build_agent

FIXTURES = Path(__file__).parent / "fixtures"
SANDBOX = RESOURCE_ROOT / "config" / "ubuntu_sandbox.json"


def plan(**changes):
    return ResearchPlan(**(dict(analysis_type="descriptive_statistics", variables=["age at diagnosis"],
                                sites=["hospital 1", "hospital 2"], database_label="default",
                                purpose="fingerprint test") | changes))


def sandbox_client(statuses=("completed",)):
    client = Mock()
    client.util.get_server_version.return_value = {"version": "4.15.0"}
    client.token = "token"
    client.whoami = Mock(id=1, organization_id=1)
    client.collaboration.get.return_value = {"encrypted": False}
    client.task.create.return_value = {"id": 41}
    client.task.get.side_effect = [{"status": status} for status in statuses]
    client.result.from_task.return_value = {"data": [{"result": "{}"}]}
    return client


class FingerprintTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def copy_metadata(self, name, *, crlf=False):
        target = self.root / name
        target.mkdir()
        paths = {}
        for role, source in {"catalog": RESOURCE_ROOT / "catalogs/algorithms.json",
                             "schema": RESOURCE_ROOT / "catalogs/AYA_cancer_schema.jsonld",
                             "deployment": RESOURCE_ROOT / "config/demo_deployment.json"}.items():
            text = source.read_text(encoding="utf-8").replace("\r\n", "\n")
            if crlf:
                text = text.replace("\n", "\r\n")
            paths[role] = target / source.name
            paths[role].write_bytes(text.encode("utf-8"))
        return paths

    def prepare_with(self, paths):
        agent = build_agent(paths["deployment"], paths["catalog"], semantic_schema_path=paths["schema"])
        return agent.prepare(plan())

    def test_same_content_in_another_directory_or_line_ending_gives_the_same_confirmation(self):
        first = self.prepare_with(self.copy_metadata("checkout-a"))
        second = self.prepare_with(self.copy_metadata("checkout b", crlf=True))
        self.assertEqual(first.task_spec.fingerprint_version, 2)
        self.assertEqual(first.confirmation_id, second.confirmation_id)
        self.assertNotIn(self.temp.name, json.dumps(task_confirmation_material(first.task_spec)))

    def test_a_content_change_still_changes_the_confirmation(self):
        paths = self.copy_metadata("checkout")
        before = self.prepare_with(paths)
        document = json.loads(paths["deployment"].read_text(encoding="utf-8"))
        document["collaboration_id"] = 2
        paths["deployment"].write_text(json.dumps(document), encoding="utf-8")
        self.assertNotEqual(before.confirmation_id, self.prepare_with(paths).confirmation_id)

    def test_moved_checkout_can_execute_a_task_prepared_elsewhere(self):
        prepared = self.prepare_with(self.copy_metadata("original"))
        moved = self.copy_metadata("moved")
        agent = build_agent(moved["deployment"], moved["catalog"], semantic_schema_path=moved["schema"])
        response = agent.execute_response(prepared, prepared.confirmation_id)
        self.assertEqual(response.status, "completed")

    def test_worker_digest_covers_only_worker_sources_and_ignores_line_endings(self):
        self.assertEqual(set(WORKER_SOURCES), {"runs/worker.py", "runs/submission.py", "runtime_adapters.py",
                                               "adapters/vantage6.py"})
        unix, windows = self.root / "unix.py", self.root / "windows.py"
        unix.write_bytes(b"a = 1\nb = 2\n")
        windows.write_bytes(b"a = 1\r\nb = 2\r\n")
        self.assertEqual(files_sha256([unix]), files_sha256([windows]))
        self.assertRegex(adapter_digest(), r"^[0-9a-f]{64}$")


class WorkerUpgradeTests(unittest.TestCase):
    """A task prepared by another worker version may be recovered, but never submitted."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.backend = Path(self.temp.name) / "backend"
        task = build_agent(SANDBOX).prepare(plan(database_label="synthetic_clean")).task_spec
        template = task.execution_template.model_copy(update={"adapter_sha256": "0" * 64})
        self.task = task.model_copy(update={"execution_template": template})

    def attempt(self, *, submitted):
        RunStore(self.backend).create(self.task)
        if submitted:
            write_json(self.backend / "submission.json", {"task_id": "41", "confirmation_id": task_confirmation_id(self.task)})

    def test_a_submitted_task_is_recoverable_after_the_worker_changed(self):
        self.attempt(submitted=True)
        with redirect_stderr(io.StringIO()):
            run_worker(self.backend, {"username": "fake", "password": "fake"}, client_factory=lambda _: sandbox_client(),
                       sleep=Mock(), resume=True)
        self.assertEqual(RunStore(self.backend).state()["status"], "completed")
        self.assertEqual(RunStore(self.backend).attempt(), Attempt.COMPLETED)

    def test_an_unsubmitted_task_from_another_worker_cannot_be_submitted(self):
        self.attempt(submitted=False)
        client = sandbox_client()
        with redirect_stderr(io.StringIO()):
            run_worker(self.backend, {"username": "fake", "password": "fake"}, client_factory=lambda _: client)
        state = RunStore(self.backend).state()
        self.assertEqual((state["status"], state["code"]), ("failed", "confirmation_invalid"))
        client.task.create.assert_not_called()


class RunStoreAndIncidentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.backend = Path(self.temp.name) / "backend"
        agent = build_agent(SANDBOX)
        self.task = agent.prepare(plan(database_label="synthetic_clean")).task_spec

    def test_attempt_classification_is_conservative(self):
        store = RunStore(self.backend)
        store.create(self.task)
        cases = [
            ({"status": "waiting_for_credentials", "stage": "authentication"}, False, False, Attempt.NOT_SUBMITTED),
            ({"status": "failed", "stage": "preflight", "code": "backend_configuration"}, False, False, Attempt.NOT_SUBMITTED),
            ({"status": "submitting"}, True, False, Attempt.UNKNOWN),
            ({"status": "running"}, True, True, Attempt.SUBMITTED),
            ({"status": "failed", "code": "backend_task_failed", "stage": "result"}, True, True, Attempt.FAILED),
            ({"status": "completed"}, True, True, Attempt.COMPLETED),
        ]
        for state, intent, submission, expected in cases:
            with self.subTest(state=state):
                for name in ("submission_intent.json", "submission.json"):
                    (self.backend / name).unlink(missing_ok=True)
                store.set_state(**state)
                if intent:
                    store.record_intent("x")
                if submission:
                    store.record_submission("41", "x")
                self.assertEqual(store.attempt(), expected)

    def test_unexpected_worker_failure_reaches_the_parent_with_an_incident_id(self):
        RunStore(self.backend).create(self.task)

        def broken_factory(_):
            raise RuntimeError("socket closed password=hunter2")

        with redirect_stderr(io.StringIO()):
            run_worker(self.backend, {"username": "fake", "password": "fake"}, client_factory=broken_factory)
        state = RunStore(self.backend).state()
        self.assertEqual((state["status"], state["code"], state["stage"]), ("failed", "backend_error", "preflight"))
        self.assertRegex(state["incident_id"], r"^[0-9a-f]{12}$")
        executor = SandboxVantage6Executor(self.backend, confirmed=True)
        executor.started = 0
        executor.process = Mock()
        with self.assertRaises(BackendExecutionError) as raised:
            executor._poll()
        self.assertEqual(raised.exception.incident_id, state["incident_id"])

    def test_expected_failures_carry_no_incident(self):
        RunStore(self.backend).create(self.task)
        client = sandbox_client()
        client.collaboration.get.return_value = {"encrypted": True}
        with redirect_stderr(io.StringIO()):
            run_worker(self.backend, {"username": "fake", "password": "fake"}, client_factory=lambda _: client)
        state = RunStore(self.backend).state()
        self.assertEqual(state["code"], "backend_configuration")
        self.assertNotIn("incident_id", state)


if __name__ == "__main__":
    unittest.main()
