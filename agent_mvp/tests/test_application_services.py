"""Phase 3 guarantees: declared capabilities, immutable metadata snapshots, one failure
mapping and study rules that do not depend on the terminal."""

import json
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from federated_agent.app import build_agent
from federated_agent.capabilities import (
    DataCapabilities,
    ExecutorCapabilities,
    PlannerCapabilities,
    PolicyCapabilities,
    ProviderCapabilities,
    capabilities_of,
)
from federated_agent.executor import MockVantage6Executor
from federated_agent.models import DataAvailabilityCheck, ExecutionRecord, ExecutionStatus, PolicyDecision, ResearchPlan
from federated_agent.planner import DemoResearchPlanner
from federated_agent.resources import DEFAULT_POLICY, RESOURCE_ROOT
from federated_agent.services.responses import backend_failure
from federated_agent.services.study import StudyChat, StudySettings
from federated_agent.storage import read_json

REQUEST = "Describe age at diagnosis and survival status in hospital 1 and hospital 2 using default."


def plan(**changes):
    return ResearchPlan(**(dict(analysis_type="descriptive_statistics", variables=["age at diagnosis"],
                                sites=["hospital 1", "hospital 2"], database_label="default",
                                purpose="service test") | changes))


class SimulatedExecutor:
    """A new executor type; only its declaration tells the agent how to treat it."""

    def __init__(self, capabilities):
        self.capabilities = capabilities
        self.submitted = []
        self.policy_config = None

    def submit(self, task_spec):
        self.submitted.append(self.policy_config)
        return ExecutionRecord(task_id="sim-1", status=ExecutionStatus.submitted)

    def wait_for_results(self, task_id):
        return ExecutionRecord(task_id=task_id, status=ExecutionStatus.completed, result={"not": "validated"})


class AvailableData:
    capabilities = DataCapabilities(verifies_availability=True)

    def check(self, plan, task):
        return [DataAvailabilityCheck(site=site.canonical_name, variable=variable.canonical_name, state="available",
                                      evidence="metadata_snapshot",
                                      source={"kind": "unit_test", "uri": "test-fixture-only", "version": "1"})
                for site in plan.sites for variable in plan.variables]


class RecordingPolicy:
    def __init__(self, capabilities=None):
        if capabilities is not None:
            self.capabilities = capabilities
        self.actions = []

    def evaluate_action(self, task_spec, *, action, task_id=None, subject=None):
        self.actions.append(action)
        return PolicyDecision(allowed=True, policy_version="test")


class CapabilityTests(unittest.TestCase):
    def setUp(self):
        self.agent = build_agent()
        self.prepared = self.agent.prepare(plan())

    def test_undeclared_collaborators_get_conservative_defaults(self):
        for kind in (ExecutorCapabilities, PolicyCapabilities, DataCapabilities, PlannerCapabilities, ProviderCapabilities):
            with self.subTest(kind=kind.__name__):
                self.assertEqual(capabilities_of(Mock(), kind), kind())
                self.assertEqual(capabilities_of(object(), kind), kind())

    def test_a_new_simulated_executor_needs_no_service_change(self):
        self.agent.executor = SimulatedExecutor(ExecutorCapabilities(simulated=True))
        response = self.agent.execute_response(self.prepared, self.prepared.confirmation_id)
        self.assertEqual(response.status, "completed")
        self.assertIn("no real Vantage6 task was submitted", response.message)

    def test_an_undeclared_executor_is_treated_as_real_and_its_result_is_validated(self):
        self.agent.executor = SimulatedExecutor(None)
        response = self.agent.execute_response(self.prepared, self.prepared.confirmation_id)
        self.assertEqual((response.status, response.issues[0].code), ("execution_failed", "result_invalid"))

    def managed(self, policy, executor):
        """Prepare under ``policy`` with available data, then install ``executor``."""
        self.agent.policy_checker = policy
        self.agent.data_checker = AvailableData()
        prepared = self.agent.prepare(plan())
        self.assertTrue(prepared.task_spec.requires_managed_execution)
        self.agent.executor = executor
        return prepared

    def test_trusted_worker_receives_the_declared_policy_file_and_owns_runtime_decisions(self):
        policy = RecordingPolicy(PolicyCapabilities(worker_config=DEFAULT_POLICY, worker_decides_runtime=True))
        executor = SimulatedExecutor(ExecutorCapabilities(simulated=True, trusted_worker=True))
        prepared = self.managed(policy, executor)
        response = self.agent.execute_response(prepared, prepared.confirmation_id)
        self.assertEqual(response.status, "completed")
        self.assertEqual(executor.submitted, [DEFAULT_POLICY])
        self.assertEqual(set(policy.actions), {"prepare"})

    def test_a_policy_without_worker_configuration_is_refused_before_submission(self):
        executor = SimulatedExecutor(ExecutorCapabilities(trusted_worker=True))
        prepared = self.managed(RecordingPolicy(), executor)
        response = self.agent.execute_response(prepared, prepared.confirmation_id)
        self.assertEqual((response.status, response.issues[0].code), ("preflight_error", "preflight_error"))
        self.assertEqual(executor.submitted, [])

    def test_an_in_process_policy_decides_every_runtime_action(self):
        policy = RecordingPolicy(PolicyCapabilities(worker_config=DEFAULT_POLICY))
        prepared = self.managed(policy, SimulatedExecutor(ExecutorCapabilities(simulated=True, trusted_worker=True)))
        self.assertEqual(self.agent.execute_response(prepared, prepared.confirmation_id).status, "completed")
        self.assertEqual(policy.actions, ["prepare", "prepare", "submit", "read_result", "release_result"])

    def test_a_planner_that_declares_context_receives_it(self):
        class ContextPlanner:
            capabilities = PlannerCapabilities(uses_context=True)
            context = None

            def create_plan(self, request, catalog):
                self.seen = self.context
                return plan()

        planner = ContextPlanner()
        self.assertEqual(self.agent.respond("Describe age", planner).status, "ready")
        self.assertEqual({site["name"] for site in planner.seen["sites"]}, {"hospital 1", "hospital 2", "hospital 3"})

    def test_managed_execution_follows_declarations(self):
        self.assertFalse(self.prepared.task_spec.requires_managed_execution)
        self.agent.policy_checker = RecordingPolicy()
        self.assertTrue(self.agent.prepare(plan()).task_spec.requires_managed_execution)


class MetadataSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.deployment = root / "deployment.json"
        self.catalog = root / "algorithms.json"
        self.deployment.write_bytes((RESOURCE_ROOT / "config/demo_deployment.json").read_bytes())
        self.catalog.write_bytes((RESOURCE_ROOT / "catalogs/algorithms.json").read_bytes())
        self.agent = build_agent(self.deployment, self.catalog)

    def rename_hospital(self, name):
        document = json.loads(self.deployment.read_text(encoding="utf-8"))
        document["sites"][0]["aliases"].append(name)
        self.deployment.write_text(json.dumps(document), encoding="utf-8")

    def test_refresh_replaces_the_snapshot_and_never_changes_the_previous_one(self):
        before = self.agent.refresh_metadata()
        self.rename_hospital("northern clinic")
        after = self.agent.refresh_metadata()
        self.assertIsNot(before, after)
        self.assertIs(self.agent.metadata, after)
        self.assertNotIn("northern clinic", before.deployment_registry.profile.sites[0].aliases)
        self.assertIn("northern clinic", after.deployment_registry.profile.sites[0].aliases)
        self.assertIsNot(before.task_compiler, after.task_compiler)

    def test_a_failed_capture_keeps_the_previous_snapshot(self):
        before = self.agent.refresh_metadata()
        self.catalog.write_text("{ not json", encoding="utf-8")
        self.assertEqual(self.agent.describe_algorithms().status, "catalog_error")
        self.assertIs(self.agent.metadata, before)

    def test_snapshots_cannot_be_modified(self):
        snapshot = self.agent.refresh_metadata()
        with self.assertRaises(FrozenInstanceError):
            snapshot.deployment_registry = None
        with self.assertRaises(TypeError):
            snapshot.local_metadata_hashes["deployment"] = "0" * 64


class FailureMappingTests(unittest.TestCase):
    def test_submission_state_is_only_as_certain_as_the_evidence(self):
        cases = [
            ("backend_authentication", "submit", None, False, "not_submitted"),
            ("backend_configuration", "preflight", None, False, "not_submitted"),
            ("backend_error", "submit", None, False, "unknown"),
            ("backend_timeout", "result", "7", False, "submitted"),
            ("backend_task_failed", "result", "7", False, "failed"),
            ("backend_configuration", "preflight", None, True, "unknown"),
            ("backend_timeout", "result", "7", True, "submitted"),
        ]
        for code, stage, task_id, recovery, expected in cases:
            with self.subTest(code=code, stage=stage, recovery=recovery):
                response = backend_failure(code, task_id=task_id, stage=stage, recovery=recovery)
                self.assertEqual((response.status, response.submission_state), ("execution_failed", expected))
                self.assertEqual(response.execution.task_id if response.execution else None, task_id)


class ScriptedUI:
    """A non-terminal transport for the study rules."""

    def __init__(self, inputs, *, approve=True):
        self.inputs = iter(inputs)
        self.approve = approve
        self.language = "en"
        self.last_response = None
        self.messages = []
        self.presented = []

    def header(self, settings, planner, session, policy):
        self.messages.append(f"policy: {policy}")

    def say(self, text="", style=None):
        self.messages.append(text)

    def ask(self, pending=None):
        return next(self.inputs)

    def menu(self):
        return "/exit"

    def select_session(self, names):
        return None

    @contextmanager
    def planning(self, planner):
        yield

    def present(self, response, planner, output, *, narrate=True):
        self.last_response = response
        self.presented.append(response.status)

    def review(self, prepared, output):
        return self.approve

    def progress(self, text):
        self.messages.append(text)


class StudyServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.agent = build_agent()

    def chat(self, inputs, **options):
        ui = ScriptedUI(inputs, **options)
        StudyChat(self.agent, DemoResearchPlanner(), ui, StudySettings(output=self.root)).run()
        return ui

    def turn(self):
        return next(self.root.glob("session-*/turn-001"))

    def test_consent_is_saved_before_the_agent_executes(self):
        execute = self.agent.execute_response

        def checked(prepared, confirmation_id):
            consent = read_json(self.turn() / "researcher_confirmation.json")
            self.assertEqual(consent["confirmation_id"], confirmation_id)
            return execute(prepared, confirmation_id)

        self.agent.execute_response = checked
        ui = self.chat([REQUEST, "/exit"])
        self.assertEqual(ui.presented, ["ready", "completed"])
        self.assertIn("policy: allow-all (development)", ui.messages)
        self.assertTrue((self.turn() / "mock_result.json").exists())
        self.assertEqual(read_json(self.turn() / "run_manifest.json")["status"], "completed")

    def test_declined_review_never_executes(self):
        self.agent.executor = Mock(spec=MockVantage6Executor)
        ui = self.chat([REQUEST, "/exit"], approve=False)
        self.agent.executor.submit.assert_not_called()
        self.assertIn("Not submitted. Plan saved; revise it or use /review later.", ui.messages)
        self.assertFalse((self.turn() / "researcher_confirmation.json").exists())

    def test_settings_accept_the_command_line_namespace(self):
        settings = StudySettings.from_args(SimpleNamespace(output=str(self.root), backend="vantage6", provider="demo",
                                                           confirm_synthetic_unencrypted_sandbox=True,
                                                           timeout_seconds=None, resume=None))
        self.assertEqual((settings.backend, settings.confirmed_sandbox, settings.timeout_seconds), ("vantage6", True, 600))


if __name__ == "__main__":
    unittest.main()
