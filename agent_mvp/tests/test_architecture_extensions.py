"""Extension contracts, trust boundaries, atomic updates and old-task protection."""

import base64
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import importlib.util
import os
from pathlib import Path
import tempfile
import threading
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

from federated_agent.resources import RESOURCE_ROOT, SOURCE_CHECKOUT
from federated_agent.errors import PolicyDeniedError, PreflightError
from federated_agent.execution_templates import TemplateRegistry, validate_installed
from federated_agent.models import PolicyDecision, ResearchPlan, ExecutionRecord
from federated_agent.planner import LLMResearchPlanner
from federated_agent.policy import FilePolicyChecker, enforce_policy
from federated_agent.registry_admin import publish
from federated_agent.registry_sync import RegistryClient, ROLES, canonical
from federated_agent.script_renderer import render_vantage6_script
from federated_agent.app import build_agent
from federated_agent.runs.executor import SandboxVantage6Executor
from federated_agent.runs.worker import run_worker
from federated_agent.storage import read_json, write_json


def plan(**changes):
    return ResearchPlan(**(dict(analysis_type="descriptive_statistics", variables=["age at diagnosis"],
        sites=["hospital 1", "hospital 2"], database_label="synthetic_clean", purpose="Public synthetic test") | changes))


class HTTPFixture:
    def __init__(self, body):
        self.body, self.calls, self.status, self.location = body, [], 200, None
        fixture = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass
            def respond(self):
                content = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                fixture.calls.append({"path": self.path, "body": json.loads(content) if content else None,
                                      "authorization": self.headers.get("Authorization")})
                body = fixture.body(self) if callable(fixture.body) else fixture.body
                if not isinstance(body, bytes):
                    body = json.dumps(body).encode()
                self.send_response(fixture.status)
                self.send_header("Content-Length", str(len(body)))
                if fixture.location:
                    self.send_header("Location", fixture.location)
                self.end_headers()
                self.wfile.write(body)
            do_GET = respond
            do_POST = respond
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/registry"
    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        now = datetime.now(timezone.utc)
        inventory = read_json(SOURCE_CHECKOUT / "tests/fixtures/ubuntu_data_availability_20260907.json")
        inventory.update(observed_at=(now-timedelta(minutes=1)).isoformat(),
                         expires_at=(now+timedelta(hours=2)).isoformat(), version="TEST_FIXTURE_NOT_REAL_EVIDENCE")
        self.documents = {
            "algorithms": read_json(RESOURCE_ROOT / "catalogs/algorithms.json"),
            "deployment": read_json(RESOURCE_ROOT / "config/ubuntu_sandbox.json"),
            "availability": inventory,
            "semantics": read_json(RESOURCE_ROOT / "catalogs/AYA_cancer_schema.jsonld"),
            "templates": read_json(RESOURCE_ROOT / "catalogs/execution_templates.json"),
        }
        self.release = dict(schema_version="1.0", release_id="test-1", sequence=1,
            deployment="ubuntu-synthetic-sandbox", created_at=(now-timedelta(seconds=10)).isoformat(),
            expires_at=(now+timedelta(hours=1)).isoformat(), documents=self.documents)
        self.source_path = self.root / "source.json"
        self.config_path = self.root / "config.json"
        self.config = {"schema_version": "1.0", "source": str(self.source_path),
            "deployment": "ubuntu-synthetic-sandbox", "cache_dir": str(self.root / "cache"),
            "allow_unsigned_local": True,
            "connection": {k: self.documents["deployment"][k] for k in
                           ("server_url", "server_port", "server_api", "collaboration_id")}}
        self.save()
        self.client = RegistryClient(self.config_path)

    def save(self, signature=None):
        write_json(self.source_path, {"release": self.release, "signature": signature})
        write_json(self.config_path, self.config)

    def advance(self):
        self.release["sequence"] += 1
        self.release["release_id"] = "test-" + str(self.release["sequence"])

    def test_coherent_snapshot_and_current_pointer(self):
        snapshot = self.client.sync()
        self.assertEqual(snapshot.reference.sequence, 1)
        self.assertEqual(set(ROLES), set(self.documents))
        for role in ROLES:
            self.assertEqual(read_json(snapshot.path(role)), self.documents[role])
        self.assertEqual(self.client.sync().reference, snapshot.reference)

    def test_new_capability_and_dataset_appear_without_restarting_agent(self):
        agent = build_agent(registry_source=self.config_path)
        first = agent.prepare(plan())
        self.advance()
        algorithm = self.documents["algorithms"]["algorithms"][0]
        new = deepcopy(algorithm)
        new["algorithm_id"] = "new_descriptive"
        algorithm["status"] = "disabled"
        for scope in new["availability"]:
            if scope["deployment"] == self.config["deployment"]:
                scope["database_labels"].append("synthetic_new")
        self.documents["algorithms"]["algorithms"].append(new)
        for site in self.documents["deployment"]["sites"]:
            site["database_labels"].append("synthetic_new")
        for entry in list(self.documents["availability"]["sites"]):
            if entry["database_label"] == "synthetic_clean":
                self.documents["availability"]["sites"].append({**entry, "database_label": "synthetic_new"})
        self.save()
        provider = Mock()
        provider.generate_json.return_value = plan(algorithm_id="new_descriptive", database_label="synthetic_new").model_dump()
        planner = LLMResearchPlanner(provider, retries=0)
        response = agent.respond("Use the new dataset", planner)
        self.assertEqual(response.status, "ready")
        self.assertEqual(response.prepared_task.task_spec.registry_release.sequence, 2)
        self.assertIn("synthetic_new", planner.context["sites"][0]["database_labels"])
        agent.executor = Mock()
        self.assertEqual(agent.execute_response(first, first.confirmation_id).issues[0].code, "catalog_changed")
        agent.executor.submit.assert_not_called()

    def test_semantic_update_is_loaded_in_same_release(self):
        agent = build_agent(registry_source=self.config_path)
        self.advance()
        self.documents["semantics"]["schema"]["variables"]["new_measure"] = {
            "class": "ncit:TEST_NEW", "dataType": "continuous"}
        for entry in self.documents["availability"]["sites"]:
            entry["variables"]["ncit:TEST_NEW"] = "available"
        self.save()
        task = agent.prepare(plan(variables=["new measure"]))
        self.assertEqual(task.resolved_plan.variables[0].class_id, "ncit:TEST_NEW")

    def test_bad_release_keeps_pointer_but_blocks_new_requests(self):
        old = self.client.sync()
        pointer = (self.root / "cache/current.json").read_bytes()
        self.advance()
        self.documents["availability"]["sites"][0]["variables"]["unknown:class"] = "available"
        self.save()
        with self.assertRaises(PreflightError):
            self.client.sync()
        self.assertEqual((self.root / "cache/current.json").read_bytes(), pointer)
        self.assertEqual(self.client.sync(existing=True).reference, old.reference)

    def test_rollback_and_same_sequence_mutation_rejected(self):
        original = deepcopy(self.release)
        self.client.sync()
        self.advance()
        self.save()
        self.client.sync()
        self.release = original
        self.save()
        with self.assertRaises(PreflightError): self.client.sync()
        self.release["sequence"] = 2
        self.release["release_id"] = "mutated-same-sequence"
        self.save()
        with self.assertRaises(PreflightError): self.client.sync()

    def test_cache_tampering_is_detected(self):
        snapshot = self.client.sync()
        snapshot.path("availability").write_text("{}")
        with self.assertRaises(PreflightError): self.client.sync()
        with self.assertRaises(PreflightError): self.client.sync(existing=True)

    def test_expired_and_future_release_rejected(self):
        now = datetime.now(timezone.utc)
        for created, expires in ((now-timedelta(hours=2), now-timedelta(hours=1)),
                                 (now+timedelta(hours=1), now+timedelta(hours=2))):
            with self.subTest(created=created):
                self.release.update(created_at=created.isoformat(), expires_at=expires.isoformat())
                self.save()
                with self.assertRaises(PreflightError): self.client.sync()

    def test_untrusted_connection_cannot_redirect_vantage6_credentials(self):
        self.documents["deployment"]["server_url"] = "https://untrusted.invalid"
        self.save()
        snapshot = self.client.sync()
        self.assertEqual(read_json(snapshot.path("deployment"))["server_url"], "http://127.0.0.1")

    def test_duplicate_json_keys_and_extra_roles_rejected(self):
        self.source_path.write_text('{"release":{},"release":{}}')
        with self.assertRaises(PreflightError): self.client.sync()
        self.documents["policy"] = {"mode": "allow_all"}
        self.save()
        with self.assertRaises(PreflightError): self.client.sync()

    def test_unknown_template_adapter_and_asset_traversal_rejected(self):
        descriptor = self.documents["templates"]["templates"][0]
        for change in ({"runtime_adapter": "os.system"}, {"export_asset": "../../payload.py"}):
            saved = deepcopy(descriptor)
            descriptor.update(change)
            self.save()
            with self.assertRaises(PreflightError): self.client.sync()
            descriptor.clear()
            descriptor.update(saved)

    def test_http_sync_checks_latest_at_each_request_and_fails_closed(self):
        server = HTTPFixture(self.source_path.read_bytes())
        self.addCleanup(server.close)
        self.config["source"] = server.url
        self.save()
        agent = build_agent(registry_source=self.config_path)
        self.assertEqual(agent.describe_algorithms().status, "algorithms")
        self.advance()
        self.documents["algorithms"]["algorithms"][0]["status"] = "disabled"
        self.save()
        server.body = self.source_path.read_bytes()
        self.assertEqual(agent.describe_algorithms().catalog.catalog_version, self.documents["algorithms"]["catalog_version"])
        self.assertEqual(agent.registry_reference.sequence, 2)
        server.status = 503
        self.assertEqual(agent.describe_algorithms().status, "preflight_error")
        self.assertGreaterEqual(len(server.calls), 4)

    def test_redirect_does_not_forward_registry_token(self):
        target = HTTPFixture(b"{}")
        source = HTTPFixture(b"{}")
        self.addCleanup(target.close)
        self.addCleanup(source.close)
        source.status, source.location = 302, target.url
        self.config.update(source=source.url, token_env="REGISTRY_TEST_TOKEN")
        self.save()
        with patch.dict(os.environ, {"REGISTRY_TEST_TOKEN": "TEST_ONLY_SECRET"}):
            with self.assertRaises(PreflightError): self.client.sync()
        self.assertEqual(len(target.calls), 0)

    def test_signatures_required_for_nonlocal_sources(self):
        from federated_agent.registry_sync import RegistrySource, verify_envelope
        config = RegistrySource.model_validate({**self.config, "source": "https://registry.example.org/current.json"})
        with self.assertRaises(ValueError): verify_envelope(self.source_path.read_bytes(), config)

    @unittest.skipUnless(importlib.util.find_spec("cryptography"), "Requires the registry extra")
    def test_signed_release_and_signature_tampering(self):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
        key = Ed25519PrivateKey.generate()
        self.config.update(trusted_public_key=base64.b64encode(key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)).decode(),
                           allow_unsigned_local=False)
        self.save(base64.b64encode(key.sign(canonical(self.release))).decode())
        self.assertEqual(self.client.sync().reference.sequence, 1)
        self.advance()
        self.save(base64.b64encode(b"x" * 64).decode())
        with self.assertRaises(PreflightError): self.client.sync()

    def test_publisher_produces_valid_bundle_and_refuses_version_reuse(self):
        mapping = {}
        for role, name in ROLES.items():
            write_json(self.root / name, self.documents[role])
            mapping[role] = name
        config = self.root / "publisher.json"
        write_json(config, mapping)
        publish(config, self.source_path, release_id="published-2", sequence=2, unsigned_local=True)
        self.assertEqual(self.client.sync().reference.sequence, 2)
        with self.assertRaises(ValueError):
            publish(config, self.source_path, release_id="published-2", sequence=2, unsigned_local=True)

    def test_offline_expired_cache_is_only_for_existing_task_recovery(self):
        self.release["created_at"] = (datetime.now(timezone.utc)-timedelta(hours=1)).isoformat()
        self.release["expires_at"] = (datetime.now(timezone.utc)+timedelta(minutes=1)).isoformat()
        self.save()
        original = self.client.sync()
        self.source_path.unlink()
        class Later(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime.now(timezone.utc) + timedelta(hours=2)
        with patch("federated_agent.registry_sync.datetime", Later):
            with self.assertRaises(PreflightError): self.client.sync()
            self.assertEqual(self.client.sync(existing=True).reference, original.reference)

    def test_concurrent_reader_cannot_activate_partial_release(self):
        from federated_agent.sessions import FileLease
        original = self.client.sync()
        lease = FileLease(self.root / "cache/.registry.lock")
        try:
            self.advance()
            self.save()
            with self.assertRaises(PreflightError): self.client.sync()
            self.assertEqual(read_json(self.root / "cache/current.json")["sequence"], original.reference.sequence)
        finally:
            lease.close()
        self.assertEqual(self.client.sync().reference.sequence, 2)

    def test_oversized_registry_payload_fails_without_activating(self):
        self.source_path.write_bytes(b" " * (4 * 1024 * 1024 + 1))
        with self.assertRaises(PreflightError): self.client.sync()
        self.assertFalse((self.root / "cache/current.json").exists())

    def test_remote_mode_does_not_depend_on_unrelated_local_catalog_files(self):
        agent = build_agent(self.root / "missing-deployment.json", self.root / "missing-algorithms.json",
                                 registry_source=self.config_path)
        self.assertEqual(agent.prepare(plan()).task_spec.registry_release.sequence, 1)

    def test_release_cannot_extend_stale_data_inventory(self):
        self.documents["availability"]["expires_at"] = (datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()
        self.save()
        with self.assertRaises(PreflightError): self.client.sync()

    def test_recovery_pins_original_semantics_without_rolling_back_current(self):
        original = self.client.sync()
        self.advance()
        self.documents["semantics"]["schema"]["variables"]["new_measure"] = {
            "class": "ncit:TEST_NEW", "dataType": "continuous"}
        self.save()
        self.client.sync()
        historical = self.client.sync(existing=True, reference=original.reference)
        self.assertEqual(historical.reference, original.reference)
        self.assertNotIn("new_measure", read_json(historical.path("semantics"))["schema"]["variables"])
        self.assertEqual(read_json(self.root / "cache/current.json")["sequence"], 2)

    def test_cli_resume_can_start_offline_but_cannot_use_cache_for_new_requests(self):
        from federated_agent.sessions import SessionStore
        self.client.sync()
        output = self.root / "output"
        session = SessionStore.create(output, "mock")
        name = session.state.session_id
        session.close()
        self.source_path.unlink()
        result = subprocess.run([sys.executable, "-m", "federated_agent.cli", "--chat", "--plain",
            "--provider", "demo", "--backend", "mock", "--registry-source", str(self.config_path),
            "--resume", name, "--output", str(output)], input="/history\n/algorithms\n/exit\n",
            capture_output=True, text=True, encoding="utf-8", timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("preflight", result.stdout.lower())
        self.assertFalse(list(output.rglob("submission.json")))


class PolicyAndTemplateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.agent = build_agent(RESOURCE_ROOT / "config/ubuntu_sandbox.json")
        self.prepared = self.agent.prepare(plan())
        self.config = self.root / "policy.json"
        write_json(self.config, {"mode": "allow_all", "version": "test-allow"})

    def opa(self, response):
        server = HTTPFixture(response)
        self.addCleanup(server.close)
        write_json(self.config, {"mode": "opa", "version": "test-opa", "opa_url": server.url})
        return server, FilePolicyChecker(self.config)

    def test_allow_all_and_action_specific_opa_contract(self):
        checker = FilePolicyChecker(self.config)
        self.assertTrue(enforce_policy(checker, self.prepared.task_spec, action="submit").allowed)
        server, checker = self.opa({"result": {"allowed": True, "decision_id": "public-id", "verification": "verified"}})
        decision = enforce_policy(checker, self.prepared.task_spec, action="read_result", task_id="12", subject={"authenticated": True, "id": 7})
        self.assertEqual(decision.decision_id, "public-id")
        request = server.calls[-1]["body"]["input"]
        self.assertEqual(request["action"], "read_result")
        self.assertEqual(request["task_id"], "12")
        self.assertEqual(request["subject"]["id"], 7)

    def test_bad_decisions_service_errors_and_obligations_fail_closed(self):
        server, checker = self.opa({})
        for result in ({}, {"result": True}, {"result": {"allowed": "true"}}):
            server.body = result
            with self.assertRaises(PreflightError): enforce_policy(checker, self.prepared.task_spec, action="submit")
        for result in ({"allowed": False}, {"allowed": True, "obligations": ["apply_dp"]}):
            server.body = {"result": result}
            with self.assertRaises(PolicyDeniedError): enforce_policy(checker, self.prepared.task_spec, action="submit")
        server.status = 503
        with self.assertRaises(PreflightError): enforce_policy(checker, self.prepared.task_spec, action="submit")

    def test_policy_reload_denial_blocks_before_submission(self):
        self.agent.policy_checker = FilePolicyChecker(self.config)
        self.prepared = self.agent.prepare(plan())
        self.agent.executor = Mock()
        server, _ = self.opa({"result": {"allowed": False}})
        result = self.agent.execute_response(self.prepared, self.prepared.confirmation_id)
        self.assertEqual(result.status, "permission_denied")
        self.agent.executor.submit.assert_not_called()

    def worker(self, checker_action, *, obligations=False):
        def response(_):
            action = server.calls[-1]["body"]["input"]["action"]
            return {"result": {"allowed": obligations or action != checker_action,
                               "obligations": ["redact"] if obligations and action == checker_action else []}}
        server, _ = self.opa(response)
        output = self.root / "worker"
        output.mkdir()
        write_json(output / "task_spec.json", self.prepared.task_spec.model_dump(mode="json"))
        client = Mock()
        client.util.get_server_version.return_value = {"version": "4.15.0"}
        client.collaboration.get.return_value = {"encrypted": False}
        client.task.create.return_value = {"id": 700}
        client.task.get.return_value = {"status": "completed"}
        client.result.from_task.return_value = {"private_test": "RESULT_SENTINEL"}
        run_worker(output, {"username": "fake", "password": "CREDENTIAL_SENTINEL", "_policy_config": str(self.config)},
                   client_factory=lambda _: client)
        self.assertEqual(read_json(output / "state.json")["code"], "backend_forbidden")
        self.assertFalse((output / "result.raw.json").exists())
        self.assertNotIn("CREDENTIAL_SENTINEL", json.dumps(server.calls))
        self.assertNotIn("RESULT_SENTINEL", json.dumps(server.calls))
        return client, server, output

    def test_worker_submit_denial_is_not_bypassable(self):
        client, _, _ = self.worker("submit")
        client.task.create.assert_not_called()

    def test_worker_result_read_denial_prevents_retrieval(self):
        client, _, _ = self.worker("read_result")
        client.result.from_task.assert_not_called()

    def test_worker_release_denial_prevents_raw_result_persistence(self):
        client, _, output = self.worker("release_result")
        client.result.from_task.assert_called_once()
        self.assertFalse(read_json(output / "policy_decisions.json")[-1]["allowed"])

    def test_worker_unimplemented_obligation_prevents_result_persistence(self):
        _, _, output = self.worker("release_result", obligations=True)
        self.assertTrue(read_json(output / "policy_decisions.json")[-1]["unhandled_obligations"])

    def test_template_version_change_invalidates_old_confirmation(self):
        path = self.root / "templates.json"
        document = read_json(RESOURCE_ROOT / "catalogs/execution_templates.json")
        write_json(path, document)
        self.agent.template_registry = TemplateRegistry(path)
        first = self.agent.prepare(plan())
        newer = {**document["templates"][0], "version": "1.2.0", "poll_interval_seconds": 0.25}
        document["templates"].append(newer)
        document["defaults"][newer["id"]] = "1.2.0"
        write_json(path, document)
        self.agent.executor = Mock()
        response = self.agent.execute_response(first, first.confirmation_id)
        self.assertEqual(response.issues[0].code, "confirmation_invalid")
        self.agent.executor.submit.assert_not_called()
        fresh = self.agent.prepare(plan())
        self.assertEqual(fresh.task_spec.execution_template.version, "1.2.0")
        self.assertNotEqual(fresh.confirmation_id, first.confirmation_id)

    def test_template_runtime_configuration_is_actually_used_by_worker(self):
        task = self.prepared.task_spec
        task.execution_template.poll_interval_seconds = 0.25
        output = self.root / "runtime"
        output.mkdir()
        write_json(output / "task_spec.json", task.model_dump(mode="json"))
        client = Mock()
        client.util.get_server_version.return_value = {"version": "4.15.0"}
        client.collaboration.get.return_value = {"encrypted": False}
        client.task.create.return_value = {"id": 42}
        client.task.get.side_effect = [{"status": "running"}, {"status": "completed"}]
        client.result.from_task.return_value = {"data": [{"result": "{}"}]}
        sleep = Mock()
        run_worker(output, {"username": "fake", "password": "fake"}, client_factory=lambda _: client, sleep=sleep)
        sleep.assert_called_once_with(0.25)
        self.assertEqual(read_json(output / "state.json")["status"], "completed")

    def test_installed_code_digest_tampering_blocks_execution(self):
        reference = self.prepared.task_spec.execution_template.model_copy(update={"adapter_sha256": "0" * 64})
        with self.assertRaises(ValueError): validate_installed(reference)

    def test_template_slots_are_literal_not_recursively_expanded(self):
        task = self.prepared.task_spec
        task.description = "__CONFIRMATION_ID__ __CONFIRMATION_MATERIAL__"
        script = render_vantage6_script(task)
        compile(script, "generated.py", "exec")
        self.assertIn(task.description, script)

    def test_direct_python_service_wires_policy_into_sdk_worker(self):
        self.agent.policy_checker = FilePolicyChecker(self.config)
        self.agent.executor = SandboxVantage6Executor(self.root / "backend", confirmed=True)
        self.agent._bind_executor_policy()
        self.assertEqual(self.agent.executor.policy_config, self.config.resolve())
        self.agent.policy_checker = Mock()
        self.agent.policy_checker.evaluate.return_value = PolicyDecision(allowed=True)
        with self.assertRaises(PreflightError): self.agent._bind_executor_policy()

    def test_direct_execute_cannot_bypass_result_release_policy(self):
        server, checker = self.opa(lambda _: {"result": {"allowed": server.calls[-1]["body"]["input"]["action"] != "release_result"}})
        self.agent.policy_checker = checker
        self.prepared = self.agent.prepare(plan())
        # A verified test checker avoids any real data or remote execution.
        self.agent.data_checker.check = Mock(return_value=[])
        from federated_agent.models import DataAvailabilityCheck
        self.agent.data_checker.check.return_value = [DataAvailabilityCheck(
            site=s.canonical_name, variable=v.canonical_name, state="available", evidence="metadata_snapshot",
            source={"kind": "unit_test", "uri": "test-fixture-only", "version": "1"})
            for s in self.prepared.resolved_plan.sites for v in self.prepared.resolved_plan.variables]
        with self.assertRaises(PolicyDeniedError):
            self.agent.execute(self.prepared, self.prepared.confirmation_id)

    def test_release_denial_retains_submission_identity_without_result(self):
        server, checker = self.opa(lambda _: {"result": {"allowed": server.calls[-1]["body"]["input"]["action"] != "release_result"}})
        self.agent.policy_checker = checker
        self.prepared = self.agent.prepare(plan())
        record = ExecutionRecord(task_id="701", status="completed", result={"secret_test_result": "not_returned"})
        with patch.object(self.agent, "_execute", return_value=record):
            response = self.agent.execute_response(self.prepared, self.prepared.confirmation_id)
        self.assertEqual(response.status, "permission_denied")
        self.assertEqual(response.submission_state, "submitted")
        self.assertEqual(response.execution.task_id, "701")
        self.assertIsNone(response.execution.result)

    def test_recovery_preserves_policy_audit_and_never_creates_another_task(self):
        client, _, output = self.worker("release_result")
        before = read_json(output / "policy_decisions.json")
        write_json(self.config, {"mode": "allow_all", "version": "test-recovery"})
        client.result.from_task.return_value = {"data": [{"result": "{}"}]}
        client.task.create.reset_mock()
        run_worker(output, {"username": "fake", "password": "fake", "_policy_config": str(self.config)},
                   client_factory=lambda _: client, resume=True)
        client.task.create.assert_not_called()
        after = read_json(output / "policy_decisions.json")
        self.assertEqual(after[:len(before)], before)
        self.assertTrue(after[-1]["recovery"])
        self.assertEqual(read_json(output / "state.json")["status"], "completed")

    def test_policy_transport_timeout_is_audited_and_fails_closed(self):
        _, checker = self.opa({"result": {"allowed": True}})
        events = []
        with patch("federated_agent.policy.fetch", side_effect=TimeoutError("SECRET_SENTINEL")):
            with self.assertRaises(PreflightError):
                enforce_policy(checker, self.prepared.task_spec, action="submit", audit=events.append)
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0]["policy_error"])
        self.assertNotIn("SECRET_SENTINEL", json.dumps(events))


if __name__ == "__main__":
    unittest.main()
