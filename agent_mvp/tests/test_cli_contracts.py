"""Cold-process CLI contracts and regressions introduced by study persistence."""

import ast
import io
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import PackageNotFoundError, version
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from contextlib import redirect_stdout

from federated_agent.resources import RESOURCE_ROOT, SOURCE_CHECKOUT
from federated_agent.cli import run_chat
from federated_agent.models import AgentResponse, ResearchPlan
from federated_agent.planner import DemoResearchPlanner
from federated_agent.sessions import SessionStore
from federated_agent.app import build_agent
from federated_agent.runs.executor import SandboxVantage6Executor
from federated_agent.services.artifacts import write_prepared
from federated_agent.storage import read_json, write_json


REQUEST = "Describe age at diagnosis and survival status in hospital 1 and hospital 2 using synthetic_clean."
DEMO_REQUEST = "Describe age at diagnosis and survival status in hospital 1 and hospital 2 using default."

try:
    HAS_SDK = version("vantage6-client") == "4.15.0"
except PackageNotFoundError:
    HAS_SDK = False


class CommandCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="fed agent regression ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def cli(self, *arguments, output=None, stdin=None):
        destination = output or self.root / "outputs with spaces"
        return subprocess.run([sys.executable, "-m", "federated_agent", "--provider", "demo",
            "--output", str(destination), *map(str, arguments)], input=stdin,
            capture_output=True, text=True, encoding="utf-8", timeout=40, cwd=self.root)

    def prepare(self):
        output = self.root / "prepared"
        result = self.cli("--request", DEMO_REQUEST, output=output)
        self.assertEqual(result.returncode, 0, result.stderr)
        return output, read_json(output / "prepared_task.json")

    def test_single_request_still_exports_fixed_script_and_payload(self):
        output, prepared = self.prepare()
        for name in ("response.json", "researcher_response.md", "catalog_snapshot.json", "task_spec.json", "run_vantage6_task.py"):
            self.assertTrue((output / name).exists(), name)
        self.assertEqual(prepared["task_spec"]["input"]["kwargs"]["organisation_ids"], [1, 2])
        script = (output / "run_vantage6_task.py").read_text(encoding="utf-8")
        tree = ast.parse(script)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "create"]
        self.assertEqual(len(calls), 1)
        self.assertEqual({kw.arg for kw in calls[0].keywords},
                         {"collaboration", "organizations", "name", "image", "description", "input_", "databases"})
        self.assertFalse((output / "mock_result.json").exists())

    def test_execute_prepared_still_works_in_another_process_without_llm(self):
        output, prepared = self.prepare()
        result = self.cli("--execute-prepared", output / "prepared_task.json", "--confirm", prepared["confirmation_id"],
                          "--provider", "local", output=self.root / "executed")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(read_json(self.root / "executed/response.json")["status"], "completed")
        for name in ("mock_result.json", "parsed_result.json", "result.html"):
            self.assertTrue((self.root / "executed" / name).exists())

    def test_wrong_confirmation_does_not_execute(self):
        output, _ = self.prepare()
        result = self.cli("--execute-prepared", output / "prepared_task.json", "--confirm", "wrong", output=self.root / "wrong")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(read_json(self.root / "wrong/response.json")["issues"][0]["code"], "confirmation_invalid")
        self.assertFalse((self.root / "wrong/mock_result.json").exists())

    def test_tampered_prepared_task_keeps_old_rejection_contract(self):
        output, prepared = self.prepare()
        prepared["task_spec"]["image"] = "untrusted:1"
        write_json(output / "prepared_task.json", prepared)
        result = self.cli("--execute-prepared", output / "prepared_task.json", "--confirm", prepared["confirmation_id"], output=self.root / "tampered")
        self.assertEqual(result.returncode, 2)
        self.assertFalse((self.root / "tampered/mock_result.json").exists())

    def test_single_request_execute_flag_still_requires_exact_confirmation(self):
        _, prepared = self.prepare()
        result = self.cli("--request", DEMO_REQUEST, "--execute", "--confirm", prepared["confirmation_id"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(read_json(self.root / "outputs with spaces/response.json")["status"], "completed")

    def test_missing_input_still_writes_clarification_without_submission(self):
        result = self.cli("--request", "Describe age at diagnosis.")
        self.assertEqual(result.returncode, 0, result.stderr)
        output = self.root / "outputs with spaces"
        self.assertTrue((output / "clarification.json").exists())
        self.assertEqual(read_json(output / "response.json")["status"], "needs_clarification")
        self.assertFalse((output / "prepared_task.json").exists())

    def test_unsupported_filter_keeps_nonzero_exit_and_no_script(self):
        result = self.cli("--request", DEMO_REQUEST + " Only patients under 25.")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(read_json(self.root / "outputs with spaces/response.json")["status"], "unsupported")
        self.assertFalse((self.root / "outputs with spaces/run_vantage6_task.py").exists())

    def test_catalog_commands_do_not_require_configured_llm(self):
        for flag in ("--list-algorithms", "--validate-catalog"):
            with self.subTest(flag=flag):
                result = self.cli(flag, "--provider", "local", output=self.root / flag[2:])
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(read_json(self.root / flag[2:] / "response.json")["status"], "algorithms")

    def test_benchmark_clock_override_is_not_a_runtime_option(self):
        result = self.cli("--fixture-time", "2026-09-07T12:00:00Z")
        self.assertEqual(result.returncode, 2)
        self.assertIn("unrecognized arguments", result.stderr)

    def test_benchmark_clock_is_scoped_and_restored_after_failure(self):
        spec = importlib.util.spec_from_file_location("benchmark_under_test", SOURCE_CHECKOUT / "scripts/evaluate_requests.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from federated_agent import data_availability
        original = data_availability.datetime
        fixed = datetime(2026, 9, 7, tzinfo=timezone.utc)
        with self.assertRaisesRegex(RuntimeError, "test"):
            with module.fixture_clock(fixed):
                self.assertEqual(data_availability.datetime.now(timezone.utc), fixed)
                raise RuntimeError("test")
        self.assertIs(data_availability.datetime, original)

    def test_benchmark_rejects_ambiguous_fixture_time_before_credentials(self):
        result = subprocess.run([sys.executable, str(SOURCE_CHECKOUT / "scripts/evaluate_requests.py"),
            "--fixture-time", "2026-09-07"], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 2)
        self.assertIn("explicit timezone", result.stderr)


class StateRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.agent = build_agent(RESOURCE_ROOT / "config/ubuntu_sandbox.json")
        self.prepared = self.agent.prepare(ResearchPlan(analysis_type="descriptive_statistics",
            variables=["age at diagnosis"], sites=["hospital 1", "hospital 2"], database_label="synthetic_clean", purpose="Synthetic test"))
        self.store = SessionStore.create(self.root, "vantage6")
        self.addCleanup(self.store.close)
        self.turn = self.store.new_turn("Describe age", "en")
        write_prepared(self.prepared, self.turn)
        self.store.record(AgentResponse(status="ready", message="Review", prepared_task=self.prepared), self.turn)

    def test_review_can_retry_after_known_authentication_failure(self):
        backend = self.turn / "backend"
        backend.mkdir()
        write_json(backend / "state.json", {"status": "failed", "stage": "authentication", "code": "backend_authentication", "task_id": None})
        self.store.close()
        args = SimpleNamespace(output=self.root, backend="vantage6", provider="demo", resume=self.store.path.name,
            plain=True, stream=False, timeout_seconds=5, confirm_synthetic_unencrypted_sandbox=True)
        replies = iter(["/review", "", "/exit"])
        with redirect_stdout(io.StringIO()):
            run_chat(self.agent, DemoResearchPlanner(), args, read=lambda _: next(replies))
        self.assertTrue((self.store.path / "turn-002/prepared_task.json").exists(),
                        "A failed login must not permanently prevent review of a new attempt")

    def test_cancelled_password_prompt_is_not_an_uncertain_submission(self):
        executor = SandboxVantage6Executor(self.turn / "backend", confirmed=True,
            credentials_provider=Mock(side_effect=EOFError), progress=lambda _: None)
        with self.assertRaises(EOFError):
            executor.submit(self.prepared.task_spec)
        self.assertEqual(self.store.unresolved_attempts(), [])

    def test_history_retains_task_id_even_when_response_file_is_missing(self):
        backend = self.turn / "backend"
        backend.mkdir()
        write_json(backend / "submission.json", {"task_id": "8801"})
        self.assertIn("task 8801", self.store.history()[0])

    def test_submit_intent_is_still_uncertain_even_with_early_failure_label(self):
        backend = self.turn / "backend"
        backend.mkdir()
        write_json(backend / "state.json", {"status": "failed", "stage": "authentication"})
        write_json(backend / "submission_intent.json", {"automatic_retry_allowed": False})
        self.assertEqual(self.store.unresolved_attempts(), [self.turn.name])

    def test_explicit_http_rejection_and_lost_submit_response_are_distinguished(self):
        backend = self.turn / "backend"
        backend.mkdir()
        write_json(backend / "submission_intent.json", {"automatic_retry_allowed": False})
        for code, expected in (("backend_forbidden", "not_submitted"), ("backend_timeout", "unknown"),
                               ("backend_error", "unknown")):
            with self.subTest(code=code):
                write_json(backend / "state.json", {"status": "failed", "stage": "submit", "code": code})
                self.assertEqual(self.store.attempt_state(self.turn), expected)


@unittest.skipUnless(HAS_SDK, "Requires the pinned Vantage6 SDK")
class FullCLITransportTests(unittest.TestCase):
    """Local HTTP fixtures, real CLI/provider/SDK/worker/report code, no Ubuntu."""

    def setUp(self):
        import jwt
        from vantage6.common.encryption import DummyCryptor
        self.temp = tempfile.TemporaryDirectory(prefix="fed agent full cli ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.calls = []
        self.created = []
        self.status = "completed"
        self.submit_failure = False
        self.llm_inputs = []
        self.llm_plans = []
        self.stream = False
        self.token = jwt.encode({"sub": "6"}, "unit-test-only-signing-key" * 3)
        raw = deepcopy(read_json(Path(__file__).parent / "fixtures/vantage6_s02_result.json"))
        raw["data"][0]["result"] = DummyCryptor().encrypt_bytes_to_str(raw["data"][0]["result"].encode(), None)
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, value, code=200):
                body = json.dumps(value).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fixture.calls.append(("POST", self.path))
                if self.path == "/llm":
                    fixture.llm_inputs.append(payload)
                    is_json = payload.get("response_format", {}).get("type") == "json_object"
                    value = fixture.llm_plans.pop(0) if is_json else "The task is prepared, not submitted. Review the checks and confirm before execution."
                    content = json.dumps(value) if is_json else value
                    if payload.get("stream"):
                        events = [{"choices": [{"index": 0, "delta": {"content": fragment}, "finish_reason": None}]}
                                  for fragment in (content[:len(content)//2], content[len(content)//2:])]
                        events.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
                        body = "".join("data: " + json.dumps(item) + "\n\n" for item in events) + "data: [DONE]\n\n"
                        encoded = body.encode()
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Content-Length", str(len(encoded)))
                        self.end_headers()
                        self.wfile.write(encoded)
                    else:
                        self.reply({"choices": [{"message": {"content": content}, "finish_reason": "stop"}]})
                elif self.path == "/api/token/user":
                    self.reply({"access_token": fixture.token, "refresh_token": "fake-refresh"})
                elif self.path == "/api/task":
                    fixture.created.append(payload)
                    self.reply({"error": "Server refused"}, 403) if fixture.submit_failure else self.reply({"id": 8801})
                else:
                    self.reply({"error": "Unknown endpoint"}, 404)

            def do_GET(self):
                fixture.calls.append(("GET", self.path))
                path = self.path.split("?", 1)[0]
                values = {"/api/version": {"version": "4.15.0"},
                    "/api/user/6": {"firstname": "test", "organization": {"id": 1}},
                    "/api/organization/1": {"name": "test", "public_key": None},
                    "/api/collaboration/1": {"id": 1, "encrypted": False},
                    "/api/task/8801": {"id": 8801, "status": fixture.status},
                    "/api/blobstream/status": {"blob_store_enabled": False}, "/api/result": raw}
                self.reply(values.get(path, {"error": "not found"}), 200 if path in values else 404)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)
        profile = read_json(RESOURCE_ROOT / "config/ubuntu_sandbox.json")
        profile["server_port"] = self.server.server_port
        self.deployment = self.root / "deployment.json"
        write_json(self.deployment, profile)
        inventory = read_json(SOURCE_CHECKOUT / "tests/fixtures/ubuntu_data_availability_20260907.json")
        inventory.update(version="test-only-not-hospital-evidence", observed_at=(datetime.now(timezone.utc)-timedelta(minutes=5)).isoformat(),
                         expires_at=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat())
        self.inventory = self.root / "test-inventory.json"
        write_json(self.inventory, inventory)

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def full_cli(self, inputs, *, resume=None, stream=False):
        args = [sys.executable, "-m", "federated_agent", "--chat", "--plain", "--provider", "local",
                "--model", "http-fixture", "--llm-endpoint", f"http://127.0.0.1:{self.server.server_port}/llm",
                "--backend", "vantage6", "--confirm-synthetic-unencrypted-sandbox", "--timeout-seconds", "15",
                "--deployment-config", str(self.deployment), "--data-availability-config", str(self.inventory),
                "--output", str(self.root / "sessions")]
        if not stream:
            args.append("--no-stream")
        if resume:
            args.extend(["--resume", resume])
        env = os.environ | {"VANTAGE6_USERNAME": "fake-user", "VANTAGE6_PASSWORD": "SDK_PASSWORD_SENTINEL",
                            "LOCAL_LLM_API_KEY": "MODEL_KEY_SENTINEL", "PYTHONUTF8": "1"}
        result = subprocess.run(args, input="\n".join(inputs) + "\n", capture_output=True, text=True,
                                encoding="utf-8", env=env, timeout=40, cwd=self.root)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        return result

    def plan(self, **changes):
        return ResearchPlan(**(dict(analysis_type="descriptive_statistics", variables=["age at diagnosis", "survival status"],
            sites=["hospital 1", "hospital 2"], database_label="synthetic_clean", purpose=REQUEST) | changes)).model_dump()

    def session(self):
        return next((self.root / "sessions").glob("session-*"))

    def test_original_cli_submit_wait_result_chart_and_readonly_recovery(self):
        self.llm_plans = [self.plan()]
        result = self.full_cli([REQUEST, "confirm", "/exit"])
        self.assertEqual(len(self.created), 1)
        turn = self.session() / "turn-001"
        self.assertEqual(read_json(turn / "execution.json")["status"], "completed")
        for name in ("parsed_result.json", "result.html", "result.png", "run_vantage6_task.py"):
            self.assertTrue((turn / name).is_file(), name)
        parsed = read_json(turn / "parsed_result.json")
        self.assertEqual(next(row["value"] for row in parsed["numerical"] if row["statistic"] == "count"), 300)
        self.assertAlmostEqual(next(row["value"] for row in parsed["numerical"] if row["statistic"] == "mean"),
                               23.7783666667, places=7)
        from PIL import Image
        with Image.open(turn / "result.png") as chart:
            self.assertGreater(chart.width, 100)
            self.assertGreater(chart.height, 100)
            self.assertNotEqual(*chart.convert("L").getextrema())
        self.assertEqual(self.created[0]["collaboration_id"], 1)
        self.assertEqual(self.created[0]["image"], "ghcr.io/strongaya/v6-descriptive-statistics:v2.1.0")
        self.assertNotIn("SDK_PASSWORD_SENTINEL", json.dumps(self.llm_inputs))
        self.assertNotIn("SDK_PASSWORD_SENTINEL", result.stdout + result.stderr)
        for path in turn.rglob("*.json"):
            self.assertNotIn("SDK_PASSWORD_SENTINEL", path.read_text(encoding="utf-8"))
            self.assertNotIn("MODEL_KEY_SENTINEL", path.read_text(encoding="utf-8"))
        http_calls = len(self.calls)
        standalone = subprocess.run([sys.executable, str(turn / "run_vantage6_task.py")], input="",
                                    capture_output=True, text=True, timeout=15)
        self.assertNotEqual(standalone.returncode, 0)
        self.assertIn("cannot bypass", standalone.stderr)
        self.assertEqual(len(self.calls), http_calls)
        calls = len(self.llm_inputs)
        self.full_cli(["/recover turn-001", "/exit"], resume=self.session().name)
        self.assertEqual(len(self.created), 1)
        self.assertEqual(len(self.llm_inputs), calls)

    def test_streamed_clarification_and_cancel_do_not_submit(self):
        self.llm_plans = [self.plan(sites=[]), {"changes": {"sites": ["hospital 1", "hospital 2"]}}]
        self.full_cli(["Describe age and survival status", "hospital 1 and hospital 2", "", "/exit"], stream=True)
        self.assertEqual(self.created, [])
        prepared = read_json(self.session() / "turn-002/prepared_task.json")
        self.assertEqual(prepared["task_spec"]["input"]["kwargs"]["organisation_ids"], [1, 2])
        self.assertTrue(read_json(self.session() / "turn-002/stream.json")["complete"])
        self.assertEqual(len(self.llm_inputs), 4)

    def test_server_denial_does_not_retry_or_generate_report(self):
        self.submit_failure = True
        self.llm_plans = [self.plan()]
        self.full_cli([REQUEST, "confirm", "/review", "", "/exit"])
        self.assertEqual(len(self.created), 1)
        turn = self.session() / "turn-001"
        self.assertEqual(read_json(turn / "response.json")["issues"][0]["code"], "backend_forbidden")
        self.assertFalse((turn / "result.html").exists())
        self.assertTrue((self.session() / "turn-002/prepared_task.json").exists())

    def test_remote_failure_still_stops_without_a_success_report(self):
        self.status = "crashed"
        self.llm_plans = [self.plan()]
        self.full_cli([REQUEST, "confirm", "/exit"])
        self.assertEqual(len(self.created), 1)
        turn = self.session() / "turn-001"
        response = read_json(turn / "response.json")
        self.assertEqual(response["submission_state"], "failed")
        self.assertEqual(response["issues"][0]["code"], "backend_task_failed")
        self.assertFalse((turn / "result.html").exists())
        self.assertFalse(any(path.startswith("/api/result") for _, path in self.calls))

    def test_current_time_expiry_blocks_before_any_backend_authentication(self):
        document = read_json(self.inventory)
        document["expires_at"] = (datetime.now(timezone.utc)-timedelta(minutes=1)).isoformat()
        write_json(self.inventory, document)
        self.llm_plans = [self.plan()]
        self.full_cli([REQUEST, "confirm", "/exit"])
        self.assertFalse(any(path.startswith("/api/") for _, path in self.calls))
        self.assertEqual(read_json(self.session() / "turn-001/response.json")["issues"][0]["code"], "data_verification_required")

    def test_greeting_and_catalog_keep_pending_clarification_across_processes(self):
        self.llm_plans = [self.plan(sites=[])]
        self.full_cli(["Describe age and survival status", "hi", "/algorithms", "/exit"])
        self.assertEqual(len(self.llm_inputs), 1)
        self.llm_plans = [{"changes": {"sites": ["hospital 1", "hospital 2"]}}]
        self.full_cli(["hospital 1 and hospital 2", "", "/exit"], resume=self.session().name)
        self.assertEqual(self.created, [])
        self.assertEqual(read_json(self.session() / "turn-004/response.json")["status"], "ready")

    def test_two_explicit_new_studies_remain_independent(self):
        self.llm_plans = [self.plan(), self.plan(variables=["survival status"], sites=["hospital 3"])]
        self.full_cli([REQUEST, "", "/new", "Describe survival status in hospital 3", "", "/exit"])
        prepared = read_json(self.session() / "turn-002/prepared_task.json")
        self.assertEqual(prepared["task_spec"]["input"]["kwargs"]["organisation_ids"], [3])
        self.assertEqual(set(prepared["task_spec"]["input"]["kwargs"]["variables_to_describe"]), {"ncit:C25717"})
        self.assertEqual(self.created, [])


if __name__ == "__main__":
    unittest.main()
