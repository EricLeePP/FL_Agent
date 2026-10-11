"""Parent-side control of the trusted worker process for the synthetic sandbox.

Credentials are read here and handed to the worker once over stdin. From then on
the parent follows the worker only through RunStore. The worker's own log and any
crash output go to backend/worker.log.
"""

import json
import logging
import os
import subprocess
import sys
import time
from getpass import getpass
from pathlib import Path

from ..adapters.vantage6 import validate_sandbox
from ..capabilities import ExecutorCapabilities
from ..confirmation import task_confirmation_id
from ..errors import BackendExecutionError
from ..executor import _failed_result
from ..models import ExecutionRecord, ExecutionStatus
from ..observability import log_incident
from ..policy import DEFAULT_POLICY_CONFIG, PolicyConfiguration
from ..storage import FileLease, LockBusyError, is_locked
from ..trusted_http import strict_json
from .store import EXECUTION_LOCK, WORKER_LOCK, RunStore

logger = logging.getLogger(__name__)
WORKER_MODULE = "federated_agent.runs.worker"
SECRET_ENV_WORDS = ("PASSWORD", "API_KEY", "TOKEN", "SECRET")


def credentials_from_terminal():
    username = os.environ.get("VANTAGE6_USERNAME") or input("Vantage6 username [sandbox_researcher]: ").strip() or "sandbox_researcher"
    password = os.environ.get("VANTAGE6_PASSWORD") or getpass("Vantage6 password (not your SSH password): ")
    return {"username": username, "password": password}


class SandboxVantage6Executor:
    """Only the trusted SDK worker executes; LLM output is never run as code."""

    capabilities = ExecutorCapabilities(trusted_worker=True, recovery=True, closeable=True)

    def __init__(self, output, *, confirmed=False, timeout_seconds=600,
                 credentials_provider=credentials_from_terminal, progress=print):
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.output = Path(output).resolve()
        self.store = RunStore(self.output)
        self.confirmed = confirmed
        self.timeout_seconds = timeout_seconds
        self.credentials_provider = credentials_provider
        self.progress = progress
        self.process = None
        self.task_id = None
        self.started = None
        self.last_status = None
        self.lease = None
        self.policy_config = DEFAULT_POLICY_CONFIG

    def submit(self, task_spec):
        validate_sandbox(task_spec, self.confirmed)
        try:
            self.store.create(task_spec)
        except FileExistsError as exc:
            raise BackendExecutionError("duplicate_execution", stage="preflight") from exc
        return self._start_worker()

    def recover(self, task_spec):
        """Read an existing remote task. This path can never call task.create."""
        validate_sandbox(task_spec, self.confirmed)
        try:
            saved = self.store.saved_task()
            if task_confirmation_id(saved) != task_confirmation_id(task_spec):
                raise ValueError("Task mismatch")
        except (OSError, ValueError):
            raise BackendExecutionError("recovery_mismatch", stage="preflight") from None
        try:
            submission = self.store.submission() or {}
            task_id = submission["task_id"]
            if not str(task_id).isdigit() or int(task_id) <= 0:
                raise ValueError("Invalid ID")
            if submission.get("confirmation_id", task_confirmation_id(saved)) != task_confirmation_id(saved):
                raise ValueError("Submission mismatch")
        except (OSError, ValueError, KeyError):
            raise BackendExecutionError("submission_unresolved", stage="preflight") from None
        self.task_id = str(task_id)
        self._start_worker(resume=True)
        return self.wait_for_results(self.task_id)

    def _start_worker(self, *, resume=False):
        env = {key: value for key, value in os.environ.items()
               if not any(word in key.upper() for word in SECRET_ENV_WORDS)}
        try:
            try:
                self.lease = FileLease(self.store.path(EXECUTION_LOCK))
                if is_locked(self.store.path(WORKER_LOCK)):
                    raise LockBusyError("worker running")
            except LockBusyError:
                raise BackendExecutionError("execution_busy", stage="preflight") from None
            if not resume:
                # No worker or network request exists yet. A cancelled prompt is
                # safely retryable, unlike a lost response after task.create.
                self.store.set_state("waiting_for_credentials", stage="authentication", task_id=None)
            credentials = self.credentials_provider()
            if self.policy_config:
                config = PolicyConfiguration.model_validate(strict_json(Path(self.policy_config).read_bytes()))
                credentials["_policy_config"] = str(self.policy_config)
                if config.token_env and config.token_env in os.environ:
                    env[config.token_env] = os.environ[config.token_env]
            self.started = time.monotonic()
            self.progress("Vantage6: reconnecting to the recorded task..." if resume else "Vantage6: connecting and authenticating...")
            if resume:
                self.store.record_recovery(self.task_id)
            self.store.set_state("reconnecting" if resume else "connecting", task_id=self.task_id)
            with self.store.log_path.open("ab") as log:
                self.process = subprocess.Popen(
                    [sys.executable, "-m", WORKER_MODULE, str(self.output)] + (["--resume"] if resume else []),
                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log,
                    text=True, encoding="utf-8", env=env,
                )
            self.process.stdin.write(json.dumps(credentials))
            self.process.stdin.close()
            del credentials
            if resume:
                return
            while True:
                state = self._poll()
                if self.task_id:
                    self.progress(f"Vantage6: submitted task {self.task_id}. Waiting for completion...")
                    return ExecutionRecord(task_id=self.task_id, status=ExecutionStatus.submitted)
                if state.get("status") == "completed":
                    raise BackendExecutionError("backend_error")
                time.sleep(0.1)
        except BaseException:
            self.close()
            raise

    def _poll(self):
        submission = self.store.submission()
        if submission is not None:
            self.task_id = str(submission["task_id"])
        state = self.store.state()
        if state.get("status") == "failed":
            raise BackendExecutionError(state["code"], task_id=self.task_id, stage=state["stage"],
                                        incident_id=state.get("incident_id"))
        if state.get("status") == "completed":
            return state
        if time.monotonic() - self.started >= self.timeout_seconds:
            self.close()
            self.store.record_timeout(self.task_id)
            raise BackendExecutionError("backend_timeout", task_id=self.task_id,
                                        stage="result" if self.task_id else "submit")
        if self.process.poll() is not None:
            # The child may have atomically published its final state after our read.
            final = self.store.state()
            if final != state:
                return self._poll()
            incident = log_incident(logger, f"Worker exited without a final state; see {self.store.log_path}")
            raise BackendExecutionError("backend_error", task_id=self.task_id, incident_id=incident)
        status = state.get("status")
        if status and status != self.last_status:
            self.progress(f"Vantage6: {status}")
            self.last_status = status
        return state

    def wait_for_results(self, task_id):
        try:
            while self._poll().get("status") != "completed":
                time.sleep(0.2)
            return self.get_result(task_id)
        finally:
            self.close()

    def get_status(self, task_id):
        state = self._poll()
        return ExecutionRecord(task_id=task_id, status=(ExecutionStatus.completed
            if state.get("status") == "completed" else ExecutionStatus.running))

    def get_result(self, task_id):
        raw = self.store.result()
        if _failed_result(raw):
            raise BackendExecutionError("backend_task_failed", task_id=task_id, stage="result")
        return ExecutionRecord(task_id=task_id, status=ExecutionStatus.completed, result=raw)

    def close(self):
        try:
            if self.process is not None:
                if self.process.poll() is None:
                    self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
        finally:
            if self.lease is not None:
                self.lease.close()
                self.lease = None
