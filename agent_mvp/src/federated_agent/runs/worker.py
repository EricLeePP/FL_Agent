"""The trusted worker process: authenticate, submit, wait, retrieve and save.

This module, runs/submission.py, runtime_adapters.py and adapters/vantage6.py form
the code identity bound into new task confirmations (execution_templates.WORKER_SOURCES).
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from ..adapters.vantage6 import SUPPORTED_VERSION, create_client, validate_sandbox
from ..confirmation import task_confirmation_id
from ..errors import AgentError, BackendExecutionError, PolicyDeniedError, PreflightError
from ..execution_templates import validate_installed
from ..executor import _failed_result
from ..observability import configure_logging, log_incident
from ..policy import DEFAULT_POLICY_CONFIG, FilePolicyChecker, enforce_policy
from ..runtime_adapters import runtime_adapter
from ..storage import FileLease
from .store import WORKER_LOCK, RunStore
from .submission import submit_task

logger = logging.getLogger(__name__)
TIMEOUT_EXCEPTIONS = {"Timeout", "ReadTimeout", "ConnectTimeout"}


def _adapter_for(task, *, resume):
    if task.execution_template is None:
        return runtime_adapter("vantage6_userclient_v1")
    try:
        return validate_installed(task.execution_template)
    except ValueError:
        if not resume:
            raise BackendExecutionError("confirmation_invalid", stage="preflight") from None
        # Read-only recovery of an already submitted task must survive worker upgrades.
        logger.warning("Recovering a task prepared by a different worker version with the installed runtime adapter")
        return runtime_adapter(task.execution_template.runtime_adapter)


def _poll_interval(task):
    return task.execution_template.poll_interval_seconds if task.execution_template else 3


def run_worker(output, credentials, client_factory=create_client, sleep=time.sleep, *, resume=False):
    store = RunStore(output)
    lease = FileLease(store.path(WORKER_LOCK))
    stage = "preflight"
    task_id = None
    events = []
    policy_path = credentials.pop("_policy_config", None)
    checker = FilePolicyChecker(policy_path or DEFAULT_POLICY_CONFIG)

    def audit(event):
        events.append({**event, "recorded_at": datetime.now(timezone.utc).isoformat(), "recovery": resume})
        store.write_audit(events)

    try:
        if resume:
            events = store.audit()
        task = store.saved_task()
        validate_sandbox(task, True)
        adapter = _adapter_for(task, resume=resume)
        if not resume and (store.has_submission() or store.has_intent()):
            raise BackendExecutionError("submission_unresolved")
        if resume:
            try:
                saved = store.submission() or {}
                task_id = str(saved["task_id"])
                if not task_id.isdigit() or int(task_id) <= 0:
                    raise ValueError()
                if saved.get("confirmation_id", task_confirmation_id(task)) != task_confirmation_id(task):
                    raise ValueError()
            except (OSError, KeyError, ValueError):
                raise BackendExecutionError("submission_unresolved") from None
        client = client_factory(task.connection)
        version = client.util.get_server_version()
        if version.get("version") != SUPPORTED_VERSION:
            raise BackendExecutionError("backend_configuration")
        stage = "authentication"
        client.authenticate(credentials["username"], credentials["password"])
        credentials.clear()
        if not client.token or client.whoami is None:
            raise BackendExecutionError("backend_authentication")
        subject = {"authenticated": True}
        for name in ("id", "organization_id"):
            value = getattr(client.whoami, name, None)
            if type(value) in (int, str):
                subject[name] = value
        stage = "preflight"
        collaboration = client.collaboration.get(task.collaboration_id)
        if collaboration.get("encrypted") is not False:
            raise BackendExecutionError("backend_configuration")
        adapter.configure(client)
        if not resume:
            enforce_policy(checker, task, action="submit", subject=subject, audit=audit)
            stage = "submit"
            store.set_state("submitting")
            payload = adapter.payload(task)
            store.record_intent(task_confirmation_id(task))
            submission = submit_task(client, payload)
            if type(submission.get("id")) is not int or submission["id"] <= 0:
                raise BackendExecutionError("backend_error")
            task_id = str(submission["id"])
            store.record_submission(task_id, task_confirmation_id(task))
            logger.info("Submitted Vantage6 task %s", task_id)
        stage = "result"
        enforce_policy(checker, task, action="read_result", task_id=task_id, subject=subject, audit=audit)
        while True:
            status = client.task.get(int(task_id))
            state = str(status.get("status", "")).lower()
            if state in {"failed", "crashed", "killed"}:
                raise BackendExecutionError("backend_task_failed")
            if state == "completed":
                break
            store.set_state("running", task_id=task_id)
            sleep(_poll_interval(task))
        enforce_policy(checker, task, action="read_result", task_id=task_id, subject=subject, audit=audit)
        raw = adapter.result(client, task_id)
        if _failed_result(raw):
            raise BackendExecutionError("backend_task_failed")
        enforce_policy(checker, task, action="release_result", task_id=task_id, subject=subject, audit=audit)
        store.record_result(raw)
        store.set_state("completed", task_id=task_id)
        logger.info("Vantage6 task %s completed and its result was saved", task_id)
    except Exception as exc:
        code = exc.code if isinstance(exc, BackendExecutionError) else "backend_error"
        if isinstance(exc, PolicyDeniedError):
            code = "backend_forbidden"
        elif isinstance(exc, PreflightError):
            code = "preflight_error"
        if isinstance(exc, TimeoutError) or type(exc).__name__ in TIMEOUT_EXCEPTIONS:
            code = "backend_timeout"
        elif stage == "authentication" and code == "backend_error":
            code = "backend_authentication"
        incident = None
        if isinstance(exc, AgentError):
            logger.info("Worker stopped during %s with %s", stage, code)
        else:
            incident = log_incident(logger, f"Worker failed unexpectedly during {stage}", exc)
        store.record_failure(code, stage=stage, task_id=task_id, incident_id=incident)
    finally:
        credentials.clear()
        lease.close()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    output = Path(argv[0])
    configure_logging(1, RunStore(output).log_path, console=False)
    run_worker(output, json.load(sys.stdin), resume="--resume" in argv[1:])


if __name__ == "__main__":
    main()
