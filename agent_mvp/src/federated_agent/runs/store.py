"""Single owner of a task's backend/ directory: the submission state machine on disk.

The parent process, the trusted worker, saved-session history and the command line
read and write these files only through RunStore, so every attempt stays readable
and recoverable.
"""

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from ..errors import AUTHORIZATION_FAILURES
from ..models import Vantage6TaskSpec
from ..storage import is_locked, read_json, write_json

TASK_SPEC = "task_spec.json"
TASK_PAYLOAD = "task_payload.json"
STATE = "state.json"
INTENT = "submission_intent.json"
SUBMISSION = "submission.json"
RESULT = "result.raw.json"
POLICY_AUDIT = "policy_decisions.json"
TIMEOUT = "timeout.json"
WORKER_LOG = "worker.log"
EXECUTION_LOCK = ".execution.lock"
WORKER_LOCK = ".worker.lock"


class Attempt(str, Enum):
    NOT_SUBMITTED = "not_submitted"
    SUBMITTED = "submitted"
    COMPLETED = "completed"
    FAILED = "failed"
    UNKNOWN = "unknown"


def _now():
    return datetime.now(timezone.utc).isoformat()


class RunStore:
    def __init__(self, root):
        self.root = Path(root)

    def path(self, name):
        return self.root / name

    @property
    def log_path(self):
        return self.path(WORKER_LOG)

    def exists(self):
        return self.root.is_dir()

    def create(self, task_spec):
        """Create the directory of a NEW attempt. FileExistsError means one already exists."""
        self.root.mkdir(parents=True, exist_ok=False)
        write_json(self.path(TASK_SPEC), task_spec.model_dump(mode="json"))
        write_json(self.path(TASK_PAYLOAD), task_spec.to_client_payload())

    def set_state(self, status, **fields):
        write_json(self.path(STATE), {"status": status, **fields})

    def record_failure(self, code, *, stage, task_id=None, incident_id=None):
        fields = {"code": code, "stage": stage, "task_id": task_id}
        if incident_id:
            fields["incident_id"] = incident_id
        self.set_state("failed", **fields)

    def record_intent(self, confirmation_id):
        write_json(self.path(INTENT), {"confirmation_id": confirmation_id, "started_at": _now(),
                                       "automatic_retry_allowed": False})

    def record_submission(self, task_id, confirmation_id):
        write_json(self.path(SUBMISSION), {"task_id": task_id, "confirmation_id": confirmation_id,
                                           "submitted_at": _now()})

    def record_result(self, raw):
        write_json(self.path(RESULT), raw)

    def record_timeout(self, task_id):
        write_json(self.path(TIMEOUT), {"task_id": task_id, "remote_task_cancelled": False,
                                        "submission_may_have_succeeded": True})

    def record_recovery(self, task_id):
        write_json(self.path(f"recovery-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.json"),
                   {"task_id": task_id, "mode": "read_existing_task", "started_at": _now()})

    def write_audit(self, events):
        write_json(self.path(POLICY_AUDIT), events)

    def state(self) -> dict:
        path = self.path(STATE)
        return read_json(path) if path.exists() else {}

    def saved_task(self) -> Vantage6TaskSpec:
        return Vantage6TaskSpec.model_validate(read_json(self.path(TASK_SPEC)))

    def submission(self) -> dict | None:
        path = self.path(SUBMISSION)
        return read_json(path) if path.exists() else None

    def has_submission(self) -> bool:
        return self.path(SUBMISSION).exists()

    def has_intent(self) -> bool:
        return self.path(INTENT).exists()

    def result(self):
        return read_json(self.path(RESULT))

    def audit(self) -> list:
        path = self.path(POLICY_AUDIT)
        if not path.exists():
            return []
        events = read_json(path)
        if not isinstance(events, list) or any(not isinstance(item, dict) for item in events):
            raise ValueError("Invalid policy audit history")
        return events

    def worker_alive(self) -> bool:
        return is_locked(self.path(WORKER_LOCK))

    def attempt(self) -> Attempt:
        """Classify an attempt conservatively; anything ambiguous blocks a new submission."""
        try:
            state = read_json(self.path(STATE))
            if not isinstance(state, dict):
                return Attempt.UNKNOWN
            if state.get("status") == "completed":
                return Attempt.COMPLETED
            if state.get("code") == "backend_task_failed":
                return Attempt.FAILED
            if self.has_submission():
                return Attempt.SUBMITTED
            if (state.get("status") == "failed" and state.get("stage") == "submit"
                    and state.get("code") in AUTHORIZATION_FAILURES):
                return Attempt.NOT_SUBMITTED
            if not self.has_intent():
                if state.get("status") == "waiting_for_credentials" and state.get("stage") == "authentication":
                    return Attempt.NOT_SUBMITTED
                if state.get("status") == "failed" and state.get("stage") in {"preflight", "authentication"}:
                    return Attempt.NOT_SUBMITTED
        except (OSError, ValueError):
            pass
        return Attempt.UNKNOWN
