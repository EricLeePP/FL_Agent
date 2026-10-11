"""Execution interfaces and a deterministic mock Vantage6 backend."""

import json
import logging
from copy import deepcopy
from typing import Any, Protocol
from uuid import uuid4
from urllib.error import HTTPError

from .capabilities import ExecutorCapabilities
from .errors import BackendExecutionError
from .observability import log_incident
from .runs.submission import submit_task
from .models import ExecutionRecord, ExecutionStatus, Vantage6TaskSpec

logger = logging.getLogger(__name__)


def backend_error(exc, *, stage, task_id=None):
    if isinstance(exc, BackendExecutionError):
        return exc
    status = exc.code if isinstance(exc, HTTPError) else getattr(getattr(exc, "response", None), "status_code", None)
    code = {401: "backend_authentication", 403: "backend_forbidden", 404: "backend_not_found"}.get(status, "backend_error")
    if isinstance(exc, TimeoutError):
        code = "backend_timeout"
    if code != "backend_error":
        logger.info("Execution backend reported %s during %s", code, stage)
        return BackendExecutionError(code, task_id=task_id, stage=stage)
    incident = log_incident(logger, f"Execution backend failed during {stage}", exc)
    return BackendExecutionError(code, task_id=task_id, stage=stage, incident_id=incident)


def _failed_result(value, depth=0):
    if depth > 8:
        return True
    if value is None or value == []:
        return True
    if isinstance(value, str):
        try:
            return _failed_result(json.loads(value), depth + 1)
        except ValueError:
            return False
    if isinstance(value, list):
        return any(_failed_result(item, depth + 1) for item in value)
    if isinstance(value, dict):
        if value.get("error") or value.get("errors"):
            return True
        if str(value.get("status", "")).lower() in {"failed", "crashed", "killed"}:
            return True
        for key in ("data", "result"):
            if key in value:
                return _failed_result(value[key], depth + 1)
    return False


class Vantage6Executor(Protocol):
    def submit(self, task_spec: Vantage6TaskSpec) -> ExecutionRecord:
        ...

    def get_status(self, task_id: str) -> ExecutionRecord:
        ...

    def get_result(self, task_id: str) -> ExecutionRecord:
        ...

    def wait_for_results(self, task_id: str) -> ExecutionRecord:
        ...


class MockVantage6Executor:
    capabilities = ExecutorCapabilities(simulated=True)

    def __init__(self) -> None:
        self._tasks: dict[str, ExecutionRecord] = {}

    def submit(self, task_spec: Vantage6TaskSpec) -> ExecutionRecord:
        task_id = f"mock-{uuid4().hex[:12]}"
        record = ExecutionRecord(
            task_id=task_id,
            status=ExecutionStatus.submitted,
        )
        self._tasks[task_id] = record
        return record.model_copy(deep=True)

    def get_status(self, task_id: str) -> ExecutionRecord:
        return deepcopy(self._tasks[task_id])

    def get_result(self, task_id: str) -> ExecutionRecord:
        return deepcopy(self._tasks[task_id])

    def wait_for_results(self, task_id: str) -> ExecutionRecord:
        record = self._tasks[task_id]
        record.status = ExecutionStatus.running
        record.result = self._mock_result()
        record.status = ExecutionStatus.completed
        return record.model_copy(deep=True)

    @staticmethod
    def _mock_result() -> dict[str, str]:
        numerical = {
            "variable": {"0": "ncit:C156420", "1": "ncit:C156420"},
            "statistic": {"0": "count", "1": "mean"},
            "value": {"0": 128, "1": 27.4},
        }
        categorical = {
            "variable": {"0": "ncit:C25717", "1": "ncit:C25717"},
            "value": {"0": "alive", "1": "deceased"},
            "count": {"0": 101, "1": 27},
        }
        return {
            "numerical_general_statistics": json.dumps(numerical),
            "categorical_general_statistics": json.dumps(categorical),
        }


class ClassicVantage6Executor:
    """Vantage6 4.15 client adapter with an externally authenticated client."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self._tasks: dict[str, ExecutionRecord] = {}

    def submit(self, task_spec: Vantage6TaskSpec) -> ExecutionRecord:
        task = submit_task(self.client, task_spec.to_client_payload())
        task_id = str(task["id"])
        record = ExecutionRecord(task_id=task_id, status=ExecutionStatus.submitted)
        self._tasks[task_id] = record
        return record.model_copy(deep=True)

    def get_status(self, task_id: str) -> ExecutionRecord:
        task = self.client.task.get(int(task_id) if task_id.isdigit() else task_id)
        status = str(task.get("status", "")).lower()
        record = self._tasks[task_id]
        if status in {"failed", "crashed", "killed"}:
            record.status = ExecutionStatus.failed
            record.error = task.get("error") or task.get("message")
        elif status in {"completed", "complete", "finished"} or task.get("complete"):
            record.status = ExecutionStatus.completed
        else:
            record.status = ExecutionStatus.running
        return record.model_copy(deep=True)

    def get_result(self, task_id: str) -> ExecutionRecord:
        lookup_id: int | str = int(task_id) if task_id.isdigit() else task_id
        record = self._tasks[task_id]
        record.result = self.client.result.from_task(task_id=lookup_id)
        if _failed_result(record.result):
            record.status = ExecutionStatus.failed
            record.error = "backend_task_failed"
            record.result = None
            return record.model_copy(deep=True)
        record.status = ExecutionStatus.completed
        return record.model_copy(deep=True)

    def wait_for_results(self, task_id: str) -> ExecutionRecord:
        record = self._tasks[task_id]
        record.status = ExecutionStatus.running
        lookup_id: int | str = int(task_id) if task_id.isdigit() else task_id
        self.client.wait_for_results(lookup_id)
        return self.get_result(task_id)
