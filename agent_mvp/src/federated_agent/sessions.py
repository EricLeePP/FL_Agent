"""Local study state, atomic checkpoints and single-writer session ownership."""

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import Field

from .runs.store import RunStore
from .storage import FileLease, LockBusyError, read_json, write_json
from .models import AgentResponse, ResearchPlan, StrictModel


class SessionError(ValueError):
    pass


class StudySession(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    dataset_selection_version: Literal["explicit-v1"] = "explicit-v1"
    session_id: str
    backend: Literal["mock", "vantage6"]
    created_at: str
    updated_at: str
    language: Literal["en", "zh"] = "en"
    revision: int = Field(default=0, ge=0)
    current_plan: ResearchPlan | None = None
    pending: AgentResponse | None = None
    active_turn: str | None = None
    last_response: AgentResponse | None = None


def _lease(path):
    try:
        return FileLease(path / ".session.lock")
    except LockBusyError as exc:
        raise SessionError(str(exc)) from None


def _now():
    return datetime.now(timezone.utc).isoformat()


def _public_response(response):
    # Research plans and diagnostics may persist; results never enter model memory.
    data = response.model_dump(mode="json", exclude={"catalog_document", "catalog_content"})
    if data.get("execution"):
        data["execution"]["result"] = None
    return AgentResponse.model_validate(data)


class SessionStore:
    def __init__(self, root, path, state, lease):
        self.root, self.path, self.state, self.lease = root, path, state, lease

    @classmethod
    def create(cls, root, backend):
        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        name = f"session-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}"
        path = root / name
        path.mkdir()
        state = StudySession(session_id=name, backend=backend, created_at=_now(), updated_at=_now(),
                             dataset_selection_version="explicit-v1")
        store = cls(root, path, state, _lease(path))
        try:
            store.save()
        except OSError:
            store.close()
            raise
        return store

    @classmethod
    def open(cls, root, name, backend):
        root = Path(root).resolve()
        if not re.fullmatch(r"session-[A-Za-z0-9-]+", name):
            raise SessionError("Use a session ID from /sessions, not a filesystem path.")
        path = (root / name).resolve()
        if path.parent != root or not path.is_dir():
            raise SessionError("Session not found in this output directory.")
        lease = _lease(path)
        try:
            state = StudySession.model_validate(read_json(path / "session.json"))
            if state.session_id != name or state.backend != backend:
                raise SessionError("Session/backend mismatch. Restart with the original backend.")
            if state.active_turn:
                cls.turn_path(path, state.active_turn)
            store = cls(root, path, state, lease)
            store.save()
            return store
        except (OSError, ValueError, KeyError, TypeError):
            lease.close()
            raise SessionError("Cannot resume: invalid session, missing files, or backend mismatch.") from None

    @staticmethod
    def turn_path(session, name):
        if not re.fullmatch(r"turn-\d{3,}", name):
            raise SessionError("Use a turn ID from /history.")
        path = (session / name).resolve()
        if path.parent != session.resolve() or not path.is_dir():
            raise SessionError("Turn not found in this session.")
        return path

    def save(self):
        self.state.updated_at = _now()
        write_json(self.path / "session.json", self.state.model_dump(mode="json"))

    def close(self):
        self.lease.close()

    def new_turn(self, request, language):
        numbers = [int(p.name[5:]) for p in self.path.glob("turn-*") if re.fullmatch(r"turn-\d{3,}", p.name)]
        turn = self.path / f"turn-{max(numbers, default=0) + 1:03d}"
        turn.mkdir()
        write_json(turn / "request.json", {"researcher_input": request, "response_language": language})
        self.state.language = language
        self.save()
        return turn

    def record(self, response, turn, *, planning=True):
        self.state.last_response = _public_response(response)
        if response.status not in {"conversation", "algorithms"}:
            plan = response.research_plan
            if response.prepared_task:
                plan = response.prepared_task.research_plan
            elif response.clarification:
                plan = response.clarification.research_plan
            if planning and plan:
                proposed = plan
                if response.prepared_task:
                    task = response.prepared_task.task_spec
                    # Pin the selected algorithm, but never promote an old dataset
                    # default into a researcher choice.
                    plan = plan.model_copy(update={
                        "algorithm_id": plan.algorithm_id or task.algorithm_id,
                    })
                old = self.state.current_plan
                self.state.revision += 1
                changes = {key: {"before": old.model_dump()[key] if old else None, "after": value}
                           for key, value in plan.model_dump().items()
                           if old is None or old.model_dump()[key] != value}
                write_json(turn / "plan_revision.json", {"revision": self.state.revision,
                    "previous_plan": old.model_dump() if old else None, "changes": changes,
                    "plan": plan.model_dump(), "proposed_plan": proposed.model_dump()})
                self.state.current_plan = plan
                self.state.active_turn = turn.name
                self.state.pending = _public_response(response) if response.clarification else None
        self.save()

    def clear_plan(self):
        self.state.current_plan = self.state.pending = self.state.active_turn = None
        self.save()

    @staticmethod
    def list_sessions(root):
        items = []
        for path in sorted(Path(root).glob("session-*"), reverse=True):
            if not path.is_dir() or path.resolve().parent != Path(root).resolve():
                continue
            try:
                state = StudySession.model_validate(read_json(path / "session.json"))
                items.append(f"{path.name}  {state.backend}  revision {state.revision}  {state.updated_at}")
            except (OSError, ValueError):
                items.append(f"{path.name}  unavailable metadata")
        return items

    def history(self):
        rows = []
        for path in sorted(self.path.glob("turn-*")):
            if not re.fullmatch(r"turn-\d{3,}", path.name):
                continue
            path = self.turn_path(self.path, path.name)
            status, task, report = "interrupted", "", None
            try:
                response = read_json(path / "response.json")
                status = response["status"]
                report = response.get("report_status")
                task = (response.get("execution") or {}).get("task_id", "")
            except (OSError, ValueError, KeyError, TypeError):
                pass
            try:
                task = RunStore(path / "backend").submission()["task_id"]
                status = self.attempt_state(path)
            except (OSError, ValueError, KeyError, TypeError):
                pass
            rows.append(f"{path.name}  {status}" + (f"  task {task}" if task else "")
                        + (f"  report {report}" if report in {"pending", "failed"} else ""))
        return rows

    @staticmethod
    def attempt_state(path):
        run = RunStore(Path(path) / "backend")
        if not run.exists():
            return "completed" if (Path(path) / "mock_result.json").exists() else "not_submitted"
        return run.attempt().value

    def unresolved_attempts(self):
        pending = []
        for path in self.path.glob("turn-*"):
            if not re.fullmatch(r"turn-\d{3,}", path.name):
                continue
            path = self.turn_path(self.path, path.name)
            if self.attempt_state(path) in {"unknown", "submitted"}:
                pending.append(path.name)
        return sorted(pending)
