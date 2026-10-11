"""A researcher's conversation over one saved study.

The rules that protect a study live here: which commands may run, when a new
submission is blocked, that consent is saved before anything is submitted, and
that recovery and report regeneration never submit. ``ui`` only talks to the
researcher; terminal_ui.TerminalUI provides the methods used here (header, say,
ask, menu, select_session, planning, present, review, progress, language and
last_response).
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..clarification import SelectionPlanner, apply_choices
from ..capabilities import (
    DataCapabilities,
    ExecutorCapabilities,
    PlannerCapabilities,
    PolicyCapabilities,
    capabilities_of,
)
from ..conversation import LOCAL_INFO_COMMANDS, conversation_response
from ..errors import PreflightError
from ..feedback import render_researcher_response
from ..models import PreparedTask
from ..run_manifest import finish_manifest, write_manifest
from ..runs.store import RunStore
from ..sessions import SessionError, SessionStore
from ..storage import read_json
from . import execution
from .artifacts import write_planning_response, write_prepared, write_response
from .reporting import saved_result
from .responses import error_response

HELP = ("/model       Configured LLM and agent runtime\n/date        Current local date\n/time        Current local time\n"
        "/algorithms  Current catalog\n/plan        Active study plan\n/history     Research turns and task IDs\n"
        "/sessions    Saved sessions\n/resume ID   Restore a session (no execution)\n"
        "/review      Revalidate the active plan for a new task\n/recover ID  Retrieve an existing task by turn ID\n"
        "/report ID   Rebuild a saved result report (no submission)\n/details     Last checked response\n"
        "/new         Clear the active study plan\n/cancel      Clear plan; does NOT cancel remote jobs\n"
        "/exit        Save and close")
RESEARCH_COMMANDS = {"/review", "/algorithms", *LOCAL_INFO_COMMANDS}


@dataclass(frozen=True)
class StudySettings:
    output: Path
    backend: str = "mock"
    provider: str | None = None
    confirmed_sandbox: bool = False
    timeout_seconds: int = 600
    resume: str | None = None

    @classmethod
    def from_args(cls, args):
        return cls(output=Path(args.output), backend=getattr(args, "backend", None) or "mock",
                   provider=getattr(args, "provider", None),
                   confirmed_sandbox=bool(getattr(args, "confirm_synthetic_unencrypted_sandbox", False)),
                   timeout_seconds=getattr(args, "timeout_seconds", None) or 600,
                   resume=getattr(args, "resume", None))


class SavedPlanPlanner:
    """Replays the saved study plan for /review; never calls a model."""

    def __init__(self, plan):
        self.plan = plan

    def create_plan(self, request, catalog):
        return self.plan.model_copy(deep=True)


def configure_executor(agent, settings, output, ui):
    execution.configure_executor(agent, settings.backend, output, confirmed=settings.confirmed_sandbox,
                                 timeout_seconds=settings.timeout_seconds, progress=ui.progress)


def save_execution_response(agent, planner, settings, store, prepared, response, output, ui):
    """Record an execution, recovery or regenerated report in the turn, the study and the manifest."""
    def persist(item):
        write_response(item, output, display=False)
        store.record(item, output, planning=False)
        finish_manifest(output, item)

    final = execution.finish_execution(agent, prepared, response, output, settings.backend, persist=persist, emit=ui.say)
    ui.present(final, planner, output)
    return final


def recover_turn(agent, planner, settings, store, name, ui):
    """Read back the existing remote task of a recorded turn. This path never submits."""
    if settings.backend != "vantage6":
        raise SessionError("Remote recovery is only available with --backend vantage6.")
    if not name:
        raise SessionError("Choose a recorded turn using /history, then /recover <turn-id>.")
    output = store.turn_path(store.path, name)
    try:
        prepared = PreparedTask.model_validate_json((output / "prepared_task.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise SessionError("This turn has no valid prepared task to recover. Nothing was submitted.") from None
    if not RunStore(output / "backend").exists():
        raise SessionError("No real execution attempt recorded for this turn. Use /review for a new submission.")
    ui.say(f"Recovering {name}: read existing task only; no new submission.", "cyan")
    configure_executor(agent, settings, output, ui)
    response = agent.recover_response(prepared)
    return save_execution_response(agent, planner, settings, store, prepared, response, output, ui)


def report_turn(agent, planner, settings, store, name, ui):
    """Rebuild the report of a completed turn from its saved result. No model call, no submission."""
    output, prepared, response = saved_result(agent, store, name, settings.backend)
    ui.say(f"Rebuilding {name} from its saved result; no model call or Vantage6 submission.", "cyan")
    response = response.model_copy(update={"status": "completed"})
    return save_execution_response(agent, planner, settings, store, prepared, response, output, ui)


class StudyChat:
    def __init__(self, agent, planner, ui, settings: StudySettings):
        self.agent = agent
        self.planner = planner
        self.ui = ui
        self.settings = settings
        self.planner_capabilities = capabilities_of(planner, PlannerCapabilities)
        self.store = None

    def run(self):
        settings = self.settings
        self.store = (SessionStore.open(settings.output, settings.resume, settings.backend) if settings.resume
                      else SessionStore.create(settings.output, settings.backend))
        try:
            policy = capabilities_of(self.agent.policy_checker, PolicyCapabilities).label
            self.ui.header(settings, self.planner, self.store.path, policy)
            self.restore(resumed=bool(settings.resume))
            self.show_inventory_health()
            for _ in range(1000):
                if not self.turn():
                    break
        finally:
            try:
                if capabilities_of(self.agent.executor, ExecutorCapabilities).closeable:
                    self.agent.executor.close()
            finally:
                self.store.close()

    def restore(self, *, resumed):
        store, ui = self.store, self.ui
        ui.language = store.state.language
        ui.last_response = store.state.last_response
        if resumed:
            ui.say(f"Resumed {store.path.name}, revision {store.state.revision}. No task submitted.", "cyan")
            ui.say("Use /plan and /history. /recover reads an existing task; /review prepares a NEW task.", "dim")
            if store.state.pending and store.state.current_plan and not store.state.current_plan.database_label:
                ui.say("Dataset selection is pending. Choose a dataset before preparing any new task.", "yellow")

    def show_inventory_health(self):
        checker = self.agent.data_checker
        if not capabilities_of(checker, DataCapabilities).inventory_file:
            return
        try:
            document = checker.load()
            if document.expires_at <= datetime.now(timezone.utc):
                self.ui.say(f"Data inventory EXPIRED at {document.expires_at.isoformat()}. New submissions are blocked.", "yellow")
                self.ui.say("An operator must reverify hospital metadata and publish a fresh snapshot. Do not just extend "
                            "its expiry. Existing task recovery remains available.", "yellow")
        except PreflightError:
            self.ui.say("Data inventory unavailable or invalid. New submissions require verified metadata.", "yellow")

    def turn(self) -> bool:
        """Handle one researcher input; False ends the conversation."""
        output = None
        try:
            pending = self.store.state.pending
            request = self.ui.ask(None if self.planner_capabilities.complete_requests_only else pending)
            if request == "/menu":
                request = self.ui.menu()
            if request == "/exit":
                self.ui.say("Session saved. No remote task was cancelled.", "dim")
                return False
            if self.command(request):
                return True
            output = self.store.new_turn(request, self.ui.language)
            self.research(request, pending, output)
        except (EOFError, KeyboardInterrupt):
            self.ui.say("\nSaved locally. Remote tasks are NOT cancelled. Use /resume and /recover; do not resubmit.", "yellow")
            return False
        except SessionError as exc:
            self.ui.say(str(exc), "yellow")
        except PreflightError:
            response = error_response("preflight_error", "preflight_error")
            if output:
                write_response(response, output, display=False)
                self.store.record(response, output, planning=False)
                self.ui.present(response, self.planner, output)
            else:
                self.ui.say(response.message, "yellow")
        return True

    def command(self, request) -> bool:
        """Run a local command. False means the input starts a research turn."""
        store, ui, settings = self.store, self.ui, self.settings
        if request in {"/new", "/cancel"}:
            store.clear_plan()
            ui.say("Active plan cleared; history retained. Remote tasks are NOT cancelled. Enter a complete request.", "dim")
        elif request == "/sessions":
            ui.say("\n".join(SessionStore.list_sessions(settings.output)) or "No saved sessions.")
        elif request == "/resume" or request.startswith("/resume "):
            name = request.partition(" ")[2].strip() or ui.select_session(SessionStore.list_sessions(settings.output))
            if name and name != store.path.name:
                other = SessionStore.open(settings.output, name, settings.backend)
                store.close()
                self.store = other
                self.restore(resumed=True)
        elif request == "/history":
            ui.say("\n".join(store.history()) or "No research turns yet.")
        elif request == "/plan":
            ui.say(store.state.current_plan.model_dump_json(indent=2) if store.state.current_plan else "No active study plan.")
        elif request == "/recover" or request.startswith("/recover "):
            name = request.partition(" ")[2].strip() or store.state.active_turn
            recover_turn(self.agent, self.planner, settings, store, name, ui)
        elif request == "/report" or request.startswith("/report "):
            name = request.partition(" ")[2].strip() or store.state.active_turn
            report_turn(self.agent, self.planner, settings, store, name, ui)
        elif request == "/help":
            ui.say(HELP)
        elif request == "/details":
            ui.say(render_researcher_response(ui.last_response) if ui.last_response else "No response yet.")
        elif not request:
            pass
        elif request.startswith("/") and request not in RESEARCH_COMMANDS:
            ui.say("Unknown command. Use /help.", "yellow")
        elif request == "/review" and not store.state.current_plan:
            ui.say("No active study plan. Enter a research request.")
        else:
            if request == "/review" and store.state.active_turn:
                active = store.turn_path(store.path, store.state.active_turn)
                if store.attempt_state(active) != "not_submitted":
                    raise SessionError("This turn already has an execution attempt. Use /recover for its task; "
                                       "/new starts an explicitly separate study.")
            return False
        return True

    def plan(self, request, pending):
        """Answer a research input with a checked response; nothing is submitted here."""
        agent, planner, store, ui = self.agent, self.planner, self.store, self.ui
        conversation = conversation_response(request, planner, ui.language)
        if conversation is not None:
            return conversation
        if request == "/algorithms":
            return agent.describe_algorithms()
        if request == "/review":
            return agent.respond("Review the saved study plan", SavedPlanPlanner(store.state.current_plan))
        previous = None
        if self.planner_capabilities.revises_plans:
            planner.response_language = ui.language
            previous = store.state.current_plan
            # A menu selection is one of the validators' own candidates: apply it without a model.
            selected = apply_choices(previous, request) if pending and previous else None
            if selected is not None:
                planner = SelectionPlanner(selected)
        questions = [q.model_dump() for q in pending.clarification.questions] if pending else []
        if store.state.current_plan and self.planner_capabilities.complete_requests_only:
            ui.say("Offline demo requires a complete request; contextual edits require an LLM.", "dim")
        with ui.planning(planner):
            return agent.respond(request, planner, previous_plan=previous, questions=questions)

    def research(self, request, pending, output):
        agent, planner, store, ui = self.agent, self.planner, self.store, self.ui
        response = self.plan(request, pending)
        write_response(response, output, display=False)
        write_planning_response(response, output)
        store.record(response, output)
        ui.present(response, planner, output, narrate=request != "/algorithms")
        if response.status != "ready":
            return
        prepared = response.prepared_task
        write_prepared(prepared, output)
        write_manifest(output, prepared, planner, store.state)
        self.show_plan_changes(output)
        unresolved = store.unresolved_attempts()
        if unresolved:
            ui.say("New submission blocked: unresolved earlier attempts " + ", ".join(unresolved) +
                   ". Use /recover <turn-id> or ask the administrator to reconcile server history. "
                   "The revised plan is saved.", "yellow")
            return
        if not ui.review(prepared, output):
            ui.say("Not submitted. Plan saved; revise it or use /review later.", "dim")
            return
        execution.record_consent(output, prepared, store.state.revision)
        configure_executor(agent, self.settings, output, ui)
        ui.progress("Rechecking the task and current catalog...")
        outcome = agent.execute_response(prepared, prepared.confirmation_id)
        save_execution_response(agent, planner, self.settings, store, prepared, outcome, output, ui)

    def show_plan_changes(self, output):
        revision = read_json(Path(output) / "plan_revision.json")
        if revision["previous_plan"] is not None:
            self.ui.say(f"Study revision {revision['revision']} changes:", "bold")
            for key, change in revision["changes"].items():
                self.ui.say(f"  {key}: {json.dumps(change['before'], ensure_ascii=False)} -> "
                            f"{json.dumps(change['after'], ensure_ascii=False)}")
