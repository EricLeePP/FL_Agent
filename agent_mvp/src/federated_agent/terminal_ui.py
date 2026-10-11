"""Inline terminal conversation, with trusted task controls and optional SSE prose."""

import json
import os
import sys
from contextlib import contextmanager
from http.client import HTTPException
from pathlib import Path
from time import monotonic
from urllib.error import URLError

from prompt_toolkit import Application, PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import CheckboxList, Label, RadioList
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from .errors import LLMResponseError
from .conversation import is_local_conversation, response_language
from .capabilities import PlannerCapabilities, capabilities_of
from .narration import explain


COMMANDS = ["/algorithms", "/details", "/menu", "/new", "/cancel", "/exit", "/help",
            "/plan", "/history", "/sessions", "/resume", "/recover", "/report", "/review", "/model", "/date", "/time"]
MENU_STYLE = Style.from_dict({
    "radio-selected": "reverse", "checkbox-selected": "reverse",
    "radio-checked": "bold", "checkbox-checked": "bold", "title": "bold",
    "prompt": "bold ansicyan", "completion-menu.completion.current": "reverse",
})
STATUS = {
    "ready": ("Task ready for review", "green"),
    "needs_clarification": ("More information needed", "yellow"),
    "unsupported": ("Analysis not supported", "yellow"),
    "algorithms": ("Registered algorithms", "cyan"),
    "completed": ("Execution completed", "green"),
    "report_failed": ("Computation completed; report failed", "yellow"),
}
def terminal_text(value):
    # Catalog/model text is untrusted terminal content, never ANSI or Rich markup.
    return "".join(char for char in str(value) if char in "\n\t" or
                   (ord(char) >= 32 and not 127 <= ord(char) <= 159)).replace("\t", "    ")


def choice_application(title, options, *, multiple=False, input=None, output=None):
    values = [(value, terminal_text(label)) for value, label in options]
    widget = CheckboxList(values=values) if multiple else RadioList(values, select_on_focus=True)
    keys = KeyBindings()

    @keys.add("enter", eager=True)
    def accept(event):
        event.app.exit(result=list(widget.current_values) if multiple else widget.current_value)

    @keys.add("escape", eager=True)
    def cancel(event):
        event.app.exit(result=None)

    @keys.add("c-c", eager=True)
    def interrupt(event):
        event.app.exit(exception=KeyboardInterrupt())

    @keys.add("c-d", eager=True)
    def eof(event):
        event.app.exit(exception=EOFError())

    return Application(layout=Layout(HSplit([Label(terminal_text(title), style="class:title"), widget]),
                                     focused_element=widget),
        key_bindings=keys, style=MENU_STYLE if "NO_COLOR" not in os.environ else Style.from_dict({}),
        full_screen=False, erase_when_done=True, mouse_support=False, input=input, output=output)


class TerminalUI:
    def __init__(self, *, read=None, plain=False, stream=True, console=None):
        self.read = read or input
        self.interactive = (not plain and read is None and sys.stdin.isatty() and sys.stdout.isatty()
                            and os.environ.get("TERM") != "dumb")
        self.console = console or Console(highlight=False, markup=False,
                                          force_terminal=None if self.interactive else False)
        self.stream = stream
        self.prompt_session = (PromptSession(history=InMemoryHistory(),
            completer=WordCompleter(COMMANDS), complete_while_typing=False, style=MENU_STYLE)
            if self.interactive else None)
        self.last_response = None
        self.language = "en"

    def say(self, value="", style=None):
        self.console.print(Text(terminal_text(value), style=style))

    @staticmethod
    def display_path(path):
        try:
            return str(Path(path).relative_to(Path.cwd()))
        except ValueError:
            return str(path)

    def header(self, args, planner, session, policy):
        self.console.rule("FED AGENT", align="left", style="bright_black")
        model = getattr(getattr(planner, "provider", None), "model", args.provider)
        language_model = capabilities_of(planner, PlannerCapabilities).language_model
        runtime = getattr(getattr(planner, "provider", None), "runtime_name", "json" if language_model else "offline")
        if not isinstance(runtime, str):
            runtime = "custom"
        self.say(f"{args.provider}  /  {model}  |  {args.backend}  |  {runtime}", "bold")
        self.say(f"Policy: {policy}  |  {'SSE streaming' if self.stream else 'local feedback'}", "dim")
        self.say("Synthetic/public research only. Credentials stay in local password prompts.", "dim")
        self.say(f"Session: {session.name}", "dim")
        self.say(f"Output: {self.display_path(session.parent)}", "dim")
        self.say("/menu   /algorithms   /plan   /sessions   /history   /exit", "dim")

    def prompt(self, label, *, track_language=False):
        if self.prompt_session:
            reply = self.prompt_session.prompt([("class:prompt", label)]).strip()
        else:
            reply = self.read(label).strip()
        if track_language:
            self.language = response_language(reply, self.language)
        return reply

    def ask(self, pending=None):
        self.say()
        if not pending or not self.interactive:
            return self.prompt("Clarification> " if pending else "> ", track_language=True)
        answers = []
        for question in pending.clarification.questions:
            if not question.candidates:
                reply = self.prompt(f"{question.field}> ", track_language=True)
                if reply.startswith("/") or is_local_conversation(reply):
                    return reply
                answers.append({"field": question.field, "requested_value": question.requested_value,
                                "reply": reply})
                continue
            candidates = question.candidates
            options = [(str(index), value) for index, value in enumerate(candidates)]
            options += [("write", "Type a reply instead"), ("cancel", "Cancel this request")]
            multi = question.field in {"sites", "variables", "stratification_variables"} and not question.requested_value
            selected = choice_application(question.question, options, multiple=multi).run()
            values = selected if isinstance(selected, list) else [selected]
            if selected is None or "cancel" in values:
                return "/cancel"
            if not selected or "write" in values:
                reply = self.prompt(f"{question.field}> ", track_language=True)
                if reply.startswith("/") or is_local_conversation(reply):
                    return reply
                answers.append({"field": question.field, "requested_value": question.requested_value,
                                "reply": reply})
                continue
            chosen = [candidates[int(value)] for value in values]
            answers.append({"field": question.field, "requested_value": question.requested_value,
                            "selected_values": chosen})
        reply = json.dumps({"confirmed_choices": answers}, ensure_ascii=False)
        self.say("Selected: " + "; ".join(f"{a['field']} = {', '.join(a.get('selected_values', [a.get('reply', '')]))}" for a in answers), "dim")
        return reply

    def menu(self):
        options = [("/algorithms", "Available algorithms"), ("/details", "Last response details"),
                   ("/plan", "Current study plan"), ("/history", "Research history and task IDs"),
                   ("/resume", "Resume a saved session"), ("/recover", "Recover the active turn's remote task"),
                   ("/report", "Rebuild the active turn's saved result report"),
                   ("/review", "Revalidate and review a new task"),
                   ("/new", "New research request"), ("/exit", "Exit")]
        if self.interactive:
            return choice_application("FED AGENT", options).run() or ""
        for command, title in options:
            self.say(f"  {command}  {title}")
        return ""

    def select_session(self, entries):
        if not entries:
            self.say("No saved sessions.")
            return None
        if self.interactive:
            return choice_application("Resume a saved session (no execution)",
                                      [(entry.split()[0], entry) for entry in entries]).run()
        self.say("\n".join(entries))
        return self.prompt("Session ID (Enter to cancel): ") or None

    @contextmanager
    def planning(self, planner):
        self.say("  > Reading intent and checking the live catalog...", "dim")
        count = 0
        old = getattr(planner, "on_delta", None)
        status = self.console.status("Reading intent...", spinner="line") if self.interactive else None

        def delta(text):
            nonlocal count
            count += len(text)
            if status:
                status.update(f"Reading intent... {count} characters received")

        reports_progress = capabilities_of(planner, PlannerCapabilities).reports_progress
        if reports_progress and self.stream:
            planner.on_delta = delta
        try:
            if status:
                status.start()
            yield
        finally:
            if reports_progress:
                planner.on_delta = old
            if status:
                status.stop()

    def present(self, response, planner, output, *, narrate=True):
        self.last_response = response
        if response.status == "conversation":
            self.say("\nAgent", "bold cyan")
            self.say(response.message)
            return
        title, color = STATUS.get(response.status, (response.status.replace("_", " ").capitalize(), "red"))
        self.say(f"\n  {title}  |  {response.submission_state}", f"bold {color}")
        # Always show trusted facts independently of optional model prose.
        self.say(response.message)
        if response.status == "ready":
            warnings = {item.code for item in response.issues if item.severity == "warning"}
            checks = []
            if "data_snapshot_only" in warnings:
                checks.append("data: metadata snapshot only")
            if "runtime_unverified" in warnings:
                checks.append("live execution: not yet verified")
            if checks:
                self.say("  " + " | ".join(checks), "dim")
        for problem in response.issues:
            if problem.severity == "warning":
                if problem.code == "dataset_retained":
                    self.say((f"  \u6cbf\u7528\u5f53\u524d\u7814\u7a76\u8ba1\u5212\u7684\u6570\u636e\u96c6\uff1a{problem.requested_value}" if self.language == "zh"
                              else f"  Dataset retained from active study: {problem.requested_value}"), "dim")
                if problem.code in {"data_metadata_expired", "data_unverified", "data_not_listed"}:
                    self.say(f"  {problem.code}: {problem.message}", "yellow")
                continue
            detail = problem.requested_value or problem.field or ""
            self.say(f"  {problem.code}: {detail}" if detail else f"  {problem.code}", "yellow")
            for step in problem.next_steps:
                self.say(f"  {step}")
        if response.clarification:
            for question in response.clarification.questions:
                self.say(f"  {question.question}")
                if question.candidates and not self.interactive:
                    self.say("  Options: " + ", ".join(question.candidates))
        if response.suggested_algorithms and response.status != "ready":
            table = Table(box=None, padding=(0, 1), show_header=True, expand=False)
            table.add_column("Algorithm", style="cyan", overflow="fold")
            table.add_column("Version")
            table.add_column("Capabilities", overflow="fold")
            for algorithm in response.suggested_algorithms:
                table.add_row(Text(terminal_text(algorithm.display_name)), Text(terminal_text(algorithm.version)),
                              Text(terminal_text("; ".join(algorithm.capabilities))))
            self.console.print(table)
        if response.catalog:
            self.say(f"  Catalog {response.catalog.catalog_version} / {response.catalog.sha256[:12]}", "dim")
        provider = planner.provider if capabilities_of(planner, PlannerCapabilities).language_model else None
        deltas = explain(provider, response, self.language) if narrate and self.stream else None
        if deltas is not None:
            self._stream_reply(deltas, output)

    def _stream_reply(self, deltas, output):
        self.say("\nAgent", "bold cyan")
        fragments = []
        chunks = 0
        started = monotonic()
        first_delta = None
        complete = False
        interrupted = False
        live = Live(Text(""), console=self.console, auto_refresh=False, vertical_overflow="visible") if self.interactive else None
        try:
            if live:
                live.start()
            for delta in deltas:
                first_delta = first_delta if first_delta is not None else monotonic() - started
                safe = terminal_text(delta)
                fragments.append(safe)
                chunks += 1
                if live:
                    live.update(Text("".join(fragments)), refresh=True)
                else:
                    self.console.print(Text(safe), end="", soft_wrap=True)
                    self.console.file.flush()
            complete = bool("".join(fragments).strip())
            if not complete:
                raise LLMResponseError("Empty explanation")
        except (LLMResponseError, URLError, TimeoutError, OSError, HTTPException):
            interrupted = True
        finally:
            if live:
                live.stop()
            else:
                self.say()
            (output / "assistant_reply.txt").write_text("".join(fragments), encoding="utf-8")
            (output / "stream.json").write_text(json.dumps({"complete": complete,
                "chunks": chunks, "first_delta_seconds": first_delta,
                "elapsed_seconds": monotonic() - started, "language": self.language,
                "authoritative": False}, indent=2), encoding="utf-8")
        if interrupted:
            self.say("Explanation interrupted. Use the checked response above; no execution was triggered.", "yellow")

    def review(self, prepared, output):
        task = prepared.task_spec
        self.console.rule("TASK REVIEW", align="left", style="bright_black")
        fields = [("Algorithm", f"{task.algorithm_id}  {task.algorithm_version}"),
                  ("Hospitals", ", ".join(f"{site.canonical_name} (org {site.organisation_id})" for site in prepared.resolved_plan.sites)),
                  ("Variables", ", ".join(variable.canonical_name for variable in prepared.resolved_plan.variables)),
                  ("Database", task.databases[0].label), ("Study context", prepared.research_plan.purpose),
                  ("Server", f"{task.connection.server_url}:{task.connection.server_port}{task.connection.server_api}"),
                  ("Review ID", prepared.confirmation_id)]
        request_path = output / "request.json"
        if request_path.is_file():
            request = json.loads(request_path.read_text(encoding="utf-8")).get("researcher_input")
            if isinstance(request, str) and not request.startswith("/"):
                fields.insert(0, ("Researcher request", request))
        for name, value in fields:
            self.say(f"{name}: {value}")
        self.say("Policy: " + ("allow-all (development)" if "mvp_allow_all_policy" in prepared.policy_decision.reasons else
                               str(prepared.policy_decision.allowed)), "dim")
        self.say("Script: " + self.display_path((output / "run_vantage6_task.py").resolve()), "dim")
        if self.interactive:
            return choice_application("Run this exact task?", [("cancel", "Keep files without running"),
                                      ("confirm", "Submit and wait for results")]).run() == "confirm"
        return self.prompt("Type 'confirm' to execute this task, or Enter to cancel: ") == "confirm"

    def progress(self, message):
        self.say(f"  > {message}", "cyan")
