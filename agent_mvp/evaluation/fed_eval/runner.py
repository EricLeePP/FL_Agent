"""Run test cases through a real Fed Agent study session and record every turn.

Each case is played into StudyChat, the same service the command line uses, so
clarification menus, follow-up edits and blocking rules behave as in production.
The scripted researcher never confirms a task, so nothing is ever executed.
"""

import json
import shutil
import tempfile
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

from federated_agent.conversation import response_language
from federated_agent.planner import LLMResearchPlanner
from federated_agent.services.study import StudyChat, StudySettings

from .world import build_world_agent, world_clock


def menu_reply(select, questions):
    """The reply the terminal menu sends for a scripted selection, built from the questions actually shown.

    The menu answers every pending question and copies its requested value, so a planner that renamed an
    unknown hospital ("7号医院" as "hospital 7") still gets a working selection. A question without the
    chosen values among its candidates can only be answered by typing, which goes to the planner. When the
    questions do not match the script, the conversation has already left it and the script is sent as is.
    """
    remaining, answers = list(select), []
    for question in questions:
        match = (next((a for a in remaining if a["field"] == question["field"]
                       and a.get("requested_value") == question.get("requested_value")), None)
                 or next((a for a in remaining if a["field"] == question["field"]), None))
        if match is None:
            break
        remaining.remove(match)
        values = match["selected_values"]
        answer = {"field": question["field"], "requested_value": question.get("requested_value")}
        if all(value in (question.get("candidates") or []) for value in values):
            answers.append({**answer, "selected_values": values})
        else:
            answers.append({**answer, "reply": ", ".join(values)})
    else:
        if questions and not remaining:
            return json.dumps({"confirmed_choices": answers}, ensure_ascii=False)
    return json.dumps({"confirmed_choices": select}, ensure_ascii=False)


def scripted_input(turn, questions):
    """What the researcher sends for one scripted turn, given the questions pending before it."""
    if "user" in turn:
        return turn["user"]
    if "select" in turn:
        return menu_reply(turn["select"], questions)
    raise ValueError("a turn needs 'user' or 'select'")


def pending_questions(outcome):
    return outcome["questions"] if outcome and outcome["status"] == "needs_clarification" else []


def case_inputs(case):
    """The inputs of a case when every menu shows exactly the scripted questions."""
    inputs = []
    for turn in case["turns"]:
        questions = [{"field": a["field"], "requested_value": a.get("requested_value"), "candidates": a["selected_values"]}
                     for a in turn.get("select", [])]
        inputs.append(scripted_input(turn, questions))
    return inputs


def summarize(response):
    plan = response.research_plan
    if response.prepared_task:
        plan = response.prepared_task.research_plan
    elif response.clarification:
        plan = response.clarification.research_plan
    prepared = None
    if response.prepared_task:
        task = response.prepared_task
        prepared = {"algorithm_id": task.task_spec.algorithm_id,
                    "variables": [variable.class_id for variable in task.resolved_plan.variables],
                    "stratification": [variable.class_id for variable in task.resolved_plan.stratification_variables],
                    "sites": [site.organisation_id for site in task.resolved_plan.sites],
                    "database": task.task_spec.databases[0].label,
                    "data_checks": sorted({check.state for check in task.data_checks})}
    questions = [{"field": q.field, "requested_value": q.requested_value, "candidates": q.candidates}
                 for q in response.clarification.questions] if response.clarification else []
    return {"status": response.status, "submission_state": response.submission_state,
            "issues": [{"code": item.code, "severity": item.severity, "field": item.field,
                        "requested_value": item.requested_value} for item in response.issues],
            "questions": questions, "plan": plan.model_dump() if plan else None, "prepared": prepared}


class ScriptedUI:
    """The researcher side of a study session, driven by a test case."""

    def __init__(self, script, provider):
        self.script = list(script)
        self.provider = provider
        self.language = "en"
        self.last_response = None
        self.turns = []
        self.started = None

    def _close_turn(self):
        """Attribute the model calls and time since the turn began; each turn closes once."""
        if self.turns and "model_calls" not in self.turns[-1]:
            turn = self.turns[-1]
            turn["model_calls"] = self.provider.drain()
            turn["seconds"] = round(time.monotonic() - self.started, 3)

    def header(self, *args):
        pass

    def say(self, value="", style=None):
        if self.turns:
            self.turns[-1]["messages"].append(str(value))

    def ask(self, pending=None):
        self._close_turn()
        if len(self.turns) >= len(self.script):
            return "/exit"
        shown = pending_questions(summarize(pending)) if pending is not None else []
        text = scripted_input(self.script[len(self.turns)], shown)
        if not text.startswith("{"):
            self.language = response_language(text, self.language)
        self.turns.append({"input": text, "pending_before": bool(pending), "responses": [], "messages": []})
        self.started = time.monotonic()
        return text

    def menu(self):
        return "/exit"

    def select_session(self, entries):
        return None

    @contextmanager
    def planning(self, planner):
        yield

    def present(self, response, planner, output, *, narrate=True):
        self.last_response = response
        if self.turns:
            self.turns[-1]["responses"].append(summarize(response))

    def review(self, prepared, output):
        return False

    def progress(self, message):
        pass


# Failures of the serving infrastructure, not of the model: the case is run again on resume.
INFRASTRUCTURE_ERRORS = {"URLError", "ConnectionResetError", "ConnectionRefusedError", "RemoteDisconnected", "TimeoutError",
                         "ConnectionError", "http_500", "http_502", "http_503", "http_504"}


def run_case(agent, provider, case):
    provider.drain()
    ui = ScriptedUI(case["turns"], provider)
    workdir = Path(tempfile.mkdtemp(prefix="fed-eval-"))
    error = None
    started = time.monotonic()
    try:
        with world_clock():
            StudyChat(agent, LLMResearchPlanner(provider), ui, StudySettings(output=workdir)).run()
    except Exception:
        error = traceback.format_exc(limit=6)
    finally:
        ui._close_turn()
        shutil.rmtree(workdir, ignore_errors=True)
    infrastructure = any(call.get("error") in INFRASTRUCTURE_ERRORS for turn in ui.turns for call in turn.get("model_calls", []))
    return {"turns": ui.turns, "harness_error": error, "infrastructure_error": infrastructure,
            "seconds": round(time.monotonic() - started, 3)}


def current(record, case):
    """True when the record played the case as it is now. A changed case makes its old results stale, and so
    does a selection that the menu, given the questions the record shows, would have sent differently."""
    questions = []
    for spec, turn in zip(case["turns"], record["turns"]):
        if turn["input"] != scripted_input(spec, questions):
            return False
        questions = pending_questions(turn["responses"][-1] if turn.get("responses") else None)
    return True


def load_records(path, cases=None):
    """The latest attempt of each case and repeat. Attempts lost to the infrastructure are left out, and so
    are attempts at an earlier version of a case when the current cases (a list or a dict by id) are given."""
    if isinstance(cases, list):
        cases = {case["id"]: case for case in cases}
    latest = {}
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if record.get("infrastructure_error"):
                    continue
                if cases is not None and (record["case_id"] not in cases or not current(record, cases[record["case_id"]])):
                    continue
                latest[(record["case_id"], record["repeat"])] = record
    return list(latest.values())


def completed(path, cases=None):
    return {(record["case_id"], record["repeat"]) for record in load_records(path, cases)}


def warm_up(provider):
    """Load the model before timing starts; a cold load would otherwise count against the first case."""
    try:
        provider.generate_json("Return a JSON research plan.", "Describe age at diagnosis in hospital 1.")
    except Exception:
        pass
    provider.drain()


def run(cases, provider, out_path, *, repeats=1, seeds=None, progress=print):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = completed(out_path, cases)
    agent = build_world_agent()
    warm_up(provider)
    total = len(cases) * repeats
    with out_path.open("a", encoding="utf-8") as out:
        for repeat in range(repeats):
            provider.seed = seeds[repeat] if seeds else None
            for index, case in enumerate(cases):
                if (case["id"], repeat) in done:
                    continue
                record = run_case(agent, provider, case)
                record.update(case_id=case["id"], repeat=repeat, seed=provider.seed)
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                final = next((r["status"] for turn in reversed(record["turns"]) for r in reversed(turn["responses"])), None)
                progress(f"[{repeat * len(cases) + index + 1}/{total}] {case['id']}: {final} ({record['seconds']:.1f}s)"
                         + (" HARNESS ERROR" if record["harness_error"] else ""))
