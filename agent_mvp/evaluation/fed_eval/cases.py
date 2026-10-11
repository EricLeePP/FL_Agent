"""Load test cases, check their structure and validate their labels with the oracle."""

import hashlib
import json
from pathlib import Path

from .oracle import OracleProvider, gold_outputs
from .runner import run_case
from .world import build_world_agent

TURN_KINDS = {"user", "select"}


def load(path):
    cases = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                cases.append(json.loads(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON: {exc}") from None
    return cases


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def structure_problems(cases):
    problems, seen = [], set()
    for case in cases:
        case_id = case.get("id")
        if not case_id or case_id in seen:
            problems.append(f"{case_id}: missing or duplicate id")
        seen.add(case_id)
        turns = case.get("turns")
        if not isinstance(turns, list) or not turns:
            problems.append(f"{case_id}: needs at least one turn")
            continue
        for index, turn in enumerate(turns):
            kinds = TURN_KINDS & set(turn)
            if len(kinds) != 1:
                problems.append(f"{case_id} turn {index + 1}: exactly one of {sorted(TURN_KINDS)} is required")
            if "select" in turn and "gold" in turn and index != len(turns) - 1:
                # A selection's gold is used only when the menu cannot offer the choice and the researcher types it;
                # on a final turn an unused gold output is harmless, anywhere else it would shift the later ones.
                problems.append(f"{case_id} turn {index + 1}: only a final menu selection may carry a gold output")
        if "expect" not in turns[-1]:
            problems.append(f"{case_id}: the final turn needs an expectation")
    return problems


def oracle_run(case, agent=None):
    agent = agent or build_world_agent()
    return run_case(agent, OracleProvider(gold_outputs(case)), case)
