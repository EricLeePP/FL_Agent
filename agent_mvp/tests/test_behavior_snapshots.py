"""Behaviour snapshots that pin user-visible outcomes of a fixed request corpus.

The snapshot is a curated projection: statuses, issue codes, resolved task
content and the artifact layout of chat turns. Volatile values (absolute paths,
hashes, confirmation IDs, timestamps and mock task IDs) are left out, so an
intentional change to fingerprints or file locations does not hide behaviour
regressions.

Regenerate only for an intentional behaviour change:
    UPDATE_BEHAVIOR_SNAPSHOTS=1 python -m unittest test_behavior_snapshots
"""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from federated_agent.cli import run_chat
from federated_agent.planner import DemoResearchPlanner, LLMResearchPlanner
from federated_agent.resources import RESOURCE_ROOT, SOURCE_CHECKOUT
from federated_agent.app import build_agent

SNAPSHOT = Path(__file__).parent / "fixtures" / "behavior_snapshots.json"
CORPUS = SOURCE_CHECKOUT / "examples" / "cli_requests.json"
UBUNTU = RESOURCE_ROOT / "config" / "ubuntu_sandbox.json"
INVENTORY = SOURCE_CHECKOUT / "tests" / "fixtures" / "ubuntu_data_availability_20260907.json"
VALID_CLOCK = datetime(2026, 9, 7, tzinfo=timezone.utc)


def issues(response):
    return [[item.code, item.severity, item.field, item.requested_value] for item in response.issues]


def task_projection(prepared):
    task = prepared.task_spec
    return {
        "algorithm": [task.algorithm_id, task.algorithm_version, task.image],
        "input": task.input,
        "databases": [database.label for database in task.databases],
        "collaboration": [task.collaboration_id, task.central_organisation_id],
        "connection": [task.connection.environment, task.connection.version, task.connection.server_url,
                       task.connection.server_port, task.connection.server_api],
        "name": task.name,
        "description": task.description,
        "warnings": task.warnings,
        "result_adapter": task.result_adapter,
        "requires_managed_execution": task.requires_managed_execution,
        "resolved_variables": [[v.requested_name, v.canonical_name, v.class_id, v.datatype]
                               for v in prepared.resolved_plan.variables],
        "resolved_sites": [[s.requested_name, s.canonical_name, s.organisation_id, s.database_label]
                           for s in prepared.resolved_plan.sites],
        "data_checks": [[c.site, c.variable, c.state, c.evidence, c.reason] for c in prepared.data_checks],
        "policy": [prepared.policy_decision.allowed, prepared.policy_decision.verification,
                   prepared.policy_decision.reasons],
    }


def response_projection(response):
    result = {"status": response.status, "submission_state": response.submission_state,
              "report_status": response.report_status, "issues": issues(response)}
    if response.research_plan is not None:
        result["plan"] = response.research_plan.model_dump(mode="json")
    if response.clarification is not None:
        result["questions"] = [[q.field, q.requested_value, q.candidates] for q in response.clarification.questions]
    if response.prepared_task is not None:
        result["task"] = task_projection(response.prepared_task)
    if response.execution is not None:
        result["execution"] = {"status": response.execution.status.value, "error": response.execution.error,
                               "result": response.execution.result}
    if response.suggested_algorithms:
        result["suggested"] = [[s.algorithm_id, s.scope] for s in response.suggested_algorithms]
    return result


def corpus_run(agent):
    cases = json.loads(CORPUS.read_text(encoding="utf-8"))["cases"]
    planner = DemoResearchPlanner()
    rows = {}
    for case in cases:
        response = agent.respond(case["request"], planner)
        row = response_projection(response)
        if response.status == "ready":
            prepared = response.prepared_task
            row["mock_execution"] = response_projection(agent.execute_response(prepared, prepared.confirmation_id))
        rows[case["id"]] = row
    return rows


def turn_projection(session):
    turns = {}
    for turn in sorted(session.glob("turn-*")):
        files = sorted(str(path.relative_to(turn)).replace(os.sep, "/") for path in turn.rglob("*")
                       if path.is_file() and not path.name.startswith("."))
        response = json.loads((turn / "response.json").read_text(encoding="utf-8"))
        projection = {key: response.get(key) for key in ("status", "submission_state", "report_status")}
        projection["issues"] = [item["code"] for item in response.get("issues", [])]
        turns[turn.name] = {"files": files, "response": projection}
    return turns


def plan(**changes):
    base = dict(request_type="analysis", analysis_type="descriptive_statistics", algorithm_id=None,
                variables=["age at diagnosis", "survival status"], stratification_variables=[],
                sites=["hospital 1", "hospital 2"], database_label=None, parameters={},
                unsupported_requirements=[], purpose="Describe age at diagnosis and survival status")
    return base | changes


def chat(agent, root, replies, provider_outputs):
    provider = Mock()
    provider.model = "snapshot-model"
    provider.generate_json.side_effect = list(provider_outputs)
    planner = LLMResearchPlanner(provider, retries=0)
    args = SimpleNamespace(output=root, provider="local", backend="mock", timeout_seconds=5,
                           confirm_synthetic_unencrypted_sandbox=True, resume=None, stream=False, plain=True)
    replies = iter(replies)
    with redirect_stdout(io.StringIO()):
        run_chat(agent, planner, args, read=lambda _: next(replies))
    session = next(root.glob("session-*"))
    return turn_projection(session)


def build_snapshot():
    snapshot = {}
    with patch("federated_agent.data_availability.datetime") as clock:
        clock.now.return_value = VALID_CLOCK
        snapshot["corpus_ubuntu_valid_inventory"] = corpus_run(build_agent(UBUNTU, data_availability_path=INVENTORY))
    snapshot["corpus_demo_deployment"] = corpus_run(build_agent())
    expired = build_agent(UBUNTU, data_availability_path=INVENTORY)
    cases = json.loads(CORPUS.read_text(encoding="utf-8"))["cases"][:3]
    snapshot["expired_inventory"] = {}
    for case in cases:
        response = expired.respond(case["request"], DemoResearchPlanner())
        row = response_projection(response)
        if response.prepared_task is not None:
            row["mock_execution"] = response_projection(
                expired.execute_response(response.prepared_task, response.prepared_task.confirmation_id))
        snapshot["expired_inventory"][case["id"]] = row

    with tempfile.TemporaryDirectory() as temp, patch("federated_agent.data_availability.datetime") as clock:
        clock.now.return_value = VALID_CLOCK
        agent = build_agent(UBUNTU, data_availability_path=INVENTORY)
        snapshot["chat_clarify_confirm_report"] = chat(agent, Path(temp), [
            "Describe age at diagnosis and survival status in hospital 1 and hospital 2.",
            "synthetic_clean", "confirm", "/history", "/report turn-002", "/exit",
        ], [plan(), {"changes": {"database_label": "synthetic_clean"}}])

    with tempfile.TemporaryDirectory() as temp:
        agent = build_agent()
        snapshot["chat_conversation_unsupported_cancel"] = chat(agent, Path(temp), [
            "hello", "/model", "/algorithms",
            "Describe age at diagnosis only for patients under 25 in hospital 1 and hospital 2 using default.",
            "/new", "Describe age at diagnosis in hospital 1 and hospital 2 using default.", "", "/exit",
        ], [plan(variables=["age at diagnosis"], database_label="default",
                 unsupported_requirements=["patients under 25"]),
            plan(variables=["age at diagnosis"], database_label="default")])
    return json.loads(json.dumps(snapshot, sort_keys=True, default=str))


class BehaviorSnapshotTests(unittest.TestCase):
    maxDiff = None

    def test_user_visible_behaviour_matches_snapshot(self):
        actual = build_snapshot()
        if os.environ.get("UPDATE_BEHAVIOR_SNAPSHOTS") == "1":
            SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
            SNAPSHOT.write_text(json.dumps(actual, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
            self.skipTest("Behaviour snapshot regenerated")
        expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
        for section in expected:
            with self.subTest(section=section):
                self.assertEqual(actual.get(section), expected[section])
        self.assertEqual(sorted(actual), sorted(expected))


if __name__ == "__main__":
    unittest.main()
