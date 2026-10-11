"""Phase 5 guarantees: algorithm-specific result handling is a plugin named by the catalog."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from federated_agent import result_adapters
from federated_agent.app import build_agent, describe
from federated_agent.config import AppConfig
from federated_agent.errors import TaskValidationError
from federated_agent.models import ExecutionRecord, ExecutionStatus, ResearchPlan
from federated_agent.resources import DEMO_DEPLOYMENT, RESOURCE_ROOT, SEMANTIC_SCHEMA
from federated_agent.result_adapters import BUILT_IN, installed, register
from federated_agent.services.reporting import generate_report


def plan():
    return ResearchPlan(analysis_type="descriptive_statistics", variables=["age at diagnosis"],
                        sites=["hospital 1", "hospital 2"], database_label="default", purpose="plugin test")


class RowCountResult:
    """A third-party result type: its own check and its own report."""

    name = "row_count_v1"
    reports = True

    def __init__(self):
        self.checked = []

    def check(self, raw, plan, semantic_catalog):
        if not isinstance(raw, dict) or not isinstance(raw.get("rows"), int):
            raise ValueError("No row count in the result")
        self.checked.append((raw, [site.canonical_name for site in plan.sites]))

    def write_report(self, raw, output, *, semantic_catalog, context, chart, emit):
        path = output / "row_count.txt"
        path.write_text(f"{context['Task ID']}: {raw.get('rows')} rows", encoding="utf-8")
        emit("  rows: reported")
        return path


class FixedPlanner:
    def create_plan(self, request, catalog):
        return plan()


class RealExecutor:
    """Undeclared capabilities: treated as a real backend, so results are checked."""

    def __init__(self, result):
        self.result = result

    def submit(self, task_spec):
        return ExecutionRecord(task_id="41", status=ExecutionStatus.submitted)

    def wait_for_results(self, task_id):
        return ExecutionRecord(task_id=task_id, status=ExecutionStatus.completed, result=self.result)


class FakeEntryPoint:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def load(self):
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


class ResultAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        registered = patch.dict(result_adapters._registered, {}, clear=True)
        registered.start()
        self.addCleanup(registered.stop)

    def catalog_with(self, adapter_name):
        document = json.loads((RESOURCE_ROOT / "catalogs/algorithms.json").read_text(encoding="utf-8"))
        for algorithm in document["algorithms"]:
            if algorithm.get("execution"):
                algorithm["execution"]["result_adapter"] = adapter_name
        path = self.root / f"{adapter_name}.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def agent_for(self, adapter_name):
        return build_agent(DEMO_DEPLOYMENT, self.catalog_with(adapter_name))

    def test_an_algorithm_naming_a_missing_adapter_cannot_be_prepared(self):
        agent = self.agent_for("row_count_v1")
        with self.assertRaises(TaskValidationError) as raised:
            agent.prepare(plan())
        self.assertEqual(raised.exception.code, "result_adapter_missing")
        response = agent.respond("Count rows", FixedPlanner())
        self.assertEqual((response.status, response.issues[0].code), ("invalid_request", "result_adapter_missing"))

    def test_a_registered_plugin_checks_and_reports_its_own_results(self):
        plugin = RowCountResult()
        register(plugin)
        agent = self.agent_for("row_count_v1")
        prepared = agent.prepare(plan())
        agent.executor = RealExecutor({"rows": 3})
        response = agent.execute_response(prepared, prepared.confirmation_id)
        self.assertEqual(response.status, "completed")
        self.assertEqual(plugin.checked, [({"rows": 3}, ["hospital 1", "hospital 2"])])
        lines = []
        reported = generate_report(agent, prepared, response, self.root, "vantage6", emit=lines.append)
        self.assertEqual((reported.status, reported.report_status), ("completed", "completed"))
        self.assertEqual((self.root / "row_count.txt").read_text(encoding="utf-8"), "41: 3 rows")
        self.assertIn("  rows: reported", lines)
        self.assertFalse((self.root / "result.html").exists())

    def test_a_result_the_plugin_rejects_is_reported_as_invalid(self):
        register(RowCountResult())
        agent = self.agent_for("row_count_v1")
        prepared = agent.prepare(plan())
        agent.executor = RealExecutor({"unexpected": True})
        response = agent.execute_response(prepared, prepared.confirmation_id)
        self.assertEqual((response.issues[0].code, response.submission_state), ("result_invalid", "submitted"))

    def test_raw_json_keeps_the_result_without_a_report(self):
        agent = self.agent_for("raw_json")
        prepared = agent.prepare(plan())
        agent.executor = RealExecutor({"anything": [1, 2]})
        response = agent.execute_response(prepared, prepared.confirmation_id)
        reported = generate_report(agent, prepared, response, self.root, "vantage6", emit=lambda _: None)
        self.assertEqual((reported.status, reported.report_status), ("completed", "not_requested"))
        self.assertTrue((self.root / "execution.json").exists())

    def test_plugins_are_discovered_but_cannot_replace_built_ins(self):
        class FakeDescriptive(RowCountResult):
            name = "descriptive_statistics_v1"

        entries = [FakeEntryPoint("rows", RowCountResult), FakeEntryPoint("shadow", FakeDescriptive),
                   FakeEntryPoint("broken", ImportError("missing dependency")),
                   FakeEntryPoint("nameless", SimpleNamespace(reports=True))]
        result_adapters._plugins.cache_clear()
        self.addCleanup(result_adapters._plugins.cache_clear)
        with patch("federated_agent.result_adapters.entry_points", return_value=entries), \
                self.assertLogs("federated_agent.result_adapters", "WARNING") as logs:
            adapters = installed()
        self.assertIsInstance(adapters["row_count_v1"], RowCountResult)
        self.assertIs(adapters["descriptive_statistics_v1"], BUILT_IN["descriptive_statistics_v1"])
        self.assertEqual(len(logs.records), 3)
        with self.assertRaises(ValueError):
            register(FakeDescriptive())

    def test_describe_reports_whether_each_adapter_is_installed(self):
        config = AppConfig(output=self.root, backend="mock", confirm_synthetic_unencrypted_sandbox=False,
                           timeout_seconds=600, algorithm_catalog=self.catalog_with("row_count_v1"),
                           semantic_schema=SEMANTIC_SCHEMA, deployment=DEMO_DEPLOYMENT)
        report = describe(config)
        executable = [item for item in report["algorithms"] if item["result_adapter"]]
        self.assertTrue(executable)
        self.assertFalse(any(item["result_adapter_installed"] for item in executable))


if __name__ == "__main__":
    unittest.main()
