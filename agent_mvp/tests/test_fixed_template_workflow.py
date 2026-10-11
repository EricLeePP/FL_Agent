"""Fixed colleague workflow, LLM parameter proposals and deterministic checks."""

import ast
from datetime import datetime, timezone
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from federated_agent.resources import RESOURCE_ROOT, SOURCE_CHECKOUT
from federated_agent.models import ResearchPlan
from federated_agent.planner import LLMResearchPlanner
from federated_agent.runs.submission import submit_task
from federated_agent.script_renderer import render_vantage6_script
from federated_agent.app import build_agent
from federated_agent.runs.worker import run_worker
from federated_agent.storage import read_json, write_json


class FixedTemplateWorkflowTests(unittest.TestCase):
    def setUp(self):
        clock = patch("federated_agent.data_availability.datetime")
        self.addCleanup(clock.stop)
        clock.start().now.return_value = datetime(2026, 9, 7, tzinfo=timezone.utc)
        self.agent = build_agent(
            RESOURCE_ROOT / "config/ubuntu_sandbox.json",
            data_availability_path=SOURCE_CHECKOUT / "tests/fixtures/ubuntu_data_availability_20260907.json",
        )
        self.provider = Mock()
        self.planner = LLMResearchPlanner(self.provider, retries=0)

    def propose(self, **changes):
        values = dict(
            analysis_type="descriptive_statistics", variables=["age at diagnosis"],
            sites=["hospital 1", "hospital 2"], database_label="synthetic_clean",
            purpose="Describe the requested synthetic data",
        )
        values.update(changes)
        self.provider.generate_json.return_value = ResearchPlan(**values).model_dump()
        return self.agent.respond("Synthetic test request", self.planner)

    @staticmethod
    def fixed_script_structure(script):
        tree = ast.parse(script)
        # Only the serialized task/confirmation values are generation slots.
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in {
                    "confirmation_material", "CONFIRMATION_ID"
                }
                for target in node.targets
            ):
                node.value = ast.Constant(value="TASK_PARAMETER_SLOT")
        return ast.dump(tree)

    @staticmethod
    def submission_keywords(script):
        calls = [node for node in ast.walk(ast.parse(script))
                 if isinstance(node, ast.Call) and ast.unparse(node.func) == "client.task.create"]
        if len(calls) != 1:
            raise AssertionError("The fixed workflow must have one submission call")
        return [keyword.arg for keyword in calls[0].keywords]

    def test_different_requests_only_change_validated_parameter_slots(self):
        first = self.propose().prepared_task.task_spec
        second = self.propose(
            variables=["survival status"], sites=["hospital 3"],
            database_label="synthetic_missing",
        ).prepared_task.task_spec
        self.assertEqual(first.input["kwargs"]["organisation_ids"], [1, 2])
        self.assertEqual(second.input["kwargs"]["organisation_ids"], [3])
        self.assertEqual(second.input["kwargs"]["variables_to_describe"],
                         {"ncit:C25717": {"datatype": "categorical"}})
        self.assertEqual(second.to_client_payload()["databases"], [{"label": "synthetic_missing"}])
        self.assertEqual(first.image, second.image)
        self.assertEqual(first.to_client_payload()["organizations"], [1])
        self.assertEqual(second.to_client_payload()["organizations"], [1])
        self.assertEqual(self.fixed_script_structure(render_vantage6_script(first)),
                         self.fixed_script_structure(render_vantage6_script(second)))

    def test_prompt_defines_parameter_role_and_supplies_algorithm_contract(self):
        self.assertEqual(self.propose().status, "ready")
        system_prompt, _ = self.provider.generate_json.call_args.args
        self.assertIn("fixed", system_prompt)
        self.assertIn("Do not rewrite that workflow", system_prompt)
        self.assertIn("CATALOG_JSON:", system_prompt)
        self.assertIn("parameters_schema", system_prompt)
        self.assertIn("strongaya_descriptive_statistics", system_prompt)

    def test_invalid_llm_proposals_cannot_fill_an_executable_template(self):
        cases = [
            ({"sites": ["hospital 99"]}, "needs_clarification"),
            ({"variables": ["unregistered_variable"]}, "needs_clarification"),
            ({"database_label": "synthetic_nonexistent"}, "needs_clarification"),
            ({"variables": ["gender"]}, "data_unavailable"),
            ({"analysis_type": "cox", "algorithm_id": "unknown_cox"}, "unsupported"),
            ({"parameters": {"learning_rate": 0.1}}, "invalid_request"),
            ({"unsupported_requirements": ["age < 25"]}, "unsupported"),
        ]
        executor = Mock()
        self.agent.executor = executor
        for changes, expected in cases:
            with self.subTest(changes=changes):
                response = self.propose(**changes)
                self.assertEqual(response.status, expected)
                self.assertIsNone(response.prepared_task)
                self.assertEqual(response.submission_state, "not_submitted")
        executor.submit.assert_not_called()

    def test_llm_cannot_supply_script_or_image_fields(self):
        self.propose()
        valid = self.provider.generate_json.return_value
        for field in ("python_code", "image", "method", "organizations"):
            with self.subTest(field=field):
                self.provider.generate_json.return_value = {**valid, field: "untrusted override"}
                response = self.agent.respond("Synthetic test request", self.planner)
                self.assertEqual(response.status, "planner_error")
                self.assertIsNone(response.prepared_task)

    def test_runtime_and_export_use_the_same_fixed_submission_shape(self):
        task = self.propose().prepared_task.task_spec
        expected = ["collaboration", "organizations", "name", "image", "description", "input_", "databases"]
        # One implementation: the worker calls submit_task and the export embeds its exact source.
        self.assertEqual(self.submission_keywords(inspect.getsource(submit_task)), expected)
        self.assertIn("submit_task(client, payload)", inspect.getsource(run_worker))
        self.assertNotIn("task.create", inspect.getsource(run_worker))
        script = render_vantage6_script(task)
        self.assertEqual(self.submission_keywords(script), expected)
        self.assertIn(inspect.getsource(submit_task).rstrip("\n"), script)
        client = Mock()
        events = []
        client.util.get_server_version.return_value = {"version": "4.15.0"}
        client.authenticate.side_effect = lambda *_: events.append("authenticate")
        client.collaboration.get.return_value = {"encrypted": False}
        client.setup_encryption.side_effect = lambda *_: events.append("encryption")
        client.task.create.side_effect = lambda **_: events.append("submit") or {"id": 71}
        client.task.get.side_effect = lambda *_: events.append("wait") or {"status": "completed"}
        raw = {"data": [{"result": "{}"}]}
        client.result.from_task.side_effect = lambda **_: events.append("retrieve") or raw
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_json(output / "task_spec.json", task.model_dump(mode="json"))
            credentials = {"username": "fixture", "password": "TEST_ONLY_NOT_A_REAL_PASSWORD"}
            run_worker(output, credentials, client_factory=lambda _: client)
            self.assertEqual(read_json(output / "state.json")["status"], "completed")
            self.assertEqual(read_json(output / "result.raw.json"), raw)
            self.assertEqual(credentials, {})
        client.task.create.assert_called_once_with(**task.to_client_payload())
        self.assertEqual(events, ["authenticate", "encryption", "submit", "wait", "retrieve"])


if __name__ == "__main__":
    unittest.main()
