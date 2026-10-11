import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from federated_agent.deployment import DeploymentRegistry
from federated_agent.errors import (
    ClarificationRequiredError,
    TaskValidationError,
    UnsupportedAnalysisError,
)
from federated_agent.executor import ClassicVantage6Executor
from federated_agent.models import ResearchPlan
from federated_agent.planner import DeepSeekJSONProvider, LLMResearchPlanner
from federated_agent.result_parser import ResultParser
from federated_agent.script_renderer import render_vantage6_script
from federated_agent.task_compiler import Vantage6TaskCompiler
from federated_agent.app import build_agent


ROOT = Path(__file__).resolve().parents[1]




def descriptive_plan(**overrides) -> ResearchPlan:
    values = {
        "analysis_type": "descriptive_statistics",
        "variables": ["age at diagnosis", "survival status"],
        "sites": ["hospital 1", "hospital 2"],
        "database_label": "default",
        "purpose": "Describe the study population",
    }
    values.update(overrides)
    return ResearchPlan(**values)


class AgentPipelineTests(unittest.TestCase):
    def test_full_upstream_schema_is_loaded(self):
        self.assertEqual(len(build_agent().semantic_catalog.variables), 208)

    def test_compiles_semantic_classes_and_deployment_ids(self):
        prepared = build_agent().prepare(descriptive_plan())
        kwargs = prepared.task_spec.input["kwargs"]
        self.assertEqual(kwargs["organisation_ids"], [1, 2])
        self.assertEqual(
            kwargs["variables_to_describe"]["ncit:C156420"]["datatype"],
            "numerical",
        )
        self.assertEqual(
            kwargs["variables_to_describe"]["ncit:C25717"]["datatype"],
            "categorical",
        )
        self.assertNotIn("organization_ids", kwargs)
        self.assertNotIn("return_partials", kwargs)
        self.assertTrue(prepared.policy_decision.allowed)

    def test_vantage6_415_payload_uses_input_and_database_contract(self):
        prepared = build_agent().prepare(descriptive_plan())
        payload = prepared.task_spec.to_client_payload()
        self.assertIn("input_", payload)
        self.assertNotIn("input", payload)
        self.assertEqual(payload["organizations"], [1])
        self.assertEqual(payload["databases"], [{"label": "default"}])
        self.assertEqual(set(payload["input_"]), {"method", "kwargs"})

    def test_deployment_registry_resolves_alias_without_llm_ids(self):
        registry = build_agent().deployment_registry
        site = registry.resolve("centre 2")
        self.assertEqual(site.canonical_name, "hospital 2")
        self.assertEqual(site.organisation_id, 2)
        self.assertIsNone(site.database_label)

    def test_missing_sites_returns_structured_clarification(self):
        with self.assertRaises(ClarificationRequiredError) as raised:
            build_agent().prepare(descriptive_plan(sites=[]))
        request = raised.exception.request
        self.assertEqual(request.status, "needs_clarification")
        self.assertEqual(request.questions[0].field, "sites")
        self.assertIn("hospital 1", request.questions[0].candidates)

    def test_unknown_site_returns_candidates_in_clarification(self):
        with self.assertRaises(ClarificationRequiredError) as raised:
            build_agent().prepare(descriptive_plan(sites=["hospital 99"]))
        question = raised.exception.request.questions[0]
        self.assertEqual(question.requested_value, "hospital 99")
        self.assertTrue(question.candidates)

    def test_missing_variables_returns_structured_clarification(self):
        with self.assertRaises(ClarificationRequiredError) as raised:
            build_agent().prepare(descriptive_plan(variables=[]))
        self.assertEqual(raised.exception.request.questions[0].field, "variables")

    def test_identifier_is_rejected_by_task_validation(self):
        with self.assertRaises(TaskValidationError):
            build_agent().prepare(
                descriptive_plan(variables=["identifier"], sites=["hospital 1"])
            )

    def test_unsupported_analysis_returns_review_workflow_message(self):
        with self.assertRaisesRegex(UnsupportedAnalysisError, "No approved algorithm"):
            build_agent().prepare(
                descriptive_plan(
                    analysis_type="survival_analysis",
                    variables=["survival status", "survival time"],
                )
            )

    def test_mock_execution_requires_matching_confirmation(self):
        agent = build_agent()
        prepared = agent.prepare(descriptive_plan())
        with self.assertRaisesRegex(TaskValidationError, "confirmation"):
            agent.execute(prepared, "wrong-id")
        execution = agent.execute(prepared, prepared.confirmation_id)
        self.assertEqual(execution.status.value, "completed")

    def test_changed_task_is_rejected_after_confirmation(self):
        agent = build_agent()
        prepared = agent.prepare(descriptive_plan())
        prepared.task_spec.description = "Changed after review"
        with self.assertRaisesRegex(TaskValidationError, "content changed"):
            agent.execute(prepared, prepared.confirmation_id)

    def test_mock_execution_and_result_report(self):
        agent = build_agent()
        prepared = agent.prepare(descriptive_plan())
        execution = agent.execute(prepared, prepared.confirmation_id)
        parsed = ResultParser(agent.semantic_catalog).parse(execution.result or {})
        self.assertEqual(parsed.numerical[0]["variable_name"], "age_at_initial_diagnosis")
        self.assertEqual(parsed.categorical[0]["variable_name"], "survival_status")
        with tempfile.TemporaryDirectory() as directory:
            report = ResultParser(agent.semantic_catalog).write_html(
                parsed, Path(directory) / "result.html"
            )
            self.assertTrue(report.exists())
            self.assertIn("Numerical statistics", report.read_text(encoding="utf-8"))

    def test_result_parser_unwraps_vantage6_result_resource(self):
        agent = build_agent()
        wrapped = [
            {
                "id": 99,
                "result": {
                    "numerical_general_statistics": json.dumps(
                        {
                            "variable": {"0": "ncit:C156420"},
                            "statistic": {"0": "count"},
                            "value": {"0": 25},
                        }
                    )
                },
            }
        ]
        parsed = agent.result_parser.parse(wrapped)
        self.assertEqual(parsed.numerical[0]["value"], 25)
        self.assertEqual(
            parsed.numerical[0]["variable_name"], "age_at_initial_diagnosis"
        )

    def test_task_records_versions_warnings_and_provenance(self):
        prepared = build_agent().prepare(descriptive_plan())
        task = prepared.task_spec
        self.assertEqual(task.algorithm_version, "2.1.0")
        self.assertEqual(task.connection.version, "4.15.0")
        self.assertTrue(task.warnings)
        fields = {item.field for item in task.provenance}
        self.assertIn("collaboration_id", fields)
        self.assertIn(
            "input.kwargs.variables_to_describe.ncit:C156420", fields
        )
        json.dumps(task.to_client_payload())

    def test_compiler_rejects_vantage6_version_mismatch(self):
        agent = build_agent()
        prepared = agent.prepare(descriptive_plan())
        profile = agent.deployment_registry.profile.model_copy(
            update={"vantage6_version": "4.14.0"}
        )
        mismatched = DeploymentRegistry(profile, agent.deployment_registry.source)
        compiler = Vantage6TaskCompiler(mismatched)
        algorithm = agent.algorithm_catalog.select("descriptive_statistics")
        with self.assertRaisesRegex(TaskValidationError, "versions differ"):
            compiler.compile(prepared.resolved_plan, algorithm)

    def test_llm_planner_repairs_invalid_structured_output(self):
        class FakeProvider:
            def __init__(self):
                self.calls = 0

            def generate_json(self, system_prompt, user_prompt):
                self.calls += 1
                if self.calls == 1:
                    return {"analysis_type": "descriptive_statistics"}
                return {
                    "analysis_type": "descriptive_statistics",
                    "variables": ["gender"],
                    "stratification_variables": [],
                    "sites": ["hospital 1"],
                    "purpose": "Describe gender",
                }

        provider = FakeProvider()
        plan = LLMResearchPlanner(provider, retries=1).create_plan("Describe gender", build_agent().algorithm_catalog.snapshot())
        self.assertEqual(plan.variables, ["gender"])
        self.assertEqual(plan.sites, ["hospital 1"])
        self.assertEqual(provider.calls, 2)

    def test_deepseek_provider_uses_remote_json_contract(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(
                    {
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "analysis_type": "descriptive_statistics",
                                            "variables": ["gender"],
                                            "stratification_variables": [],
                                            "sites": ["hospital 1"],
                                            "purpose": "Describe gender",
                                        }
                                    )
                                },
                            }
                        ]
                    }
                ).encode("utf-8")

        provider = DeepSeekJSONProvider(api_key="secret")
        with patch(
            "federated_agent.planner.urllib.request.urlopen",
            return_value=FakeResponse(),
        ) as urlopen:
            result = provider.generate_json("Return JSON", "Describe gender")

        request = urlopen.call_args.args[0]
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, DeepSeekJSONProvider.DEFAULT_ENDPOINT)
        self.assertEqual(request.headers["Authorization"], "Bearer secret")
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(body["thinking"], {"type": "disabled"})
        self.assertEqual(result["sites"], ["hospital 1"])

    def test_generated_script_keeps_credentials_outside_task(self):
        prepared = build_agent().prepare(descriptive_plan())
        script = render_vantage6_script(prepared.task_spec)
        compile(script, "run_vantage6_task.py", "exec")
        self.assertIn("actual_confirmation", script)
        self.assertIn("def submit_task(client, payload):", script)
        self.assertIn(prepared.confirmation_id, script)
        self.assertIn(f"--execute-prepared prepared_task.json --confirm {{CONFIRMATION_ID}}", script)
        # The review copy never handles credentials; only the agent's worker does.
        for absent in ("getpass", "password", "VANTAGE6_USERNAME", "VANTAGE6_SERVER_URL", "DEEPSEEK_API_KEY", "UserClient"):
            self.assertNotIn(absent, script)

    def test_classic_vantage6_adapter_uses_documented_client_contract(self):
        class FakeTaskClient:
            def __init__(self):
                self.payload = None

            def create(self, **payload):
                self.payload = payload
                return {"id": 42}

            def get(self, task_id):
                return {"id": task_id, "status": "completed"}

        class FakeResultClient:
            def __init__(self):
                self.task_id = None

            def from_task(self, task_id):
                self.task_id = task_id
                return [{"result": {"numerical_general_statistics": "{}"}}]

        class FakeClient:
            def __init__(self):
                self.task = FakeTaskClient()
                self.result = FakeResultClient()
                self.waited_for = None

            def wait_for_results(self, task_id):
                self.waited_for = task_id

        prepared = build_agent().prepare(descriptive_plan(variables=["gender"]))
        client = FakeClient()
        executor = ClassicVantage6Executor(client)
        submitted = executor.submit(prepared.task_spec)
        completed = executor.wait_for_results(submitted.task_id)

        self.assertEqual(submitted.task_id, "42")
        self.assertEqual(client.waited_for, 42)
        self.assertEqual(client.result.task_id, 42)
        self.assertEqual(client.task.payload["organizations"], [1])
        self.assertIn("input_", client.task.payload)
        self.assertEqual(client.task.payload["databases"], [{"label": "default"}])
        self.assertEqual(completed.status.value, "completed")


if __name__ == "__main__":
    unittest.main()
