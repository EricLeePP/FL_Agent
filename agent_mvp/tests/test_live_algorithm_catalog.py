import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

from federated_agent.algorithm_catalog import AlgorithmCatalog
from federated_agent.confirmation import task_confirmation_id
from federated_agent.errors import CatalogError, TaskValidationError
from federated_agent.planner import DemoResearchPlanner, LLMResearchPlanner
from federated_agent.script_renderer import render_vantage6_script
from federated_agent.app import build_agent
from federated_agent.resources import ALGORITHM_CATALOG
from federated_agent.services.artifacts import write_response


def plan_data(**changes):
    result = {
        "analysis_type": "descriptive_statistics",
        "variables": ["age at diagnosis"],
        "sites": ["hospital 1", "hospital 2"],
        "database_label": "default",
        "purpose": "Describe age at diagnosis",
    }
    result.update(changes)
    return result


class RecordingProvider:
    def __init__(self, result=None, callback=None):
        self.result = result or plan_data()
        self.callback = callback
        self.prompts = []

    def generate_json(self, system_prompt, user_prompt):
        self.prompts.append(system_prompt)
        if self.callback:
            return self.callback(system_prompt, user_prompt)
        return deepcopy(self.result)


class LiveAlgorithmCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "algorithms.json"
        self.raw = json.loads(ALGORITHM_CATALOG.read_text(encoding="utf-8"))
        self.write_catalog()
        self.agent = build_agent(algorithm_catalog_path=self.path)

    def write_catalog(self):
        self.path.write_text(json.dumps(self.raw, indent=2), encoding="utf-8")

    def respond(self, **changes):
        return self.agent.respond(
            "Research request", LLMResearchPlanner(RecordingProvider(plan_data(**changes))),
        )

    def add_test_algorithm(self):
        item = deepcopy(self.raw["algorithms"][0])
        item.update(
            algorithm_id="test_quality_summary", algorithm_version="1.0.0",
            display_name="Test-only quality summary", analysis_types=["quality_summary"],
            description="Test fixture only, not a deployed algorithm.",
            image="example.invalid/test-only/quality:1.0.0", function="summarize",
            capabilities=["Test-only quality summaries"], request_keywords=["quality summary"],
            accepted_arguments=["columns", "sites", "iterations"],
        )
        item["execution"] = {
            "adapter": "vantage6_kwargs_v1", "result_adapter": "raw_json",
            "input_bindings": {
                "columns": {"source": "variables_by_class"},
                "sites": {"source": "organisation_ids"},
                "iterations": {"source": "parameter", "parameter": "iterations"},
            },
            "parameters_schema": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "iterations": {"type": "integer", "minimum": 1, "maximum": 10, "default": 3},
                },
            },
            "kwargs_schema": {
                "type": "object", "additionalProperties": False,
                "required": ["columns", "sites", "iterations"],
                "properties": {
                    "columns": {"type": "object", "minProperties": 1},
                    "sites": {"type": "array", "minItems": 1, "items": {"type": "integer"}},
                    "iterations": {"type": "integer", "minimum": 1, "maximum": 10},
                },
            },
        }
        self.raw["algorithms"].append(item)
        return item

    def test_each_turn_reads_new_content_even_without_version_or_mtime_change(self):
        provider = RecordingProvider()
        planner = LLMResearchPlanner(provider)
        first = self.agent.respond("Describe age", planner)
        old_stat = self.path.stat()
        self.raw["algorithms"][0]["description"] = "Updated catalog description"
        self.write_catalog()
        os.utime(self.path, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
        second = self.agent.respond("Describe age", planner)
        self.assertEqual(first.status, "ready")
        self.assertEqual(second.status, "ready")
        self.assertNotEqual(first.catalog.sha256, second.catalog.sha256)
        self.assertNotIn("Updated catalog description", provider.prompts[0])
        self.assertIn("Updated catalog description", provider.prompts[1])
        self.assertIn(second.catalog.sha256, provider.prompts[1])
        self.assertNotIn("only currently supported analysis_type", provider.prompts[1])

    def test_file_only_addition_reaches_prompt_and_generic_compiler(self):
        first = self.respond(analysis_type="quality_summary")
        self.assertEqual(first.status, "unsupported")
        self.add_test_algorithm()
        self.write_catalog()
        provider = RecordingProvider(plan_data(analysis_type="quality_summary"))
        response = self.agent.respond("Quality summary", LLMResearchPlanner(provider))
        self.assertEqual(response.status, "ready")
        self.assertIn("test_quality_summary", provider.prompts[0])
        task = response.prepared_task.task_spec
        self.assertEqual(task.algorithm_id, "test_quality_summary")
        self.assertEqual(task.input["method"], "summarize")
        self.assertEqual(task.input["kwargs"]["iterations"], 3)
        self.assertEqual(task.input["kwargs"]["sites"], [1, 2])
        self.assertIn("ncit:C156420", task.input["kwargs"]["columns"])
        self.assertEqual(task.result_adapter, "raw_json")
        script = render_vantage6_script(task)
        compile(script, "generated.py", "exec")
        self.assertIn("example.invalid/test-only/quality:1.0.0", script)

    def test_disable_delete_and_empty_catalog_are_not_cached(self):
        self.assertEqual(self.respond().status, "ready")
        for status in ("draft", "disabled"):
            self.raw["algorithms"][0]["status"] = status
            self.write_catalog()
            self.assertEqual(self.respond().status, "unsupported")
        self.raw["algorithms"] = []
        self.write_catalog()
        self.assertEqual(self.respond().status, "unsupported")
        self.assertEqual(self.agent.describe_algorithms().algorithms, [])

    def test_missing_or_broken_file_prevents_llm_call_without_stale_fallback(self):
        provider = RecordingProvider()
        self.path.write_text("{broken", encoding="utf-8")
        response = self.agent.respond("Describe age", LLMResearchPlanner(provider))
        self.assertEqual(response.status, "catalog_error")
        self.assertIsNone(response.catalog)
        self.assertEqual(provider.prompts, [])
        self.path.unlink()
        self.assertEqual(self.agent.describe_algorithms().status, "catalog_error")

    def test_same_snapshot_is_used_if_catalog_changes_during_llm_call(self):
        original = self.path.read_bytes()

        def change_catalog(system_prompt, user_prompt):
            self.raw["algorithms"][0]["status"] = "disabled"
            self.write_catalog()
            return plan_data()

        response = self.agent.respond(
            "Describe age", LLMResearchPlanner(RecordingProvider(callback=change_catalog)),
        )
        self.assertEqual(response.status, "ready")
        self.assertEqual(response.catalog.sha256, hashlib.sha256(original).hexdigest())
        self.assertEqual(response.prepared_task.task_spec.catalog, response.catalog)
        self.assertEqual(response.catalog_content.encode("utf-8"), original)
        with self.assertRaisesRegex(TaskValidationError, "catalog changed"):
            self.agent.execute(response.prepared_task, response.prepared_task.confirmation_id)
        self.assertEqual(self.respond().status, "unsupported")

    def test_retries_retain_the_original_catalog_snapshot(self):
        def retry(system_prompt, user_prompt):
            if len(provider.prompts) == 1:
                self.raw["algorithms"][0]["description"] = "Changed during retry"
                self.write_catalog()
                return {"analysis_type": "descriptive_statistics"}
            return plan_data()

        provider = RecordingProvider(callback=retry)
        response = self.agent.respond("Describe age", LLMResearchPlanner(provider))
        self.assertEqual(response.status, "ready")
        self.assertEqual(provider.prompts[0], provider.prompts[1])
        self.assertNotIn("Changed during retry", provider.prompts[1])
        with patch.object(provider, "generate_json", side_effect=URLError("offline test")):
            failure = self.agent.respond("Describe age", LLMResearchPlanner(provider))
        self.assertEqual(failure.status, "planner_error")
        self.assertEqual(failure.catalog.sha256, hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_clarification_unsupported_and_invalid_replies_include_source(self):
        cases = [
            ({"sites": []}, "needs_clarification"),
            ({"analysis_type": "survival_analysis"}, "unsupported"),
            ({"parameters": {"invented": 10}}, "invalid_request"),
            ({"unsupported_requirements": ["Only patients younger than 25"]}, "unsupported"),
        ]
        for changes, status in cases:
            with self.subTest(status=status):
                response = self.respond(**changes)
                self.assertEqual(response.status, status)
                self.assertEqual(response.catalog.sha256, hashlib.sha256(self.path.read_bytes()).hexdigest())
                if response.clarification:
                    self.assertEqual(response.clarification.catalog, response.catalog)

    def test_catalog_questions_return_data_not_model_authored_capability_claims(self):
        response = self.respond(request_type="catalog_query", analysis_type="catalog_query")
        self.assertEqual(response.status, "algorithms")
        self.assertIsNone(response.prepared_task)
        self.assertEqual(response.algorithms[0]["availability"][0]["verification"], "declared")
        self.assertEqual(response.algorithms[0]["capabilities"], self.raw["algorithms"][0]["capabilities"])

    def test_model_cannot_invent_algorithm_or_use_another_analysis_contract(self):
        self.assertEqual(self.respond(algorithm_id="invented").status, "unsupported")
        self.assertEqual(self.respond(
            analysis_type="survival_analysis", algorithm_id="strongaya_descriptive_statistics",
        ).status, "unsupported")

    def test_multiple_matching_algorithms_require_explicit_selection(self):
        extra = self.add_test_algorithm()
        extra["analysis_types"] = ["descriptive_statistics"]
        self.write_catalog()
        response = self.respond()
        self.assertEqual(response.status, "needs_clarification")
        self.assertEqual(response.clarification.questions[0].field, "algorithm_id")
        self.assertEqual(len(response.clarification.questions[0].candidates), 2)
        self.assertEqual(self.respond(algorithm_id="test_quality_summary").status, "ready")

    def test_parameters_are_validated_and_defaults_are_attributed(self):
        self.add_test_algorithm()
        self.write_catalog()
        for value in (0, 11, "3", True):
            with self.subTest(value=value):
                self.assertEqual(self.respond(
                    analysis_type="quality_summary", parameters={"iterations": value},
                ).status, "invalid_request")
        default = self.respond(analysis_type="quality_summary").prepared_task.task_spec
        requested = self.respond(
            analysis_type="quality_summary", parameters={"iterations": 5},
        ).prepared_task.task_spec
        def source(task):
            return next(item.source.kind for item in task.provenance if item.field == "input.kwargs.iterations")
        self.assertEqual(source(default), "algorithm_registry")
        self.assertEqual(source(requested), "research_plan")
        self.assertEqual(requested.input["kwargs"]["iterations"], 5)

    def test_missing_required_parameter_asks_instead_of_guessing(self):
        extra = self.add_test_algorithm()
        schema = extra["execution"]["parameters_schema"]
        del schema["properties"]["iterations"]["default"]
        schema["required"] = ["iterations"]
        self.write_catalog()
        response = self.respond(analysis_type="quality_summary")
        self.assertEqual(response.status, "needs_clarification")
        self.assertEqual(response.clarification.questions[0].field, "parameters.iterations")

    def test_declared_hospital_deployment_and_database_scope_are_enforced(self):
        scope = self.raw["algorithms"][0]["availability"][0]
        for changes in (
            {"organisation_ids": [1]}, {"organisation_ids": [2]},
            {"deployment": "other-deployment"}, {"database_labels": ["other-label"]},
        ):
            original = deepcopy(scope)
            scope.update(changes)
            self.write_catalog()
            self.assertEqual(self.respond().status, "invalid_request")
            scope.clear()
            scope.update(original)

    def test_unknown_execution_adapter_is_visible_but_cannot_compile(self):
        self.raw["algorithms"][0]["execution"]["adapter"] = "new_iterative_protocol"
        self.write_catalog()
        self.assertFalse(self.agent.describe_algorithms().algorithms[0]["adapter_supported"])
        response = self.respond()
        self.assertEqual(response.status, "invalid_request")
        self.assertIn("requires a new adapter", response.message)

    def test_invalid_contracts_fail_loading(self):
        original = deepcopy(self.raw)
        variants = [
            lambda raw: raw.update(schema_version="999"),
            lambda raw: raw["algorithms"].append(deepcopy(raw["algorithms"][0])),
            lambda raw: raw["algorithms"][0].update(image="example.invalid/algo:latest"),
            lambda raw: raw["algorithms"][0]["execution"]["input_bindings"].update(extra={"source": "constant", "value": 1}),
            lambda raw: raw["algorithms"][0]["execution"]["parameters_schema"].update(additionalProperties=True),
            lambda raw: raw["algorithms"][0]["execution"]["parameters_schema"].update({"$ref": "https://example.invalid/schema"}),
            lambda raw: raw["algorithms"][0]["execution"]["input_bindings"]["organisation_ids"].update(source="eval_python"),
            lambda raw: raw["algorithms"][0]["execution"]["input_bindings"].update(organisation_ids={"source": "constant", "value": [1, 2]}),
            lambda raw: raw["algorithms"][0].update(supports_stratification=True),
            lambda raw: raw["algorithms"][0]["execution"]["kwargs_schema"].update({"$schema": "https://json-schema.org/draft-07/schema"}),
        ]
        for index, mutation in enumerate(variants):
            with self.subTest(index=index):
                self.raw = deepcopy(original)
                mutation(self.raw)
                self.write_catalog()
                with self.assertRaises(CatalogError):
                    AlgorithmCatalog.from_json(self.path)
        self.raw = deepcopy(original)
        extra = self.add_test_algorithm()
        parameter = extra["execution"]["parameters_schema"]["properties"]["iterations"]
        parameter["default"] = 100
        self.write_catalog()
        with self.assertRaises(CatalogError):
            AlgorithmCatalog.from_json(self.path)

    def test_duplicate_json_keys_are_rejected(self):
        self.path.write_text('{"schema_version": "1.0", "schema_version": "2.0"}', encoding="utf-8")
        with self.assertRaisesRegex(CatalogError, "Duplicate JSON key"):
            AlgorithmCatalog.from_json(self.path)
        self.write_catalog()
        self.path.write_text(self.path.read_text(encoding="utf-8").replace('"value": null', '"value": 1e999'), encoding="utf-8")
        with self.assertRaises(CatalogError):
            AlgorithmCatalog.from_json(self.path)

    def test_byte_exact_snapshot_is_archived_even_with_bom(self):
        self.path.write_bytes(b"\xef\xbb\xbf" + self.path.read_bytes())
        response = self.respond()
        output = Path(self.temporary.name)
        write_response(response, output)
        self.assertEqual((output / "catalog_snapshot.json").read_bytes(), self.path.read_bytes())
        data = json.loads((output / "response.json").read_text(encoding="utf-8"))
        self.assertEqual(data["catalog"]["sha256"], response.catalog.sha256)

    def test_prepared_task_revalidation_rejects_a_forged_payload(self):
        prepared = self.respond().prepared_task
        prepared.task_spec.input["kwargs"]["organisation_ids"] = [99]
        prepared.confirmation_id = task_confirmation_id(prepared.task_spec)
        with self.assertRaisesRegex(TaskValidationError, "no longer matches"):
            self.agent.execute(prepared, prepared.confirmation_id)

    def test_changed_catalog_blocks_submission_through_export_and_agent(self):
        prepared = self.respond().prepared_task
        script = render_vantage6_script(prepared.task_spec)
        fake_client = types.ModuleType("vantage6.client")
        fake_client.UserClient = lambda *args, **kwargs: self.fail("Client must not be created")
        self.raw["catalog_version"] = "new-version"
        self.write_catalog()
        with patch.dict(sys.modules, {"vantage6": types.ModuleType("vantage6"), "vantage6.client": fake_client}):
            with self.assertRaisesRegex(SystemExit, "cannot bypass"), redirect_stdout(io.StringIO()):
                exec(compile(script, "generated.py", "exec"), {})
        response = self.agent.execute_response(prepared, prepared.confirmation_id)
        self.assertEqual(response.issues[0].code, "catalog_changed")
        self.assertEqual(response.submission_state, "not_submitted")

    def test_cli_lists_validates_and_generates_offline(self):
        root = Path(__file__).resolve().parents[1]
        for args in (
            ["--list-algorithms"], ["--validate-catalog"],
            ["--provider", "demo", "--request", "Describe age at diagnosis in hospital 1 and hospital 2 using default"],
        ):
            with self.subTest(args=args):
                result = subprocess.run(
                    [sys.executable, "-m", "federated_agent", "--catalog", str(self.path),
                     "--output", self.temporary.name, *args],
                    cwd=root, capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((Path(self.temporary.name) / "run_vantage6_task.py").exists())

    def test_demo_catalog_query_is_file_backed(self):
        response = self.agent.respond("List algorithms", DemoResearchPlanner())
        self.assertEqual(response.status, "algorithms")
        self.assertEqual(len(response.algorithms), 1)


if __name__ == "__main__":
    unittest.main()
