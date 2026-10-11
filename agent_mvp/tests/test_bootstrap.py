import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from federated_agent.app import describe
from federated_agent.bootstrap import ConfigurationError, build_app_config, load_settings
from federated_agent.cli import build_parser, main
from federated_agent.resources import (
    ALGORITHM_CATALOG, DEFAULT_POLICY, DEMO_DEPLOYMENT, SANDBOX_DATA_AVAILABILITY, SANDBOX_DEPLOYMENT,
)

CLEAN_ENV = {name: "" for name in ("LLM_PROVIDER", "LOCAL_LLM_ENDPOINT", "DEEPSEEK_ENDPOINT", "FED_AGENT_CONFIG")}


def config_for(argv, settings_path=None):
    args = build_parser().parse_args(argv)
    settings, source = load_settings(settings_path)
    return build_app_config(args, settings, source)


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, CLEAN_ENV)
        environment.start()
        self.addCleanup(environment.stop)
        for name in CLEAN_ENV:
            os.environ.pop(name, None)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_config(self, document):
        path = self.root / "fed-agent.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def test_defaults_use_bundled_demo_metadata(self):
        config = config_for([])
        self.assertEqual(config.algorithm_catalog, ALGORITHM_CATALOG)
        self.assertEqual(config.deployment, DEMO_DEPLOYMENT)
        self.assertEqual((config.backend, config.llm.provider, config.llm.runtime), ("mock", "demo", "json"))
        self.assertIsNone(config.data_availability)
        self.assertFalse(config.managed_execution)
        self.assertTrue(config.algorithm_catalog.is_file())

    def test_real_backend_defaults_to_sandbox_metadata_and_managed_execution(self):
        config = config_for(["--backend", "vantage6"])
        self.assertEqual(config.deployment, SANDBOX_DEPLOYMENT)
        self.assertEqual(config.data_availability, SANDBOX_DATA_AVAILABILITY)
        self.assertEqual(config.policy_config, DEFAULT_POLICY)
        self.assertTrue(config.managed_execution)

    def test_file_paths_resolve_relative_to_the_configuration_file(self):
        path = self.write_config({"output": "runs", "metadata": {"algorithm_catalog": "metadata/algorithms.json"},
                                  "logging": {"file": "logs/agent.log", "verbosity": 1}})
        config = config_for([], path)
        self.assertEqual(config.output, (self.root / "runs").resolve())
        self.assertEqual(config.algorithm_catalog, (self.root / "metadata/algorithms.json").resolve())
        self.assertEqual(config.log_file, (self.root / "logs/agent.log").resolve())
        self.assertEqual(config.verbosity, 1)
        self.assertEqual(config.source, path.resolve())

    def test_flags_override_environment_which_overrides_the_file(self):
        path = self.write_config({"llm": {"provider": "deepseek"}})
        self.assertEqual(config_for([], path).llm.provider, "deepseek")
        os.environ["LLM_PROVIDER"] = "local"
        self.assertEqual(config_for([], path).llm.provider, "local")
        self.assertEqual(config_for(["--provider", "demo"], path).llm.provider, "demo")

    def test_configuration_file_can_be_named_by_environment_variable(self):
        os.environ["FED_AGENT_CONFIG"] = str(self.write_config({"backend": {"timeout_seconds": 42}}))
        self.assertEqual(config_for([]).timeout_seconds, 42)

    def test_registry_source_cannot_be_mixed_with_individual_metadata(self):
        path = self.write_config({"metadata": {"registry_source": "registry.json"}})
        with self.assertRaisesRegex(ConfigurationError, "registry-source owns all metadata"):
            config_for(["--catalog", str(ALGORITHM_CATALOG)], path)

    def test_invalid_configuration_names_the_failing_fields(self):
        path = self.write_config({"llm": {"provider": "openai"}, "unknown": True})
        with self.assertRaises(ConfigurationError) as raised:
            config_for([], path)
        self.assertIn("llm.provider", str(raised.exception))
        self.assertIn("unknown", str(raised.exception))

    def test_example_configuration_is_valid(self):
        example = Path(__file__).resolve().parents[1] / "examples" / "fed-agent.example.json"
        settings, source = load_settings(example)
        self.assertEqual(settings.backend.kind, "vantage6")
        self.assertEqual(source, example)

    def test_unreadable_configuration_is_reported(self):
        with self.assertRaisesRegex(ConfigurationError, "Cannot read configuration file"):
            load_settings(self.root / "missing.json")

    def test_describe_reports_health_without_credentials(self):
        key_file = self.root / "key.txt"
        key_file.write_text("sk-file-secret1", encoding="utf-8")
        os.environ["DEEPSEEK_API_KEY"] = "sk-env-secret1"
        config = config_for(["--provider", "deepseek", "--deepseek-api-key-file", str(key_file), "--backend", "vantage6"])
        report = describe(config)
        text = json.dumps(report)
        self.assertNotIn("secret1", text)
        self.assertEqual(report["planner"]["api_key_env"], "DEEPSEEK_API_KEY")
        self.assertTrue(report["planner"]["api_key_file_configured"])
        self.assertEqual(report["policy"]["mode"], "allow_all")
        self.assertTrue(report["metadata"]["data_availability"]["expired"])
        self.assertIn("data availability snapshot has expired; new managed submissions are blocked", report["issues"])
        self.assertTrue(report["algorithms"])

    def test_cli_describe_prints_json_and_exits(self):
        output = io.StringIO()
        with redirect_stdout(output):
            main(["--describe"])
        report = json.loads(output.getvalue())
        self.assertEqual(report["backend"]["kind"], "mock")
        self.assertFalse(report["managed_execution"])

    def test_cli_uses_configuration_file_for_a_request(self):
        path = self.write_config({"output": "out", "llm": {"provider": "demo"}})
        with redirect_stdout(io.StringIO()):
            main(["--config", str(path), "--request",
                  "Describe age at diagnosis and survival status in hospital 1 and hospital 2 using default."])
        response = json.loads((self.root / "out" / "response.json").read_text(encoding="utf-8"))
        self.assertEqual(response["status"], "ready")
        self.assertTrue((self.root / "out" / "prepared_task.json").is_file())


if __name__ == "__main__":
    unittest.main()
