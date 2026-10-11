"""Exercise the short Windows entry point without a model or remote task."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "agent.cmd"
RUNTIME = ROOT / "integration" / ".venv-vantage6" / "Scripts" / "python.exe"


@unittest.skipUnless(os.name == "nt", "Windows command launcher")
class WindowsLauncherTests(unittest.TestCase):
    def run_launcher(self, launcher, *arguments, cwd, text=None):
        environment = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        return subprocess.run(
            [str(launcher), *map(str, arguments)],
            cwd=cwd,
            input=text,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            env=environment,
        )

    def require_runtime(self):
        if not RUNTIME.is_file():
            self.skipTest("Project's Windows demonstration environment is not installed")

    def test_help_from_another_working_directory(self):
        self.require_runtime()
        with tempfile.TemporaryDirectory(prefix="agent launch ") as directory:
            result = self.run_launcher(LAUNCHER, "--help", cwd=directory)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--provider", result.stdout)
        self.assertIn("--backend", result.stdout)

    def test_invalid_option_preserves_nonzero_exit_code(self):
        self.require_runtime()
        result = self.run_launcher(LAUNCHER, "--not-an-agent-option", cwd=ROOT)
        self.assertEqual(result.returncode, 2)
        self.assertIn("unrecognized arguments", result.stderr)

    def test_offline_override_and_output_path_with_spaces(self):
        self.require_runtime()
        with tempfile.TemporaryDirectory(prefix="agent launch ") as directory:
            output = Path(directory) / "output with spaces"
            result = self.run_launcher(
                LAUNCHER, "--provider", "demo", "--backend", "mock", "--plain",
                "--output", output, cwd=directory, text="hi\n/exit\n",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(list(output.glob("session-*/turn-*/response.json")))
            self.assertFalse(list(output.rglob("submission.json")))
            self.assertFalse(list(output.rglob("task_spec.json")))

    def test_missing_runtime_in_relocated_project_with_spaces(self):
        with tempfile.TemporaryDirectory(prefix="agent launch ") as directory:
            relocated = Path(directory) / "agent.cmd"
            shutil.copyfile(LAUNCHER, relocated)
            result = self.run_launcher(relocated, cwd=ROOT)
        self.assertEqual(result.returncode, 1)
        self.assertIn("project Python environment was not found", result.stderr)
        self.assertIn("AGENT_OPERATIONS_AND_REGISTRY.md", result.stderr)


if __name__ == "__main__":
    unittest.main()
