import io
import logging
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from federated_agent import cli
from federated_agent.feedback import render_researcher_response
from federated_agent.models import ResearchPlan
from federated_agent.observability import configure_logging, log_incident, scrub
from federated_agent.app import build_agent


class ScrubTests(unittest.TestCase):
    def test_common_credential_forms_are_removed(self):
        text = ("key sk-abcdefghijklmnop Bearer abc.def-ghi postgresql://user:hunter2@db/x "
                "password=hunter2 token: tok123 {\"api_key\": \"value123\"}")
        cleaned = scrub(text)
        for secret in ("abcdefghijklmnop", "abc.def-ghi", "hunter2", "tok123", "value123"):
            self.assertNotIn(secret, cleaned)
        self.assertIn("postgresql://user:***@db/x", cleaned)


class IncidentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.log = Path(self.temp.name) / "agent.log"
        configure_logging(0, self.log)
        self.addCleanup(configure_logging, 0, None)

    def test_incident_is_logged_with_a_redacted_traceback(self):
        logger = logging.getLogger("federated_agent.test")
        try:
            raise RuntimeError("driver failed password=hunter2")
        except RuntimeError as exc:
            with redirect_stderr(io.StringIO()):
                incident = log_incident(logger, "Unexpected failure", exc)
        text = self.log.read_text(encoding="utf-8")
        self.assertIn(incident, text)
        self.assertIn("RuntimeError", text)
        self.assertIn("Traceback", text)
        self.assertNotIn("hunter2", text)

    def test_unexpected_backend_failure_reaches_the_researcher_with_an_incident_id(self):
        agent = build_agent()
        prepared = agent.prepare(ResearchPlan(analysis_type="descriptive_statistics", variables=["age at diagnosis"],
                                              sites=["hospital 1", "hospital 2"], database_label="default",
                                              purpose="incident test"))
        agent.executor = Mock()
        agent.executor.submit.side_effect = RuntimeError("socket reset token=abc123")
        with redirect_stderr(io.StringIO()):
            response = agent.execute_response(prepared, prepared.confirmation_id)
        self.assertEqual(response.status, "execution_failed")
        incident = response.issues[0].incident_id
        self.assertRegex(incident, r"^[0-9a-f]{12}$")
        self.assertIn(f"Incident ID: {incident}", render_researcher_response(response))
        log = self.log.read_text(encoding="utf-8")
        self.assertIn(incident, log)
        self.assertNotIn("abc123", log)

    def test_cli_reports_an_incident_instead_of_a_raw_traceback(self):
        with patch.object(cli, "_run", side_effect=RuntimeError("unexpected")), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as stderr, \
                self.assertRaises(SystemExit) as raised:
            cli.main(["--log-file", str(self.log)])
        self.assertEqual(raised.exception.code, 2)
        self.assertRegex(stderr.getvalue(), r"Unexpected internal error \(incident [0-9a-f]{12}\)")


if __name__ == "__main__":
    unittest.main()
