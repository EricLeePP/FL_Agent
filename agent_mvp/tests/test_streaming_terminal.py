import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from http.client import IncompleteRead
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.output.vt100 import Vt100_Output
from prompt_toolkit.data_structures import Size
from rich.console import Console
from wcwidth import wcswidth

from federated_agent.cli import run_chat
from federated_agent.errors import LLMResponseError
from federated_agent.models import AgentResponse, ExecutionRecord, ExecutionStatus, ResearchPlan
from federated_agent.planner import DeepSeekJSONProvider, LLMResearchPlanner
from federated_agent.terminal_ui import TerminalUI, choice_application, terminal_text
from federated_agent.app import build_agent
from federated_agent.narration import reply_facts


def event(content=None, finish=None, **delta):
    value = {"choices": [{"index": 0, "delta": {"content": content, **delta}, "finish_reason": finish}]}
    return b"data: " + json.dumps(value, ensure_ascii=False).encode() + b"\n\n"


class StreamingTests(unittest.TestCase):
    def setUp(self):
        self.provider = DeepSeekJSONProvider("sk-fake-stream-test")

    def test_real_delta_iteration_not_buffered_and_transport_contract(self):
        raw = io.BytesIO(b": keepalive\n\n" + event("First ") + event("second") +
                         event(None, "stop") + b"data: {" + b'"choices": [], "usage": {}}\n\n' + b"data: [DONE]\n\n")
        with patch("urllib.request.urlopen", return_value=raw) as call:
            stream = self.provider.stream_text("system", "public facts")
            self.assertEqual(next(stream), "First ")
            self.assertLess(raw.tell(), len(raw.getvalue()))
            self.assertEqual(list(stream), ["second"])
        self.assertTrue(raw.closed)
        request = call.call_args.args[0]
        body = json.loads(request.data)
        self.assertTrue(body["stream"])
        self.assertEqual(body["response_format"], {"type": "text"})
        self.assertNotIn("sk-fake-stream-test", request.data.decode())

    def test_unicode_json_chunks_are_assembled_before_validation(self):
        fragments = ['{"name": "', '\u533b\u9662', '"}']
        raw = io.BytesIO(b"".join(event(part) for part in fragments) + event(None, "stop") + b"data: [DONE]\n\n")
        seen = []
        with patch("urllib.request.urlopen", return_value=raw):
            result = self.provider.generate_json_stream("Return JSON", "test", seen.append)
        self.assertEqual(result, {"name": "\u533b\u9662"})
        self.assertEqual(seen, fragments)

    def test_reasoning_is_not_exposed(self):
        raw = io.BytesIO(event(None, reasoning_content="private reasoning") + event("answer", "stop") + b"data: [DONE]\n\n")
        with patch("urllib.request.urlopen", return_value=raw):
            self.assertEqual(list(self.provider.stream_text("s", "u")), ["answer"])

    def test_incomplete_malformed_refused_and_tool_streams_fail_closed(self):
        for raw in [event("partial"), event("partial", "length") + b"data: [DONE]\n\n",
                    event("partial", "content_filter"), b"data: [DONE]\n\n",
                    b'data: {"error": "private provider error"}\n\n', b"data: not-json\n\n",
                    b"data: \xff\n\n", event(None, tool_calls=[{"name": "execute"}])]:
            with self.subTest(raw_length=len(raw)), patch("urllib.request.urlopen", return_value=io.BytesIO(raw)):
                with self.assertRaises(LLMResponseError) as caught:
                    list(self.provider.stream_text("s", "u"))
                self.assertNotIn("private provider error", str(caught.exception))

    def test_closing_stream_closes_transport(self):
        raw = io.BytesIO(event("part") + event("remaining", "stop") + b"data: [DONE]\n\n")
        with patch("urllib.request.urlopen", return_value=raw):
            stream = self.provider.stream_text("s", "u")
            next(stream)
            stream.close()
        self.assertTrue(raw.closed)

    def test_bounded_stream_does_not_hang_on_keepalive(self):
        with patch("urllib.request.urlopen", return_value=io.BytesIO(b": keepalive\n")), \
                patch("federated_agent.planner.time.monotonic", side_effect=[0, 121]):
            with self.assertRaises(LLMResponseError):
                list(self.provider.stream_text("s", "u"))

    def test_planner_does_not_compile_a_partial_plan(self):
        agent = build_agent()
        planner = LLMResearchPlanner(self.provider, retries=0)
        planner.on_delta = lambda text: None
        agent.executor.submit = Mock()
        with patch("urllib.request.urlopen", return_value=io.BytesIO(event('{"analysis_type":'))):
            response = agent.respond("Describe age", planner)
        self.assertEqual(response.status, "planner_error")
        self.assertIsNone(response.prepared_task)
        agent.executor.submit.assert_not_called()

    def test_connection_loss_is_sanitized(self):
        planner = Mock()
        planner.create_plan.side_effect = IncompleteRead(b"private bytes")
        response = build_agent().respond("test", planner)
        self.assertEqual(response.status, "planner_error")
        self.assertNotIn("private bytes", response.model_dump_json())


class TerminalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stream = io.StringIO()
        self.ui = TerminalUI(plain=True, console=Console(file=self.stream, width=72, force_terminal=False, highlight=False))
        self.agent = build_agent()
        self.plan = ResearchPlan(analysis_type="descriptive_statistics", variables=["age at diagnosis"],
                                 sites=["hospital 1", "hospital 2"], database_label="default", purpose="Describe age in two hospitals")

    def response(self, plan=None):
        planner = Mock()
        planner.create_plan.return_value = plan or self.plan
        return self.agent.respond("test", planner)

    def menu(self, keys, multiple=False, options=None, width=None):
        options = options or [("cancel", "Keep files without running"), ("confirm", "Submit and wait for results")]
        capture = io.StringIO()
        output = DummyOutput() if width is None else Vt100_Output(capture, lambda: Size(rows=24, columns=width),
                                                                 enable_cpr=False)
        with create_pipe_input() as pipe:
            pipe.send_text(keys)
            app = choice_application("Select", options, multiple=multiple, input=pipe, output=output)
            return app.run(), capture.getvalue()

    def test_arrows_enter_and_default_cancel_use_real_key_events(self):
        self.assertEqual(self.menu("\r")[0], "cancel")
        self.assertEqual(self.menu("\x1b[B\r")[0], "confirm")
        self.assertEqual(self.menu("\x1b[B\x1b[A\r")[0], "cancel")

    def test_multiselect_space_and_arrows(self):
        chosen, _ = self.menu(" \x1b[B \r", multiple=True,
                             options=[("a", "hospital 1"), ("b", "hospital 2"), ("c", "hospital 3")])
        self.assertEqual(chosen, ["a", "b"])
        self.assertEqual(self.menu("\r", multiple=True)[0], [])

    def test_escape_and_interrupt_do_not_confirm(self):
        self.assertIsNone(self.menu("\x1b")[0])
        with self.assertRaises(KeyboardInterrupt):
            self.menu("\x03")
        with self.assertRaises(EOFError):
            self.menu("\x04")

    def test_real_menu_draws_at_narrow_and_wide_widths(self):
        for width in (40, 80, 120):
            result, captured = self.menu("\x1b[B\r", width=width)
            self.assertEqual(result, "confirm")
            self.assertIn("Select", captured)
            self.assertIn("Keep files", captured)

    def test_terminal_controls_and_markup_are_never_executed(self):
        self.ui.say("\x1b[2J[bold]literal\x07\r\x9bescape")
        self.assertNotIn("\x1b", self.stream.getvalue())
        self.assertIn("[bold]literal", self.stream.getvalue())
        self.assertNotIn("\x9b", terminal_text("\x9bescape"))

    def test_report_values_and_connection_secrets_not_in_narration_facts(self):
        completed = AgentResponse(status="completed", message="done", execution=ExecutionRecord(
            task_id="1", status=ExecutionStatus.completed, result={"patient": "private-value"}))
        self.assertIsNone(reply_facts(completed))
        response = self.response()
        facts = json.dumps(reply_facts(response))
        self.assertNotIn("server_url", facts)
        self.assertNotIn("password", facts)
        self.assertNotIn("task_payload", facts)
        self.assertIn("hospital 1", facts)
        self.assertIn("runtime_unverified", facts)
        self.assertIn("development_allow_all", facts)

    def test_partial_explanation_cannot_change_task_or_trigger_execution(self):
        provider = DeepSeekJSONProvider("sk-fake-test")

        def broken(*args, **kwargs):
            yield "partial reply"
            raise LLMResponseError("private transport details")

        response = self.response()
        before = response.model_dump_json()
        with patch.object(provider, "stream_text", side_effect=broken):
            self.ui.present(response, LLMResearchPlanner(provider), self.root)
        self.assertEqual(response.model_dump_json(), before)
        self.assertFalse(json.loads((self.root / "stream.json").read_text())["complete"])
        self.assertIn("Explanation interrupted", self.stream.getvalue())
        self.assertNotIn("private transport details", self.stream.getvalue())

    def test_streamed_explanation_records_chunks_and_not_credentials(self):
        provider = DeepSeekJSONProvider("sk-fake-test")
        with patch.object(provider, "stream_text", return_value=iter(["first ", "second"])) as stream:
            self.ui.present(self.response(), LLMResearchPlanner(provider), self.root)
        record = json.loads((self.root / "stream.json").read_text())
        self.assertEqual(record["chunks"], 2)
        self.assertTrue(record["complete"])
        self.assertFalse(record["authoritative"])
        self.assertNotIn("sk-fake-test", str(stream.call_args))
        self.assertEqual((self.root / "assistant_reply.txt").read_text(), "first second")

    def test_no_stream_never_calls_explanation_provider(self):
        self.ui.stream = False
        provider = DeepSeekJSONProvider("sk-fake-test")
        with patch.object(provider, "stream_text") as stream:
            self.ui.present(self.response(), LLMResearchPlanner(provider), self.root)
        stream.assert_not_called()

    def test_explicit_catalog_command_never_calls_explanation_provider(self):
        provider = DeepSeekJSONProvider("sk-fake-test")
        with patch.object(provider, "stream_text") as stream:
            self.ui.present(self.agent.describe_algorithms(), LLMResearchPlanner(provider), self.root, narrate=False)
        stream.assert_not_called()

    def test_live_reflows_full_text_on_each_real_delta(self):
        self.ui.interactive = True
        with patch("federated_agent.terminal_ui.Live") as live:
            self.ui._stream_reply(iter(["\u533b", "\u9662 ", "1"]), self.root)
        self.assertEqual(live.return_value.update.call_count, 3)
        self.assertEqual(live.return_value.update.call_args.args[0].plain, "\u533b\u9662 1")
        self.assertTrue(live.return_value.update.call_args.kwargs["refresh"])
        live.return_value.stop.assert_called_once()

    def test_dumb_terminal_falls_back_without_cursor_ui(self):
        with patch.dict("os.environ", {"TERM": "dumb"}), patch("sys.stdin.isatty", return_value=True), \
                patch("sys.stdout.isatty", return_value=True):
            ui = TerminalUI(console=self.ui.console)
        self.assertFalse(ui.interactive)
        self.assertIsNone(ui.prompt_session)

    def test_response_layout_wraps_cjk_and_long_names(self):
        for width in (40, 80, 120):
            capture = io.StringIO()
            ui = TerminalUI(plain=True, stream=False, read=lambda _: "", console=Console(file=capture, width=width))
            prepared = self.response().prepared_task
            prepared.research_plan.purpose = "\u533b\u9662\u7814\u7a76" * 20
            ui.review(prepared, self.root)
            self.assertTrue(all(wcswidth(line) <= width for line in capture.getvalue().splitlines()))

    def test_menu_clarification_preserves_multiple_answers_and_manual_text(self):
        response = self.response(self.plan.model_copy(update={"sites": [], "variables": []}))
        self.ui.interactive = True
        self.ui.prompt = Mock(return_value="age at diagnosis")
        with patch("federated_agent.terminal_ui.choice_application") as application:
            application.return_value.run.return_value = ["0", "1"]
            reply = json.loads(self.ui.ask(response))
        values = {answer["field"]: answer for answer in reply["confirmed_choices"]}
        self.assertEqual(values["variables"]["reply"], "age at diagnosis")
        self.assertEqual(values["sites"]["selected_values"], ["hospital 1", "hospital 2"])

    def test_planning_callback_restored_after_interrupt(self):
        planner = LLMResearchPlanner(DeepSeekJSONProvider("sk-fake-test"))
        original = Mock()
        planner.on_delta = original
        with self.assertRaises(KeyboardInterrupt):
            with self.ui.planning(planner):
                self.assertIsNot(planner.on_delta, original)
                raise KeyboardInterrupt()
        self.assertIs(planner.on_delta, original)

    def test_allow_all_remains_extensible_and_mock_execution_still_requires_task_choice(self):
        prepared = self.response().prepared_task
        self.assertTrue(prepared.policy_decision.allowed)
        self.assertIn("mvp_allow_all_policy", prepared.policy_decision.reasons)
        args = SimpleNamespace(output=self.root, provider="demo", backend="mock", stream=False)
        planner = Mock()
        planner.create_plan.return_value = self.plan
        answers = iter(["Describe age", "", "/exit"])
        self.agent.executor.submit = Mock(side_effect=AssertionError("cancelled task cannot run"))
        with redirect_stdout(io.StringIO()):
            run_chat(self.agent, planner, args, read=lambda _: next(answers))
        self.agent.executor.submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
