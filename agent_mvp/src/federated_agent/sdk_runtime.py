"""Agents SDK model/tool runtime; domain execution stays application-owned.

Only public research intent reaches the SDK. There are no execution, filesystem,
network-discovery or credential tools. Session persistence remains in sessions.py.
"""

import asyncio
import json
import time
from contextlib import closing
from importlib.metadata import version
from urllib.error import HTTPError
from urllib.parse import urlsplit

import httpx2 as httpx
from agents import (
    Agent, ModelSettings, OpenAIChatCompletionsModel, RunConfig, Runner, StopAtTools,
    function_tool, set_tracing_disabled, set_trace_processors,
)
from openai import APIStatusError, AsyncOpenAI

from .capabilities import ProviderCapabilities
from .errors import LLMResponseError
from .planner import OpenAICompatibleJSONProvider
from .prompts import LOOKUP_PROMPT, TOOL_DELIVERY_PROMPT, prompt_sha256


MAX_BYTES = 1_000_000
PROPOSAL_TOOL = "propose_research_plan"


def _lookup_tools(lookups):
    """Read-only lookups (lookups.ResearchLookups) the model may call before it proposes."""

    @function_tool(failure_error_function=None)
    def resolve_variable(name: str) -> str:
        """Map the researcher's wording for a variable to the semantic catalog; candidates come back instead of a guess.

        Args:
            name: The variable name in the researcher's own words.
        """
        return json.dumps(lookups.resolve_variable(name), ensure_ascii=True)

    @function_tool(failure_error_function=None)
    def resolve_site(name: str) -> str:
        """Map a hospital name or alias to a configured hospital and its dataset labels.

        Args:
            name: The hospital name in the researcher's own words.
        """
        return json.dumps(lookups.resolve_site(name), ensure_ascii=True)

    @function_tool(failure_error_function=None)
    def list_sites() -> str:
        """List configured hospitals with their aliases and dataset labels."""
        return json.dumps(lookups.list_sites(), ensure_ascii=True)

    @function_tool(failure_error_function=None)
    def list_algorithms() -> str:
        """List registered algorithms with their capabilities and limitations."""
        return json.dumps(lookups.list_algorithms(), ensure_ascii=True)

    return [resolve_variable, resolve_site, list_sites, list_algorithms]


class _CheckedStream(httpx.AsyncByteStream):
    """Keep the JSON runtime's fail-closed SSE boundary around the SDK's own parser."""

    def __init__(self, inner, expected_finish):
        self.inner = inner
        self.expected_finish = expected_finish
        self.total = 0
        self.buffer = b""
        self.data = []
        self.finish = None
        self.done = False

    def _line(self, raw):
        line = raw.rstrip(b"\r")
        if line.startswith(b"data:"):
            self.data.append(line[5:].lstrip(b" "))
        if line or not self.data:
            return
        event = b"\n".join(self.data)
        self.data = []
        if event == b"[DONE]":
            if self.finish != self.expected_finish or self.done:
                raise LLMResponseError("SDK model stream did not finish successfully")
            self.done = True
            return
        try:
            payload = json.loads(event.decode("utf-8"))
            if not isinstance(payload, dict) or "error" in payload or self.done:
                raise ValueError()
            choices = payload["choices"]
            if choices == []:
                return
            if len(choices) != 1 or choices[0].get("index", 0) != 0 or self.finish:
                raise ValueError()
            reason = choices[0].get("finish_reason")
            if reason is not None:
                if reason != self.expected_finish:
                    raise ValueError()
                self.finish = reason
        except (ValueError, KeyError, TypeError, IndexError, AttributeError):
            raise LLMResponseError("SDK model stream was invalid, refused or truncated") from None

    async def __aiter__(self):
        async for chunk in self.inner:
            self.total += len(chunk)
            if self.total > MAX_BYTES:
                raise LLMResponseError("SDK model response exceeded the size limit")
            self.buffer += chunk
            while b"\n" in self.buffer:
                line, self.buffer = self.buffer.split(b"\n", 1)
                self._line(line)
            yield chunk
        if not self.done:
            raise LLMResponseError("SDK model stream disconnected before completion")

    async def aclose(self):
        await self.inner.aclose()


class AgentsSDKProvider(OpenAICompatibleJSONProvider):
    """Compatibility boundary for the existing validated research planner/UI.

SDK Agent + Runner + function_tool are used for every proposal. Narration uses a
separate SDK Agent with no tools. No automatic fallback to the JSON runtime or to OpenAI models.
"""

    runtime_name = "openai-agents-sdk"
    capabilities = ProviderCapabilities(streams_json=True, narrates=True, tool_lookups=True)
    runtime_version = version("openai-agents")
    runtime_prompt_sha256 = prompt_sha256(TOOL_DELIVERY_PROMPT)

    def __init__(self, *args, transport=None, timeout_seconds=120, **kwargs):
        super().__init__(*args, **kwargs)
        parsed = urlsplit(self.endpoint)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username
                or parsed.password or parsed.query or parsed.fragment
                or not parsed.path.endswith("/chat/completions")):
            raise ValueError("SDK endpoint must be a full http(s) /chat/completions URL without credentials or query parameters.")
        if timeout_seconds <= 0:
            raise ValueError("SDK model timeout must be positive.")
        self.base_url = self.endpoint[:-len("chat/completions")]
        self.transport = transport
        self.timeout_seconds = timeout_seconds
        # The application never sends SDK traces, model input or tool arguments to OpenAI.
        set_tracing_disabled(True)
        set_trace_processors([])

    @classmethod
    def from_provider(cls, provider):
        instance = cls(endpoint=provider.endpoint, model=provider.model, api_key=provider.api_key,
                       extra_body=provider.extra_body, max_tokens=provider.max_tokens)
        instance.provider_name = provider.provider_name
        return instance

    async def _events(self, system, user, *, proposal, streaming, lookups=None, max_turns=1):
        accepted = []
        # With lookups the model may call read-only tools over several turns; the proposal ends the run.
        lookup_loop = proposal and lookups is not None
        expected_finish = "tool_calls" if proposal else "stop"

        async def prepare_request(request):
            if str(request.url) != self.endpoint:
                raise LLMResponseError("SDK attempted to use an unconfigured model endpoint")
            if not self.api_key:
                request.headers.pop("authorization", None)

        async def check_response(response):
            if response.status_code >= 400:
                return
            if streaming:
                if "text/event-stream" not in response.headers.get("content-type", ""):
                    raise LLMResponseError("SDK model returned a non-SSE response")
                response.stream = _CheckedStream(response.stream, expected_finish)
            else:
                body = await response.aread()
                if len(body) > MAX_BYTES:
                    raise LLMResponseError("SDK model response exceeded the size limit")
                try:
                    choices = response.json()["choices"]
                    if len(choices) != 1 or choices[0].get("finish_reason") != expected_finish:
                        raise ValueError()
                except (ValueError, KeyError, TypeError, IndexError):
                    raise LLMResponseError("SDK model response was invalid, refused or truncated") from None

        @function_tool(failure_error_function=None)
        def propose_research_plan(proposal_json: str) -> str:
            """Record a proposed research JSON object for subsequent application validation.

            Args:
                proposal_json: Serialized ResearchPlan or partial {changes: ...} object.
            """
            if accepted:
                raise LLMResponseError("SDK model proposed more than one plan")
            try:
                data = json.loads(proposal_json)
                if not isinstance(data, dict):
                    raise ValueError()
            except (ValueError, TypeError):
                raise LLMResponseError("SDK model proposed invalid JSON") from None
            accepted.append(data)
            return json.dumps(data, ensure_ascii=True)

        async with AsyncOpenAI(
            api_key=self.api_key or "local-no-key", base_url=self.base_url,
            organization="", project="", admin_api_key="",
            max_retries=0, timeout=self.timeout_seconds,
            http_client=httpx.AsyncClient(transport=self.transport, follow_redirects=False,
                timeout=self.timeout_seconds, event_hooks={"request": [prepare_request], "response": [check_response]}),
        ) as client:
            agent = Agent(
                name="ResearchIntentAgent" if proposal else "ResearchFeedbackAgent",
                instructions=system + (TOOL_DELIVERY_PROMPT + (LOOKUP_PROMPT if lookup_loop else "") if proposal else ""),
                model=OpenAIChatCompletionsModel(model=self.model, openai_client=client),
                tools=([propose_research_plan, *(_lookup_tools(lookups) if lookup_loop else [])] if proposal else []),
                tool_use_behavior=StopAtTools(stop_at_tool_names=[PROPOSAL_TOOL]) if lookup_loop else "stop_on_first_tool",
                reset_tool_choice=not lookup_loop,
                model_settings=ModelSettings(temperature=0, max_tokens=self.max_tokens,
                    tool_choice="required" if proposal else None, parallel_tool_calls=False,
                    extra_body=self.extra_body),
            )
            config = RunConfig(tracing_disabled=True, trace_include_sensitive_data=False)
            turns = max_turns if lookup_loop else 1
            if streaming:
                result = Runner.run_streamed(agent, user, max_turns=turns, run_config=config)
                try:
                    async for event in result.stream_events():
                        if event.type == "raw_response_event":
                            data = event.data
                            if data.type == ("response.function_call_arguments.delta" if proposal else "response.output_text.delta"):
                                yield "delta", data.delta
                finally:
                    result.cancel()
            else:
                result = await Runner.run(agent, user, max_turns=turns, run_config=config)
            if proposal:
                if len(accepted) != 1:
                    raise LLMResponseError("SDK agent did not call the proposal tool exactly once")
                yield "result", accepted[0]
            else:
                if not isinstance(result.final_output, str) or not result.final_output.strip():
                    raise LLMResponseError("SDK model returned empty feedback")
                yield "result", result.final_output

    def _run(self, system, user, *, proposal, streaming, lookups=None, max_turns=1):
        # The CLI is synchronous. Own and close one loop/client per run; no orphan worker threads.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise LLMResponseError("Use this synchronous planner outside a running asyncio loop")
        loop = asyncio.new_event_loop()
        events = self._events(system, user, proposal=proposal, streaming=streaming, lookups=lookups, max_turns=max_turns)
        started = time.monotonic()
        try:
            while True:
                remaining = self.timeout_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    raise LLMResponseError("SDK model run exceeded its time limit")
                try:
                    yield loop.run_until_complete(asyncio.wait_for(events.__anext__(), timeout=remaining))
                except StopAsyncIteration:
                    break
        except APIStatusError as exc:
            # Preserve the existing 401/402/429 user feedback without leaking provider bodies.
            raise HTTPError(self.endpoint, exc.status_code, "Model service request failed", None, None) from None
        except LLMResponseError:
            raise
        except Exception:
            raise LLMResponseError("SDK model run failed; no Vantage6 task was submitted by the planner") from None
        finally:
            try:
                loop.run_until_complete(events.aclose())
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                loop.close()

    def generate_json(self, system_prompt, user_prompt, *, lookups=None, max_turns=1):
        with closing(self._run(system_prompt, user_prompt, proposal=True, streaming=False,
                               lookups=lookups, max_turns=max_turns)) as events:
            for kind, value in events:
                if kind == "result":
                    return value
        raise LLMResponseError("SDK agent produced no plan")

    def generate_json_stream(self, system_prompt, user_prompt, on_delta, *, lookups=None, max_turns=1):
        with closing(self._run(system_prompt, user_prompt, proposal=True, streaming=True,
                               lookups=lookups, max_turns=max_turns)) as events:
            for kind, value in events:
                if kind == "delta":
                    on_delta(value)
                elif kind == "result":
                    return value
        raise LLMResponseError("SDK agent produced no plan")

    def stream_text(self, system_prompt, user_prompt, *, json_mode=False):
        if json_mode:
            raise LLMResponseError("Use the SDK proposal tool for structured plans")
        with closing(self._run(system_prompt, user_prompt, proposal=False, streaming=True)) as events:
            for kind, value in events:
                if kind == "delta":
                    yield value
