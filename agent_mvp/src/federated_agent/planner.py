"""Planner interfaces for a future local LLM and an offline demo planner."""

import json
import re
import time
import urllib.request
from typing import Any, Callable, Protocol

from pydantic import ValidationError

from .algorithm_catalog import AlgorithmCatalog
from .capabilities import PlannerCapabilities, ProviderCapabilities, capabilities_of
from .conversation import response_language
from .errors import LLMResponseError
from .models import ResearchPlan, StrictModel
from .prompts import PLAN_PROMPT, UPDATE_PROMPT


class PlanUpdate(StrictModel):
    changes: dict[str, Any]


class ResearchPlanner(Protocol):
    def create_plan(self, request: str, catalog: AlgorithmCatalog) -> ResearchPlan:
        ...


class JSONProvider(Protocol):
    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        ...


class OpenAICompatibleJSONProvider:
    """Small HTTP adapter for local servers exposing a chat-completions endpoint."""

    provider_name = "OpenAI-compatible"
    capabilities = ProviderCapabilities(streams_json=True, narrates=True)

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: str | None = None,
        extra_body: dict[str, Any] | None = None,
        max_tokens: int = 1200,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.extra_body = extra_body or {}
        self.max_tokens = max_tokens

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        request_body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
            **self.extra_body,
        }
        body = json.dumps(request_body).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                **(
                    {"Authorization": f"Bearer {self.api_key}"}
                    if self.api_key
                    else {}
                ),
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
        try:
            choice = payload["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError("LLM response has no chat message content") from exc
        if choice.get("finish_reason") == "length":
            raise LLMResponseError("LLM JSON output was truncated by the token limit")
        if not content or not content.strip():
            raise LLMResponseError("LLM returned empty JSON content")
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMResponseError("LLM returned invalid JSON content") from exc

    def stream_text(self, system_prompt: str, user_prompt: str, *, json_mode=False):
        """Yield real SSE content deltas, never hidden reasoning or tool calls.

        A closed connection without a successful finish and DONE is incomplete.
        No automatic non-streaming retry is made after partial output.
        """
        body = {
            **self.extra_body,
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": user_prompt}],
            "temperature": 0, "max_tokens": self.max_tokens, "stream": True,
            "response_format": {"type": "json_object" if json_mode else "text"},
        }
        request = urllib.request.Request(self.endpoint, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream",
                     **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})}, method="POST")
        started = time.monotonic()
        total = 0
        data = []
        finish = None
        with urllib.request.urlopen(request, timeout=120) as response:
            for raw in response:
                total += len(raw)
                if total > 1_000_000 or time.monotonic() - started > 120:
                    raise LLMResponseError("LLM stream exceeded the response limit")
                try:
                    line = raw.decode("utf-8").rstrip("\r\n")
                except UnicodeError:
                    raise LLMResponseError("LLM stream contains invalid UTF-8") from None
                if line.startswith("data:"):
                    data.append(line[5:].lstrip(" "))
                if line or not data:
                    continue
                event = "\n".join(data)
                data = []
                if event == "[DONE]":
                    if finish != "stop":
                        raise LLMResponseError("LLM stream did not finish successfully")
                    return
                try:
                    payload = json.loads(event)
                    if not isinstance(payload, dict) or "error" in payload:
                        raise ValueError()
                    choices = payload["choices"]
                    if choices == []:
                        continue  # Optional usage-only event.
                    choice = choices[0]
                    if choice.get("index", 0) != 0:
                        raise ValueError()
                    delta = choice["delta"]
                    if delta.get("tool_calls") or delta.get("function_call"):
                        raise ValueError()
                    content = delta.get("content") or ""
                    if not isinstance(content, str) or finish is not None:
                        raise ValueError()
                    reason = choice.get("finish_reason")
                    if reason is not None:
                        if reason != "stop":
                            raise LLMResponseError("LLM stream was truncated or refused")
                        finish = reason
                except (ValueError, KeyError, IndexError, TypeError, AttributeError):
                    raise LLMResponseError("LLM returned an invalid stream event") from None
                if content:
                    yield content
        raise LLMResponseError("LLM stream disconnected before completion")

    def generate_json_stream(self, system_prompt, user_prompt, on_delta):
        fragments = []
        for delta in self.stream_text(system_prompt, user_prompt, json_mode=True):
            fragments.append(delta)
            on_delta(delta)
        try:
            return json.loads("".join(fragments))
        except json.JSONDecodeError:
            raise LLMResponseError("LLM returned invalid streamed JSON") from None


class DeepSeekJSONProvider(OpenAICompatibleJSONProvider):
    """Remote DeepSeek JSON-output provider; credentials are transport-only."""

    provider_name = "DeepSeek"

    DEFAULT_ENDPOINT = "https://api.deepseek.com/chat/completions"
    DEFAULT_MODEL = "deepseek-v4-flash"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        endpoint: str = DEFAULT_ENDPOINT,
    ) -> None:
        if not api_key:
            raise ValueError("A DeepSeek API key is required")
        super().__init__(
            endpoint=endpoint,
            model=model,
            api_key=api_key,
            extra_body={"thinking": {"type": "disabled"}},
            max_tokens=1200,
        )


class LLMResearchPlanner:
    SYSTEM_PROMPT = PLAN_PROMPT
    UPDATE_PROMPT = UPDATE_PROMPT

    def __init__(self, provider: JSONProvider, retries: int = 2, *, tool_lookups: bool = False, max_turns: int = 6) -> None:
        self.provider = provider
        self.retries = retries
        self.tool_lookups = tool_lookups
        self.max_turns = max_turns
        self.context = {}
        self.lookups = None
        self.response_language: str | None = None
        self.on_delta: Callable[[str], None] | None = None

    @property
    def capabilities(self):
        return PlannerCapabilities(language_model=True, revises_plans=True, uses_context=True,
                                   reports_progress=True, uses_lookups=self.tool_lookups)

    def create_plan(self, request: str, catalog: AlgorithmCatalog) -> ResearchPlan:
        return self._generate(request, catalog)

    def revise_plan(self, reply, previous, catalog, questions=None):
        request = json.dumps({"previous_plan": previous.model_dump(),
                              "questions": questions or [], "researcher_reply": reply}, ensure_ascii=True)
        plan = self._generate(request, catalog, previous=previous)
        if plan.request_type == "analysis" and plan.purpose == previous.purpose:
            plan.purpose = previous.purpose + "\nResearcher follow-up: " + reply
        return plan

    def _generate(self, request, catalog, previous=None):
        prompt = request
        system_prompt = self.SYSTEM_PROMPT + "\nCATALOG_JSON:\n" + json.dumps(
            catalog.knowledge(), ensure_ascii=True,
        )
        system_prompt += "\nPUBLIC_CONTEXT_JSON:\n" + json.dumps(self.context, ensure_ascii=True)
        if previous is not None:
            system_prompt += self.UPDATE_PROMPT
        language = self.response_language or response_language(request)
        system_prompt += (
            "\nWrite diagnostic text in Chinese. " if language == "zh" else
            "\nWrite diagnostic text in English. "
        ) + "Keep requested names and purpose faithful to the original request; catalog language does not set the response language."
        provider = capabilities_of(self.provider, ProviderCapabilities)
        # Lookups are bound to the request's metadata snapshot by the agent (planner.lookups).
        options = ({"lookups": self.lookups, "max_turns": self.max_turns}
                   if self.tool_lookups and self.lookups is not None and provider.tool_lookups else {})
        last_error: Exception | None = None
        for _ in range(self.retries + 1):
            try:
                if self.on_delta is not None and provider.streams_json:
                    data = self.provider.generate_json_stream(system_prompt, prompt, self.on_delta, **options)
                else:
                    data = self.provider.generate_json(system_prompt, prompt, **options)
                if previous is not None:
                    update = PlanUpdate.model_validate(data)
                    if not update.changes:
                        raise LLMResponseError("No explicit study change was provided; please clarify the follow-up.")
                    return ResearchPlan.model_validate(previous.model_dump() | update.changes)
                return ResearchPlan.model_validate(data)
            except (ValidationError, LLMResponseError) as exc:
                last_error = exc
                prompt = (
                    f"Original request: {request}\n"
                    f"Your previous JSON failed validation: {exc}. Return corrected JSON."
                )
        assert last_error is not None
        raise last_error


class DemoResearchPlanner:
    """Offline smoke-test planner; production should use LLMResearchPlanner."""

    capabilities = PlannerCapabilities(complete_requests_only=True)

    VARIABLE_PHRASES = {
        "age at initial diagnosis": "age at initial diagnosis",
        "age at diagnosis": "age at diagnosis",
        "survival status": "survival status",
        "survival time": "survival time",
        "gender": "gender",
    }

    def create_plan(self, request: str, catalog: AlgorithmCatalog) -> ResearchPlan:
        lowered = request.lower()
        if any(phrase in lowered for phrase in ("list algorithms", "available algorithms", "what algorithms")):
            return ResearchPlan(request_type="catalog_query", analysis_type="catalog_query", purpose=request)
        variables = [
            canonical
            for phrase, canonical in self.VARIABLE_PHRASES.items()
            if phrase in lowered
        ]
        variables = list(dict.fromkeys(variables))
        sites = re.findall(
            r"\b(?:hospital|organisation|organization)\s+[a-z0-9][a-z0-9_-]*",
            lowered,
        )
        matches = [
            item for item in catalog.algorithms if item.status == "enabled"
            and any(keyword.lower() in lowered for keyword in item.request_keywords)
        ]
        # This planner is deliberately narrow and is not a substitute for an LLM.
        unsupported = []
        if re.search(r"\b(cox|kaplan|regression|train|filter|under|over|older|younger|only|between|exclude|before|after)\b", lowered):
            unsupported.append(f"Offline demo cannot safely interpret this condition or analysis: {request}")
        database = (re.search(r"\b(?:using|use|dataset|database)\s+(?:(?:dataset|database)\s+)?(?:label\s+)?[\"']?([a-z0-9_-]+)\b", lowered)
                    or re.search(r"\b(synthetic_[a-z0-9_]+)\b", lowered))
        return ResearchPlan(
            analysis_type=matches[0].analysis_types[0] if len(matches) == 1 else "unsupported",
            algorithm_id=matches[0].algorithm_id if len(matches) == 1 else None,
            variables=variables,
            sites=sites,
            database_label=database.group(1) if database else None,
            unsupported_requirements=unsupported,
            purpose=request,
        )
