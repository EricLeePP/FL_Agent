"""An OpenAI-compatible model client for evaluation runs.

It sends Fed Agent's own prompts, constrains the output to the plan's JSON schema
(the configuration recommended for local models) and records every call so that
model-level diagnostics can be scored next to the pipeline outcome.
"""

import json
import time
import urllib.error
import urllib.request

from pydantic import ValidationError

from federated_agent.capabilities import ProviderCapabilities
from federated_agent.errors import LLMResponseError
from federated_agent.models import ResearchPlan
from federated_agent.planner import PlanUpdate
from federated_agent.prompts import UPDATE_PROMPT

# The order the planning prompt lists the fields in; free text (purpose) comes last.
PLAN_FIELD_ORDER = ("request_type", "analysis_type", "algorithm_id", "variables", "stratification_variables", "sites",
                    "database_label", "parameters", "unsupported_requirements", "purpose")


def strict_schema(model, order=None):
    """Every property required, in a fixed order, as OpenAI strict mode demands.

    Grammar-constrained decoders (Ollama, llama.cpp) emit keys in schema order and
    let a model skip optional keys for good; with Pydantic's default schema small
    models silently dropped "variables". Required keys force an explicit [] or null.
    """
    schema = model.model_json_schema()
    properties = schema["properties"]
    names = [name for name in (order or properties) if name in properties]
    schema["properties"] = {name: properties[name] for name in names}
    schema["required"] = names
    schema["additionalProperties"] = False
    return schema


PLAN_FORMAT = {"name": "research_plan", "strict": True, "schema": strict_schema(ResearchPlan, PLAN_FIELD_ORDER)}
UPDATE_FORMAT = {"name": "plan_update", "strict": True, "schema": strict_schema(PlanUpdate)}
# No tokenizer averages six characters per token on this prompt; fewer tokens means the server cut it.
MAX_CHARS_PER_TOKEN = 6


class EvalProvider:
    provider_name = "evaluation"
    capabilities = ProviderCapabilities()

    def __init__(self, endpoint, model, *, options=None, structured_output="json_schema", temperature=0.0,
                 seed=None, max_tokens=1200, timeout_seconds=1800):
        if structured_output not in {"json_schema", "json_object", "none"}:
            raise ValueError("structured_output must be json_schema, json_object or none")
        self.endpoint = endpoint
        self.model = model
        self.options = dict(options or {})
        self.structured_output = structured_output
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.calls = []

    def drain(self):
        calls, self.calls = self.calls, []
        return calls

    def _body(self, system_prompt, user_prompt, update):
        body = {**self.options, "model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens,
                "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]}
        if self.structured_output == "json_schema":
            body["response_format"] = {"type": "json_schema", "json_schema": UPDATE_FORMAT if update else PLAN_FORMAT}
        elif self.structured_output == "json_object":
            body["response_format"] = {"type": "json_object"}
        if self.seed is not None:
            body["seed"] = self.seed
        return body

    def generate_json(self, system_prompt, user_prompt):
        update = UPDATE_PROMPT in system_prompt
        record = {"kind": "update" if update else "plan", "prompt_chars": len(system_prompt) + len(user_prompt)}
        self.calls.append(record)
        request = urllib.request.Request(self.endpoint, data=json.dumps(self._body(system_prompt, user_prompt, update)).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            record.update(seconds=round(time.monotonic() - started, 3), error=f"http_{exc.code}")
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            record.update(seconds=round(time.monotonic() - started, 3), error=type(exc).__name__)
            raise
        record["seconds"] = round(time.monotonic() - started, 3)
        try:
            choice = payload["choices"][0]
            message = choice["message"]
            content = message.get("content") or ""
        except (KeyError, IndexError, TypeError):
            record["error"] = "malformed_response"
            raise LLMResponseError("Model response has no message") from None
        usage = payload.get("usage") or {}
        record.update(prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
                      finish_reason=choice.get("finish_reason"), reasoning_chars=len(message.get("reasoning") or ""),
                      content=content)  # in full, so that replays can parse it
        if record["prompt_tokens"] and record["prompt_tokens"] * MAX_CHARS_PER_TOKEN < record["prompt_chars"]:
            record["error"] = "context_truncated"
            raise LLMResponseError("The model server truncated the prompt; raise its context window")
        if record["finish_reason"] == "length":
            record["error"] = "output_truncated"
            raise LLMResponseError("Model output reached the token limit")
        try:
            data = json.loads(content)
        except ValueError:
            record.update(json_valid=False, error="invalid_json")
            raise LLMResponseError("Model returned invalid JSON") from None
        record["json_valid"] = True
        try:
            (PlanUpdate if update else ResearchPlan).model_validate(data)
            record["schema_valid"] = True
        except ValidationError:
            record["schema_valid"] = False
        return data


def server_version(endpoint):
    """The Ollama version behind an OpenAI-compatible endpoint, when it is Ollama."""
    base = endpoint.split("/v1/")[0]
    try:
        with urllib.request.urlopen(base + "/api/version", timeout=10) as response:
            return json.loads(response.read()).get("version")
    except (OSError, ValueError):
        return None


def unload(endpoint, model):
    """Ask an Ollama server to free the model's memory; ignored by other servers."""
    base = endpoint.split("/v1/")[0]
    request = urllib.request.Request(base + "/api/generate", data=json.dumps({"model": model, "keep_alive": 0}).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=60):
            pass
    except OSError:
        pass
