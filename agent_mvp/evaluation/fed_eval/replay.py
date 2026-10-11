"""Counterfactual replay: feed recorded model outputs through a modified pipeline.

No model is called. Each recorded call is returned in order (failed calls raise the
same error), so a product change that acts after the planner can be measured on
existing runs. Multi-turn replays are approximate: a later turn's recorded output
was produced from the original earlier turn.
"""

import json
import re
from contextlib import contextmanager
from unittest.mock import patch

from federated_agent.capabilities import ProviderCapabilities
from federated_agent.errors import LLMResponseError
from federated_agent.requirements import protect_requirements

STOPWORDS = {"with", "from", "that", "this", "which", "only", "using", "data", "dataset", "analysis", "requested", "request",
             "algorithm", "supported", "support", "statistics", "descriptive", "hospital", "hospitals", "patients", "patient"}


class ReplayProvider:
    provider_name = "replay"
    capabilities = ProviderCapabilities()
    seed = None

    def __init__(self, record):
        self.model = record.get("model", "replay")
        self.queue = [call for turn in record["turns"] for call in turn["model_calls"]]
        self.calls = []

    def drain(self):
        calls, self.calls = self.calls, []
        return calls

    def generate_json(self, system_prompt, user_prompt):
        if not self.queue:
            raise LLMResponseError("No recorded output left for this call")
        call = self.queue.pop(0)
        self.calls.append(call)
        if call.get("error"):
            raise LLMResponseError(call["error"])
        return json.loads(call["content"])


def grounded(requirement, text):
    """True when a requirement shares a content word (or a two-character CJK run) with the researcher's text."""
    text = text.lower()
    words = {w for w in re.findall(r"[a-z][a-z0-9_-]{3,}", requirement.lower()) if w not in STOPWORDS}
    if any(word in text for word in words):
        return True
    chinese = re.findall(r"[一-鿿]+", requirement)
    return any(run[i:i + 2] in text for run in chinese for i in range(len(run) - 1))


def grounded_protection(request, plan, previous=None):
    """Drop model-written requirements that do not come from the researcher's words, then protect as usual."""
    text = request + " " + (previous.purpose if previous else "")
    kept = [item for item in plan.unsupported_requirements if grounded(item, text)]
    return protect_requirements(request, plan.model_copy(update={"unsupported_requirements": kept}), previous)


@contextmanager
def grounding_filter():
    with patch("federated_agent.service.protect_requirements", grounded_protection):
        yield
