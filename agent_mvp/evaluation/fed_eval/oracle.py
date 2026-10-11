"""A provider that returns each case's gold model outputs, to validate the labels.

Running a case with the oracle answers the question "what does Fed Agent do when the
planner is perfect?". Every expected outcome in the dataset must match that run,
so a label can never demand something no planner could achieve.
"""

import copy

from federated_agent.capabilities import ProviderCapabilities
from federated_agent.errors import LLMResponseError
from federated_agent.prompts import UPDATE_PROMPT


class OracleProvider:
    provider_name = "oracle"
    capabilities = ProviderCapabilities()
    model = "oracle"
    seed = None

    def __init__(self, outputs=()):
        self.queue = [copy.deepcopy(output) for output in outputs]
        self.calls = []

    def drain(self):
        calls, self.calls = self.calls, []
        return calls

    def generate_json(self, system_prompt, user_prompt):
        kind = "update" if UPDATE_PROMPT in system_prompt else "plan"
        if not self.queue:
            self.calls.append({"kind": kind, "error": "no_gold_output"})
            raise LLMResponseError("The case has no gold output for this model call")
        output = self.queue.pop(0)
        self.calls.append({"kind": kind, "oracle": True, "json_valid": True, "schema_valid": True})
        return output


def gold_outputs(case):
    """Gold model outputs in the order the planner will be called."""
    return [turn["gold"] for turn in case["turns"] if "gold" in turn]
