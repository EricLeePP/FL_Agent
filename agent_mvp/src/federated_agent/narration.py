"""Optional model explanation of a checked response. It is never authoritative.

Only allowlisted public facts reach the model. The explanation cannot change the
checked response, submit a task or replace the local task review.
"""

import json

from .capabilities import ProviderCapabilities, capabilities_of
from .prompts import NARRATION_PROMPT


def reply_facts(response):
    """Allowlisted public preparation facts only, never a serialized execution."""
    if response.status not in {"ready", "needs_clarification", "unsupported", "invalid_request", "algorithms"}:
        return None
    facts = {"status": response.status, "submission_state": response.submission_state,
             "message": response.message,
             "issues": [item.model_dump(include={"code", "message", "field", "requested_value", "next_steps"})
                        for item in response.issues if item.severity != "warning"],
             "warnings": [item.model_dump(include={"code", "message"})
                          for item in response.issues if item.severity == "warning"],
             "algorithms": [item.model_dump(include={"algorithm_id", "display_name", "version", "capabilities", "limitations"})
                            for item in response.suggested_algorithms]}
    if response.clarification:
        facts["questions"] = [question.model_dump() for question in response.clarification.questions]
    if response.prepared_task:
        prepared = response.prepared_task
        facts["policy_mode"] = ("development_allow_all" if "mvp_allow_all_policy" in prepared.policy_decision.reasons
                                else "custom_checker")
        facts["prepared"] = {"algorithm": prepared.task_spec.algorithm_id,
            "variables": [variable.canonical_name for variable in prepared.resolved_plan.variables],
            "sites": [site.canonical_name for site in prepared.resolved_plan.sites],
            "database": prepared.task_spec.databases[0].label}
    return facts


def explanation_prompt(language):
    return NARRATION_PROMPT + f"\nResponse language: {'Chinese' if language == 'zh' else 'English'}."


def explain(provider, response, language):
    """Return a stream of explanation text, or None when there is nothing to explain or no narrator."""
    facts = reply_facts(response)
    if facts is None or not capabilities_of(provider, ProviderCapabilities).narrates:
        return None
    return provider.stream_text(explanation_prompt(language), json.dumps(facts, ensure_ascii=True))
