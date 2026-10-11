"""Apply a researcher's menu selections to the study plan without a language model.

A selection picks one of the candidates that the validators offered, so applying
it is deterministic. Typed replies, and selections for fields not listed here,
still go to the planner. The revised plan passes the same validators either way.
"""

import json

from .capabilities import PlannerCapabilities
from .models import ResearchPlan

SINGLE_VALUE_FIELDS = {"database_label", "algorithm_id"}
LIST_FIELDS = {"sites", "variables", "stratification_variables"}


def selected_choices(reply: str) -> list[dict] | None:
    """The menu answers in ``reply`` when every answer is a selection; otherwise None."""
    try:
        answers = json.loads(reply)["confirmed_choices"]
    except (ValueError, TypeError, KeyError):
        return None
    if not isinstance(answers, list) or not answers:
        return None
    for answer in answers:
        values = answer.get("selected_values") if isinstance(answer, dict) else None
        if not isinstance(values, list) or not values or not all(isinstance(value, str) for value in values):
            return None
    return answers


def apply_choices(plan: ResearchPlan, reply: str) -> ResearchPlan | None:
    """Return ``plan`` with the selections in ``reply`` applied, or None when a model must interpret it."""
    answers = selected_choices(reply)
    if answers is None:
        return None
    changes = {}
    for answer in answers:
        field, values, requested = answer.get("field"), answer["selected_values"], answer.get("requested_value")
        if field in SINGLE_VALUE_FIELDS and len(values) == 1:
            changes[field] = values[0]
        elif field in LIST_FIELDS:
            current = list(changes.get(field, getattr(plan, field)))
            if requested:
                if requested not in current:
                    return None
                position = current.index(requested)
                current[position:position + 1] = values
            else:
                current.extend(values)
            changes[field] = list(dict.fromkeys(current))
        else:
            return None
    return plan.model_copy(update=changes)


class SelectionPlanner:
    """Replays a plan produced from menu selections; it never calls a model."""

    capabilities = PlannerCapabilities()

    def __init__(self, plan: ResearchPlan):
        self.plan = plan

    def create_plan(self, request, catalog):
        return self.plan.model_copy(deep=True)

    def revise_plan(self, reply, previous, catalog, questions=None):
        return self.plan.model_copy(deep=True)
