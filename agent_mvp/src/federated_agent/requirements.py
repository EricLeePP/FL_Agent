"""Conservative protection for conditions the task compiler cannot represent.

These rules catch common English/Chinese cohort, time and grouping expressions;
they are not a complete natural-language verifier. The original request remains
visible at task review. Existing unmet requirements cannot be erased by an LLM.
"""

import re


CONDITION_PATTERNS = [
    r"\b(?:younger|older|less|greater|more)\s+than\s+\d+(?:\.\d+)?",
    r"\b(?:under|over|below|above|at least|at most)\s+\d+(?:\.\d+)?",
    r"(?:<=|>=|(?<![<>])[<>]|\u2264|\u2265)\s*\d+(?:\.\d+)?",
    r"\bbetween\s+\d+(?:\.\d+)?\s+and\s+\d+(?:\.\d+)?",
    r"\b(?:from|since|before|after|during)\s+(?:19|20)\d{2}\b",
    r"\b(?:last|past)\s+\d+\s+(?:days?|months?|years?)\b",
    r"\b(?:only|just)\s+(?:for\s+)?(?:the\s+)?(?:patients?|participants?|records?|women|men|males?|females?)\b[^.;\n]*",
    r"\b(?:patients?|participants?)\s+(?:with|without|who|aged)\b[^.;\n]*",
    r"\b(?:male|female)\s+patients?\b",
    r"\b(?:where|excluding|stratify(?:ing|ied)?\s+by|group(?:ed)?\s+by|filter(?:ed)?\s+(?:by|to))\s+[^.;\n]+",
    r"(?:\u4ec5\u9650|\u53ea\u5305\u542b|\u6392\u9664|\u7b5b\u9009|\u5206\u5c42)[^\u3002\uff1b\n]*",
    r"(?:\u5c0f\u4e8e|\u5927\u4e8e|\u4e0d\u8d85\u8fc7|\u81f3\u5c11)\s*\d+",
    r"\d+\s*\u5c81(?:\u4ee5\u4e0b|\u4ee5\u4e0a)",
]
CONDITIONS = re.compile("|".join(f"(?:{pattern})" for pattern in CONDITION_PATTERNS), re.IGNORECASE)


def protect_requirements(request, plan, previous=None):
    if plan.request_type != "analysis":
        return plan
    retained = list(previous.unsupported_requirements) if previous else []
    requirements = list(dict.fromkeys([*retained, *plan.unsupported_requirements]))
    if not requirements:
        requirements = list(dict.fromkeys(match.group().strip() for match in CONDITIONS.finditer(request)))
    return plan.model_copy(update={"unsupported_requirements": requirements})
