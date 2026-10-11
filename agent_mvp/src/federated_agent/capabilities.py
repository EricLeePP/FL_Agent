"""Capabilities that collaborators declare.

Services branch on these declarations, never on a collaborator's class, so a new
executor, policy, data checker, planner or model provider plugs in without
editing the services. A collaborator declares a ``capabilities`` attribute that
holds one of these dataclasses. Anything else, including a test double, receives
the conservative defaults below.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class ExecutorCapabilities:
    simulated: bool = False
    """Results are synthetic and nothing reaches a real backend."""
    trusted_worker: bool = False
    """A separate worker process submits; it loads its policy from ``policy_config``."""
    recovery: bool = False
    """Can reconnect to an already submitted remote task without submitting again."""
    closeable: bool = False
    """Holds a process or lock that ``close()`` releases."""


@dataclass(frozen=True)
class PolicyCapabilities:
    label: str = "custom checker"
    development_allow_all: bool = False
    """Allows every valid task. Development only; it never makes execution managed."""
    worker_config: Path | None = None
    """Policy file from which the trusted worker makes the same decisions."""
    worker_decides_runtime: bool = False
    """Runtime decisions need the authenticated backend identity, so only the worker makes them."""


@dataclass(frozen=True)
class DataCapabilities:
    verifies_availability: bool = True
    """False only for the placeholder that reports every check as unknown."""
    inventory_file: bool = False
    """Backed by an operator snapshot file (``path``, ``content``, ``load()``) captured with the other metadata."""


@dataclass(frozen=True)
class PlannerCapabilities:
    language_model: bool = False
    """Calls a language model through ``planner.provider``."""
    revises_plans: bool = False
    """Applies follow-up edits to the previous plan, answers clarifications and replies in ``response_language``."""
    uses_context: bool = False
    """Accepts public deployment and semantic context in ``planner.context``."""
    reports_progress: bool = False
    """Calls ``planner.on_delta`` while the model is producing its answer."""
    uses_lookups: bool = False
    """Accepts read-only lookups (lookups.ResearchLookups) in ``planner.lookups`` for each request."""
    complete_requests_only: bool = False
    """Deterministic demo: every request must be complete on its own."""


@dataclass(frozen=True)
class ProviderCapabilities:
    streams_json: bool = False
    """Implements ``generate_json_stream(system, user, on_delta)``."""
    narrates: bool = False
    """Implements ``stream_text(system, user)`` for the optional explanation of checked facts."""
    tool_lookups: bool = False
    """Accepts ``lookups=`` and ``max_turns=`` so the model can call read-only lookups before proposing."""


def capabilities_of(collaborator, kind: type[T]) -> T:
    """Return the declared capabilities of ``collaborator``, or conservative defaults."""
    declared = getattr(collaborator, "capabilities", None)
    return declared if isinstance(declared, kind) else kind()
