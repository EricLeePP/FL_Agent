"""Stable policy extension point; the MVP allows every valid task."""

from typing import Protocol
from pathlib import Path
from typing import Literal
import os

from pydantic import Field

from .capabilities import PolicyCapabilities
from .models import PolicyDecision, Vantage6TaskSpec, StrictModel
from .errors import PolicyDeniedError, PreflightError
from .trusted_http import fetch, strict_json
from .resources import DEFAULT_POLICY

DEFAULT_POLICY_CONFIG = DEFAULT_POLICY


class PolicyChecker(Protocol):
    def evaluate(self, task_spec: Vantage6TaskSpec) -> PolicyDecision:
        ...


class AllowAllPolicyChecker:
    capabilities = PolicyCapabilities(label="allow-all (development)", development_allow_all=True,
                                      worker_config=DEFAULT_POLICY_CONFIG)

    def evaluate(self, task_spec: Vantage6TaskSpec) -> PolicyDecision:
        return PolicyDecision(
            allowed=True,
            reasons=["mvp_allow_all_policy"],
            obligations=[],
        )


class PolicyConfiguration(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    mode: Literal["allow_all", "opa"]
    version: str = Field(min_length=1)
    opa_url: str | None = None
    token_env: str | None = None
    timeout_seconds: float = Field(default=5, gt=0, le=30)


class FilePolicyChecker:
    """Operator-owned PDP configuration, reread at every enforcement point."""

    def __init__(self, path):
        self.path = Path(path).resolve()

    @property
    def capabilities(self):
        return PolicyCapabilities(worker_config=self.path, worker_decides_runtime=True)

    def evaluate_action(self, task_spec, *, action, task_id=None, subject=None):
        config = PolicyConfiguration.model_validate(strict_json(self.path.read_bytes()))
        if config.mode == "allow_all":
            return PolicyDecision(allowed=True, reasons=["development_allow_all"], policy_version=config.version)
        if not config.opa_url:
            raise ValueError("OPA mode requires an endpoint")
        token = os.environ.get(config.token_env) if config.token_env else None
        if config.token_env and not token:
            raise ValueError("Configured policy credential is missing")
        request = {"schema_version": "1.0", "action": action, "task_id": task_id,
                   "subject": subject or {"authenticated": False},
                   "task": task_spec.model_dump(mode="json")}
        raw = strict_json(fetch(config.opa_url, body={"input": request}, token=token,
                                timeout=config.timeout_seconds, limit=65536))
        decision = PolicyDecision.model_validate(raw["result"])
        decision.policy_version = decision.policy_version or config.version
        return decision

    def evaluate(self, task_spec):
        return self.evaluate_action(task_spec, action="prepare")


def enforce_policy(checker, task, *, action, task_id=None, subject=None, audit=None):
    """PEP: errors, invalid decisions and unimplemented obligations fail closed."""
    if action not in {"prepare", "submit", "read_result", "release_result"}:
        raise PreflightError("Unknown policy action")
    try:
        if callable(getattr(type(checker), "evaluate_action", None)):
            decision = checker.evaluate_action(task, action=action, task_id=task_id, subject=subject)
        else:
            decision = checker.evaluate(task)
        decision = PolicyDecision.model_validate(decision)
    except Exception:
        if audit:
            audit({"action": action, "task_id": task_id, "allowed": False, "policy_error": True})
        raise PreflightError("Policy decision service unavailable or invalid") from None
    if audit:
        audit({"action": action, "task_id": task_id, "allowed": decision.allowed,
               "policy_version": decision.policy_version, "decision_id": decision.decision_id,
               "unhandled_obligations": bool(decision.obligations)})
    if not decision.allowed or decision.obligations:
        # No obligation may be silently ignored. Add an explicit handler before
        # accepting future redaction, approval, budget or differential-privacy duties.
        raise PolicyDeniedError("Policy denied the operation or requires an unsupported obligation")
    return decision
