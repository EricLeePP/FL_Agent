"""Run a confirmed task and persist what happened: the one path every transport uses."""

from datetime import datetime, timezone
from pathlib import Path

from ..runs.executor import SandboxVantage6Executor
from ..storage import write_json
from .reporting import generate_report

CONSENT_FILE = "researcher_confirmation.json"


def configure_executor(agent, backend, output, *, confirmed=False, timeout_seconds=600, progress=print):
    """Install the executor for ``backend``. The mock executor needs no per-task setup.

    The agent binds the policy configuration just before it submits or recovers.
    """
    if backend == "vantage6":
        agent.executor = SandboxVantage6Executor(Path(output) / "backend", confirmed=confirmed,
                                                 timeout_seconds=timeout_seconds, progress=progress)


def record_consent(output, prepared, revision):
    """Persist the researcher's confirmation before anything is submitted; it is never replayed."""
    write_json(Path(output) / CONSENT_FILE, {"confirmation_id": prepared.confirmation_id,
                                             "confirmed_at": datetime.now(timezone.utc).isoformat(),
                                             "study_revision": revision})


def finish_execution(agent, prepared, execution, output, backend, *, persist, emit=print):
    """Persist an execution outcome and, when it completed, build the local report.

    ``persist`` is called with every state the researcher may later need: first
    the outcome itself (marked report pending when it completed), then the outcome
    with its report. A report failure never hides a completed computation.
    """
    completed = execution.status == "completed"
    if completed:
        execution = execution.model_copy(update={"report_status": "pending"})
    persist(execution)
    if completed:
        execution = generate_report(agent, prepared, execution, output, backend, emit=emit)
        persist(execution)
    return execution
