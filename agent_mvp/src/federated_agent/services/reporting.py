"""Local result artifacts and reports. Nothing here submits a task or calls a model."""

from pathlib import Path

from ..confirmation import task_confirmation_id
from ..errors import PolicyDeniedError
from ..feedback import issue
from ..models import AgentResponse, PreparedTask
from ..policy import enforce_policy
from ..result_adapters import result_adapter
from ..runs.store import RunStore
from ..sessions import SessionError

REPORT_FAILURES = (OSError, ValueError, TypeError, KeyError, ImportError)


def write_execution_artifacts(agent, prepared, response, output, backend, emit=print):
    output = Path(output)
    record = response.execution
    (output / ("mock_result.json" if backend == "mock" else "execution.json")).write_text(
        record.model_dump_json(indent=2), encoding="utf-8")
    emit(f"{'Mock' if backend == 'mock' else 'Real Vantage6'} execution completed: {record.task_id}")
    adapter = result_adapter(prepared.task_spec.result_adapter)
    report = adapter.write_report(record.result, output, semantic_catalog=agent.semantic_catalog,
                                  chart=backend == "vantage6", emit=emit, context={
        "Task ID": record.task_id, "Backend": backend,
        "Algorithm": f"{prepared.task_spec.algorithm_id} {prepared.task_spec.algorithm_version}",
        "Hospitals": ", ".join(site.canonical_name for site in prepared.resolved_plan.sites),
        "Database": prepared.task_spec.databases[0].label,
    })
    if report is not None:
        emit(f"Report: {Path(report).resolve()}")
    return adapter


def generate_report(agent, prepared, response, output, backend, emit=print) -> AgentResponse:
    """Write the result artifacts of a completed execution; a failure keeps the computation."""
    try:
        adapter = write_execution_artifacts(agent, prepared, response, output, backend, emit=emit)
    except REPORT_FAILURES:
        problem = issue("report_failed")
        return response.model_copy(update={"status": "report_failed", "message": problem.message,
                                           "report_status": "failed", "issues": [problem]})
    return response.model_copy(update={
        "status": "completed",
        "message": "Computation and local report generation completed." if backend == "vantage6" else
                   "Mock execution and local report generation completed; no real Vantage6 task was submitted.",
        "report_status": "completed" if adapter.reports else "not_requested",
        "issues": [item for item in response.issues if item.code != "report_failed"]})


def saved_result(agent, store, name, backend):
    """Return (turn directory, prepared task, response) of a completed turn whose result may be reported again.

    Regeneration is local and never a way around the current result-access policy:
    a policy that needs a fresh authenticated identity must use recovery instead.
    """
    if not name:
        raise SessionError("Choose a recorded turn using /history, then /report <turn-id>.")
    output = store.turn_path(store.path, name)
    try:
        prepared = PreparedTask.model_validate_json((output / "prepared_task.json").read_text(encoding="utf-8"))
        response = AgentResponse.model_validate_json((output / "response.json").read_text(encoding="utf-8"))
        if (prepared.confirmation_id != task_confirmation_id(prepared.task_spec)
                or response.submission_state != "completed" or not response.execution
                or response.execution.status.value != "completed" or response.execution.result is None):
            raise ValueError("No completed result")
        if backend == "vantage6":
            submission = RunStore(output / "backend").submission()
            if submission is None or str(submission["task_id"]) != response.execution.task_id:
                raise ValueError("Task identity mismatch")
        for action in ("read_result", "release_result"):
            enforce_policy(agent.policy_checker, prepared.task_spec, action=action, task_id=response.execution.task_id)
        if backend == "vantage6":
            result_adapter(prepared.task_spec.result_adapter).check(response.execution.result, prepared.resolved_plan,
                                                                     agent.semantic_catalog)
    except (OSError, ValueError, KeyError, TypeError):
        raise SessionError("No valid completed result is available for local reporting. Use /recover for an existing "
                           "remote task; no task was submitted.") from None
    except PolicyDeniedError:
        raise SessionError("Current policy denies local report regeneration. Use the approved access process; "
                           "no task was submitted.") from None
    return output, prepared, response
