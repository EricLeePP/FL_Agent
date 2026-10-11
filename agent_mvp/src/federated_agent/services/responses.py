"""Researcher-facing responses for every failure, mapped in one place.

The agent and the transports raise typed errors. This module decides the
response status, the researcher issue and what is known about submission, so
the same failure reads the same way in a one-shot command, a saved study or a
recovery. Raw exception text never reaches a researcher.
"""

from http.client import HTTPException
from urllib.error import HTTPError, URLError

from pydantic import ValidationError

from ..errors import (
    BackendExecutionError,
    CatalogError,
    ClarificationRequiredError,
    DataUnavailableError,
    LLMResponseError,
    PolicyDeniedError,
    PreflightError,
    TaskValidationError,
    UnsupportedAnalysisError,
    nothing_submitted,
)
from ..feedback import clarify_issues, issue, suggestions
from ..models import AgentResponse, ExecutionRecord, ExecutionStatus

PLANNER_HTTP_CODES = {401: "planner_authentication", 402: "planner_billing", 429: "planner_rate_limit"}
PLANNER_FAILURES = (LLMResponseError, ValidationError, URLError, TimeoutError, ConnectionError, HTTPException)
PLANNING_ERRORS = (ClarificationRequiredError, UnsupportedAnalysisError, *PLANNER_FAILURES, TaskValidationError,
                   PolicyDeniedError, DataUnavailableError, PreflightError)
EXECUTION_ERRORS = (CatalogError, PolicyDeniedError, DataUnavailableError, PreflightError, TaskValidationError,
                    ClarificationRequiredError, UnsupportedAnalysisError, BackendExecutionError)
RECOVERY_ERRORS = (BackendExecutionError, PreflightError, PolicyDeniedError)


def error_response(status, code, *, deployment=None, catalog=None, plan=None, issues=None, private=False,
                   incident_id=None) -> AgentResponse:
    """A failure response. ``private`` hides catalog content and suggestions (denials, backend errors)."""
    requested = None
    if plan and code == "unsupported_analysis":
        requested = plan.algorithm_id or (plan.purpose if plan.analysis_type == "unsupported" else plan.analysis_type)
    problem = issue(code, requested_value=requested, incident_id=incident_id)
    context = {"catalog": catalog.reference} if catalog else {}
    if catalog and not private:
        context.update(catalog_document=catalog.document, catalog_content=catalog.content)
    return AgentResponse(status=status, message=problem.message, research_plan=plan, issues=issues or [problem],
                         suggested_algorithms=[] if private else suggestions(catalog, deployment, plan), **context)


def metadata_failure(exc) -> AgentResponse:
    """Metadata could not be captured, so nothing was planned or prepared."""
    if isinstance(exc, CatalogError):
        return error_response("catalog_error", "catalog_error")
    return error_response("preflight_error", "preflight_error")


def data_unavailable(exc, *, catalog, plan, deployment) -> AgentResponse:
    problems = [issue("data_unavailable", field=f"{check.site}.{check.variable}", source=check.source) for check in exc.checks]
    # No site/variable substitution is proposed for a known unavailable request.
    return error_response("data_unavailable", "data_unavailable", deployment=deployment, catalog=catalog,
                          plan=plan, issues=problems, private=True)


def planning_failure(exc, *, catalog, plan, deployment) -> AgentResponse:
    """Map a failure while planning or preparing a task. Nothing was submitted."""
    context = {"catalog": catalog.reference, "catalog_document": catalog.document, "catalog_content": catalog.content}
    common = {"deployment": deployment, "catalog": catalog, "plan": plan}
    if isinstance(exc, ClarificationRequiredError):
        return AgentResponse(status="needs_clarification", message="Please clarify the items below. No task was submitted.",
                             clarification=exc.request, issues=clarify_issues(exc.request),
                             suggested_algorithms=suggestions(catalog, deployment, plan), **context)
    if isinstance(exc, UnsupportedAnalysisError):
        if plan and plan.unsupported_requirements:
            return error_response("unsupported", "unmet_requirements", **common,
                                  issues=[issue("unmet_requirements", requested_value=value) for value in plan.unsupported_requirements])
        return error_response("unsupported", "unsupported_analysis", **common)
    if isinstance(exc, HTTPError):
        return error_response("planner_error", PLANNER_HTTP_CODES.get(exc.code, "planner_error"), **common)
    if isinstance(exc, PLANNER_FAILURES):
        return error_response("planner_error", "planner_error", **common)
    if isinstance(exc, TaskValidationError):
        return error_response("invalid_request", exc.code, **common,
                              issues=[issue(exc.code, field=exc.field, constraint=exc.constraint)])
    if isinstance(exc, PolicyDeniedError):
        return error_response("permission_denied", "permission_denied", **common, private=True)
    if isinstance(exc, DataUnavailableError):
        return data_unavailable(exc, catalog=catalog, plan=plan, deployment=deployment)
    if isinstance(exc, PreflightError):
        return error_response("preflight_error", exc.code, **common, private=True, incident_id=exc.incident_id)
    raise exc


def backend_failure(code, *, task_id=None, stage="submit", incident_id=None, catalog=None, recovery=False) -> AgentResponse:
    """A failed or uncertain execution. Recovery never proves that nothing was submitted."""
    response = error_response("execution_failed", code, catalog=catalog, private=True, incident_id=incident_id)
    if task_id:
        failed = code == "backend_task_failed"
        response.execution = ExecutionRecord(task_id=task_id, error=code,
                                             status=ExecutionStatus.failed if failed else ExecutionStatus.submitted)
        response.submission_state = "failed" if failed else "submitted"
    elif not recovery and nothing_submitted(code, stage):
        response.submission_state = "not_submitted"
    else:
        response.submission_state = "unknown"
    return response


def submitted_but_withheld(response, record) -> AgentResponse:
    """The task ran, but releasing its result was refused or could not be checked."""
    response.submission_state = "submitted"
    response.execution = ExecutionRecord(task_id=record.task_id, status=ExecutionStatus.submitted)
    return response


def execution_failure(exc, *, catalog, prepared, record, deployment) -> AgentResponse:
    """Map a failure of a confirmed execution; ``record`` is set once the backend has run it."""
    plan = prepared.research_plan
    if isinstance(exc, CatalogError):
        return error_response("catalog_error", "catalog_error")
    if isinstance(exc, PolicyDeniedError):
        response = error_response("permission_denied", "permission_denied", catalog=catalog, private=True)
        return submitted_but_withheld(response, record) if record else response
    if isinstance(exc, DataUnavailableError):
        return data_unavailable(exc, catalog=catalog, plan=plan, deployment=deployment)
    if isinstance(exc, PreflightError):
        response = error_response("preflight_error", exc.code, catalog=catalog, private=True, incident_id=exc.incident_id)
        return submitted_but_withheld(response, record) if record else response
    if isinstance(exc, TaskValidationError):
        return error_response("invalid_request", exc.code, deployment=deployment, catalog=catalog, plan=plan,
                              issues=[issue(exc.code, field=exc.field, constraint=exc.constraint)])
    if isinstance(exc, ClarificationRequiredError):
        code = ("missing_database" if any(q.field == "database_label" and not q.requested_value
                for q in exc.request.questions) else "catalog_changed")
        return error_response("invalid_request", code, deployment=deployment, catalog=catalog, plan=plan)
    if isinstance(exc, UnsupportedAnalysisError):
        return error_response("invalid_request", "catalog_changed", deployment=deployment, catalog=catalog, plan=plan)
    if isinstance(exc, BackendExecutionError):
        return backend_failure(exc.code, task_id=exc.task_id, stage=exc.stage, incident_id=exc.incident_id, catalog=catalog)
    raise exc


def recovery_failure(exc, *, task_id) -> AgentResponse:
    """Map a failure while reading back an existing remote task."""
    code = "permission_denied" if isinstance(exc, PolicyDeniedError) else exc.code
    return backend_failure(code, task_id=task_id, incident_id=getattr(exc, "incident_id", None), recovery=True)
