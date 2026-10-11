"""Deterministic, source-backed explanations; the LLM never invents alternatives."""

import logging

from .algorithm_catalog import SUPPORTED_ADAPTERS
from .errors import AmbiguousSiteError, SiteNotFoundError
from .models import AgentResponse, AlgorithmSuggestion, ResearcherIssue

logger = logging.getLogger(__name__)


MESSAGES = {
    "submission_unresolved": ("This attempt has no trustworthy recorded task ID; submission may already have succeeded.", "Do not resubmit. Ask the administrator to reconcile this attempt with Vantage6 server task history. Automatic resubmission is disabled."),
    "recovery_mismatch": ("The saved task, confirmation or current deployment does not match the original execution.", "Restore the original configuration and inspect the saved artifacts. Recovery will not redirect to another server or submit a replacement task."),
    "execution_busy": ("Another local process is still monitoring this task.", "Use its terminal or wait for it to stop before /recover. No new task was submitted."),
    "planner_billing": ("The LLM provider rejected the request with HTTP 402 (billing/payment required). No Vantage6 task was submitted.", "Check the provider account balance and billing, or select a configured local model. Do not paste API keys into the research request."),
    "planner_authentication": ("The LLM provider rejected the API credential with HTTP 401.", "Update the key through the local credential prompt or environment, never through the research request."),
    "planner_rate_limit": ("The LLM provider rate limit was reached (HTTP 429). No Vantage6 task was submitted.", "Retry planning later or select another configured provider. This does not authorize a task submission."),
    "duplicate_execution": ("This prepared artifact already has an execution attempt recorded.", "Inspect its backend folder and server task history. Do not resubmit until the previous attempt is accounted for."),
    "unknown_database": ("The requested database is not registered at the required sites.", "Select a registered database explicitly or ask the administrator to configure it. The agent will not substitute the default silently."),
    "missing_database": ("The study dataset has not been selected.", "Select a listed dataset explicitly. No default or sole candidate will be chosen automatically."),
    "dataset_retained": ("The dataset is retained from the active study plan.", "Specify a different dataset to change it, or explicitly clear the dataset selection to choose again. Use /new for a separate study."),
    "site_count_unsupported": ("This algorithm version cannot execute the requested variable combination with this number of hospitals.", "Review the catalog limitation or request a fixed algorithm version. Adding hospitals changes your study and requires an explicit revised request."),
    "result_adapter_missing": ("This algorithm's result handling is not installed in this agent, so its result cannot be checked or reported.", "Ask the administrator to install the result adapter that the algorithm catalog names. No new task was submitted."),
    "result_invalid": ("A result was returned, but its structure or requested-variable coverage failed validation.", "Inspect the recorded task with the administrator. No scientific report is produced and no task is resubmitted automatically."),
    "backend_configuration": ("The real backend is restricted to the configured synthetic Vantage6 4.15 sandbox.", "Check the sandbox configuration and explicit unencrypted-synthetic confirmation. Production and encrypted collaborations need a separate validated connection setup."),
    "unknown_variable": ("The requested variable is not an exact, unambiguous match in the semantic catalog.", "Confirm a listed variable or ask the data steward to register the missing semantic definition."),
    "missing_variables": ("The request does not specify which variables to analyze.", "Specify the variables; no variables will be selected automatically."),
    "unknown_site": ("The requested hospital cannot be resolved in the deployment registry.", "Confirm a configured hospital or ask the deployment administrator to register the intended site."),
    "missing_sites": ("The participating hospitals have not been specified.", "Select the intended hospitals; the agent will not silently use all sites."),
    "missing_parameter": ("An algorithm parameter needs clarification.", "Provide the requested parameter or select one of the listed algorithm IDs."),
    "unsupported_analysis": ("No enabled catalog algorithm matches the requested analysis and algorithm ID.", "Review the catalog options below. A different analysis changes the research question and requires your confirmation; otherwise request algorithm registration or development."),
    "unmet_requirements": ("One or more requested conditions are not supported by the current algorithm contract.", "The agent will not drop filters or substitute an analysis automatically. Existing conditions remain protected across follow-ups. To withdraw them, use /new and explicitly state the complete revised study, then review it before execution; otherwise request a supporting algorithm."),
    "report_failed": ("Computation completed, but the local report could not be generated. The returned result is retained.", "Correct the local report/output problem, then use /report <turn-id> from /history. This rebuilds the report without another model call or Vantage6 submission."),
    "invalid_request": ("The request cannot be compiled under the current task contract.", "Review the variables and settings, or ask the administrator to check the algorithm contract."),
    "invalid_parameters": ("An input parameter does not satisfy the registered algorithm schema.", "Check parameter names, types and permitted ranges; provide corrected values."),
    "unsupported_variable_type": ("The variable exists in the schema, but its datatype is not supported by this analysis contract.", "Choose an appropriate analysis or ask the administrator for a validated datatype mapping. Do not coerce identifiers into numeric measurements."),
    "unsupported_stratification": ("This agent contract does not implement the requested stratification.", "Request a validated stratification contract, or explicitly revise the analysis."),
    "adapter_unavailable": ("The algorithm is registered but requires a new adapter before this agent can execute it.", "Ask the developer to implement and test its execution protocol; changing the model or enabling the entry is insufficient."),
    "algorithm_deployment_unavailable": ("The algorithm is not declared for this deployment.", "Ask the administrator to verify and register its availability in the intended deployment."),
    "algorithm_site_unavailable": ("The algorithm is not declared for every requested site and the central execution organization.", "Ask the administrator to verify the missing deployment coverage. Do not silently remove hospitals."),
    "algorithm_database_unavailable": ("The algorithm is not declared for the configured database label.", "Ask the administrator to verify the data binding and register the correct database label."),
    "version_mismatch": ("The algorithm and deployment require different Vantage6 versions.", "Ask the administrator for a compatible, validated algorithm version; do not change versions automatically."),
    "data_unavailable": ("The data preflight reports that a requested variable is unavailable at a selected site.", "Ask the data steward to verify the mapping and availability. Changing variables or excluding sites requires an explicit revised request."),
    "data_unverified": ("Actual data availability has not been verified for every selected variable and hospital.", "Obtain current site metadata or a trusted backend preflight. A semantic definition is not proof of data availability."),
    "data_snapshot_only": ("Data availability is based on an operator metadata snapshot, not a live query.", "Verify that the snapshot is current before real execution. It does not prove non-empty cohorts, data quality or permission."),
    "data_metadata_expired": ("The data availability metadata have expired; previous availability claims cannot be relied on.", "Ask the data steward for a fresh snapshot or use a live preflight check."),
    "data_not_listed": ("No data availability declaration covers this selected site and variable.", "Ask the data steward to check and declare it. A missing declaration means unknown, not that the data do not exist."),
    "data_verification_required": ("This managed task cannot be submitted while any requested data availability check is unknown or expired.", "Refresh the selected-site metadata or complete a trusted backend check, then prepare and confirm the task again."),
    "authorization_unverified": ("Researcher-specific authorization has not been checked; the MVP policy is not an access grant.", "Use a trusted authorization service and an authenticated researcher context before production execution."),
    "runtime_unverified": ("Catalog availability does not guarantee that the image, nodes and algorithm are executable now.", "Verify the deployment and retain task/result evidence from a real sandbox run."),
    "permission_denied": ("The authorization service does not permit this request for the current identity.", "Contact the project administrator through the approved access-request process. Do not change identity or remove access controls."),
    "preflight_error": ("A required preflight service or metadata source could not be validated.", "Ask the administrator to restore or correct the source, then prepare the request again. This is not proof that data do not exist."),
    "catalog_error": ("The current algorithm catalog could not be loaded or validated.", "Ask the administrator to repair the catalog. No cached or model-invented alternatives will be used."),
    "planner_error": ("The research request could not be converted into a valid structured plan.", "Clarify the analysis, variables and hospitals. Check the model service if the problem persists."),
    "confirmation_invalid": ("The confirmation does not match the exact prepared task.", "Review and confirm a newly prepared task; do not reuse an old confirmation."),
    "catalog_changed": ("The algorithm catalog changed after this task was prepared.", "Prepare and review a new task using the current catalog before submitting."),
    "backend_authentication": ("The execution backend did not accept the current authentication.", "Check the runtime login or session with the administrator. Never put credentials in the research request."),
    "backend_forbidden": ("The execution backend denied access to this operation.", "Request the necessary authorization through the project administrator. No restricted-resource details are disclosed."),
    "backend_not_found": ("The backend could not access a resource required by this task.", "Ask the administrator to verify the task, deployment and resource references. This response does not establish whether a hidden resource exists."),
    "backend_timeout": ("The backend did not return a response within the available wait period.", "Check the original task before retrying. Restore the connection, then use /history and /recover <turn-id> for a recorded task. Without a task ID, ask the administrator to reconcile server history; do not resubmit."),
    "backend_error": ("The execution backend did not complete the operation successfully.", "Ask the administrator to inspect protected logs and node/image status. Do not blindly resubmit or interpret this as a successful analysis."),
    "backend_task_failed": ("The backend reported that the submitted task failed.", "Ask the administrator to inspect the task and node logs. No report will be generated from a failed result."),
    "backend_incomplete": ("The submitted task has not returned a completed result.", "Check its recorded task ID and node status. Do not submit a duplicate task or treat partial completion as full success."),
}


def issue(code, *, severity="error", field=None, requested_value=None, candidates=None, source=None, constraint=None,
          incident_id=None):
    if code not in MESSAGES:
        logger.warning("Unknown researcher issue code %r; using the generic invalid_request message", code)
    message, action = MESSAGES.get(code, MESSAGES["invalid_request"])
    return ResearcherIssue(code=code, severity=severity, message=message, field=field,
                           requested_value=requested_value, next_steps=[action], candidates=candidates or [], source=source,
                           constraint=constraint, incident_id=incident_id)


def suggestions(catalog, deployment, plan=None):
    if catalog is None:
        return []
    profile = deployment.profile
    selected = set()
    if plan and plan.sites:
        try:
            selected = {deployment.resolve(name).organisation_id for name in plan.sites}
        except (SiteNotFoundError, AmbiguousSiteError):
            return []
    choices = []
    for algorithm in catalog.algorithms:
        if algorithm.status != "enabled" or not algorithm.execution or algorithm.execution.adapter not in SUPPORTED_ADAPTERS:
            continue
        if algorithm.vantage6_version.split(".")[:2] != profile.vantage6_version.split(".")[:2]:
            continue
        scope = next((entry for entry in algorithm.availability if entry.deployment == profile.name), None)
        label = plan.database_label if plan else None
        if not scope or (label is not None and label not in scope.database_labels):
            continue
        if not {profile.central_organisation_id, *selected} <= set(scope.organisation_ids):
            continue
        if selected and len(selected) < algorithm.constraints.min_sites:
            continue
        choices.append(AlgorithmSuggestion(
            algorithm_id=algorithm.algorithm_id, display_name=algorithm.display_name,
            version=algorithm.algorithm_version, capabilities=algorithm.capabilities,
            limitations=algorithm.limitations, scope="declared_for_selection" if selected else "select_sites_first",
            verification=scope.verification, catalog=catalog.reference,
        ))
    return choices


def clarify_issues(clarification):
    items = []
    for question in clarification.questions:
        if question.field in ("variables", "stratification_variables"):
            code = "unknown_variable" if question.requested_value else "missing_variables"
        elif question.field == "sites":
            code = "unknown_site" if question.requested_value else "missing_sites"
        elif question.field == "database_label":
            code = "unknown_database" if question.requested_value else "missing_database"
        else:
            code = "missing_parameter"
        items.append(issue(code, severity="question", field=question.field,
                           requested_value=question.requested_value, candidates=question.candidates,
                           constraint=question.question if question.field.startswith("parameters.") else None))
    return items


def render_researcher_response(response: AgentResponse) -> str:
    lines = ["# Researcher Response", "", f"Status: {response.status}",
             f"Submission: {response.submission_state}", "", response.message, ""]
    if response.execution:
        lines.extend([f"Task ID: {response.execution.task_id}", ""])
    if response.clarification:
        lines.extend(["## Clarification", *[question.question for question in response.clarification.questions], ""])
    for item in response.issues:
        lines.extend([f"## {item.severity.title()}: {item.code}", item.message])
        if item.field:
            lines.append(f"Field: {item.field}")
        if item.requested_value:
            lines.append(f"Requested: {item.requested_value}")
        if item.candidates:
            lines.append("Candidates requiring confirmation: " + ", ".join(item.candidates))
        if item.source:
            lines.append(f"Evidence: {item.source.kind}; version: {item.source.version}")
        if item.constraint:
            lines.append(f"Expected: {item.constraint}")
        if item.incident_id:
            lines.append(f"Incident ID: {item.incident_id} (the full error is in the agent log)")
        lines.extend(f"- {step}" for step in item.next_steps)
        lines.append("")
    if response.suggested_algorithms:
        lines.extend(["## Catalog Options", "These are alternatives to discuss, not automatic replacements or authorization grants.", ""])
        for item in response.suggested_algorithms:
            lines.extend([f"### {item.display_name} ({item.algorithm_id}, {item.version})",
                          f"Scope: {item.scope}; evidence: {item.verification}; authorization and data availability: not verified."])
            lines.extend(f"- Capability: {value}" for value in item.capabilities)
            lines.extend(f"- Limitation: {value}" for value in item.limitations)
            lines.append("")
    if response.catalog:
        lines.extend([f"Catalog version: {response.catalog.catalog_version}", f"Catalog SHA-256: {response.catalog.sha256}"])
    return "\n".join(lines).rstrip() + "\n"
