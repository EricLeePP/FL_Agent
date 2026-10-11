"""The research agent: plan, prepare, execute and recover one task at a time.

Each operation captures one immutable metadata snapshot (metadata.py) and uses
only it. Collaborators are chosen by their declared capabilities
(capabilities.py), and every failure is mapped by services/responses.py.
"""

import logging
from dataclasses import replace

from .algorithm_catalog import AlgorithmCatalog
from .capabilities import (
    DataCapabilities,
    ExecutorCapabilities,
    PlannerCapabilities,
    PolicyCapabilities,
    capabilities_of,
)
from .confirmation import same_catalog, task_confirmation_id
from .conversation import conversation_response
from .errors import (
    AmbiguousAlgorithmError,
    AmbiguousSiteError,
    AmbiguousVariableError,
    BackendExecutionError,
    CatalogError,
    ClarificationRequiredError,
    DataUnavailableError,
    PolicyDeniedError,
    PreflightError,
    SiteNotFoundError,
    TaskValidationError,
    UnsupportedAnalysisError,
    VariableNotFoundError,
)
from .executor import Vantage6Executor, backend_error
from .feedback import issue, suggestions
from .lookups import ResearchLookups
from .metadata import MetadataSnapshot, capture_local, release_snapshot, with_fresh_deployment
from .models import (
    AgentResponse,
    ClarificationQuestion,
    ClarificationRequest,
    DataAvailabilityCheck,
    ExecutionRecord,
    ExecutionStatus,
    PreparedTask,
    ResearchPlan,
    ResolvedResearchPlan,
    ResolvedSite,
    ResolvedVariable,
)
from .observability import log_incident
from .planner import ResearchPlanner
from .policy import PolicyChecker, enforce_policy
from .requirements import protect_requirements
from .result_adapters import result_adapter
from .services.responses import (
    EXECUTION_ERRORS,
    PLANNING_ERRORS,
    RECOVERY_ERRORS,
    execution_failure,
    metadata_failure,
    planning_failure,
    recovery_failure,
)

logger = logging.getLogger(__name__)


class FederatedResearchAgent:
    def __init__(self, metadata: MetadataSnapshot, *, policy_checker: PolicyChecker, executor: Vantage6Executor,
                 metadata_client=None) -> None:
        self.policy_checker = policy_checker
        self.executor = executor
        self.metadata_client = metadata_client
        # What the operator configured; every local capture reads these sources again.
        self._configured = metadata
        self.metadata = metadata

    # Read access to the current snapshot, kept for existing callers.
    semantic_catalog = property(lambda self: self.metadata.semantic_catalog)
    deployment_registry = property(lambda self: self.metadata.deployment_registry)
    algorithm_catalog = property(lambda self: self.metadata.algorithm_catalog)
    task_compiler = property(lambda self: self.metadata.task_compiler)
    result_parser = property(lambda self: self.metadata.result_parser)
    registry_reference = property(lambda self: self.metadata.registry_reference)
    local_metadata_hashes = property(lambda self: self.metadata.local_metadata_hashes)

    @property
    def data_checker(self):
        """The configured availability checker; each capture binds a content snapshot of it."""
        return self._configured.data_checker if self.metadata_client is None else self.metadata.data_checker

    @data_checker.setter
    def data_checker(self, checker):
        self._configured = replace(self._configured, data_checker=checker)
        self.metadata = replace(self.metadata, data_checker=checker)

    @property
    def template_registry(self):
        return self.metadata.template_registry

    @template_registry.setter
    def template_registry(self, registry):
        self._configured = replace(self._configured, template_registry=registry)
        self.metadata = replace(self.metadata, template_registry=registry)

    def refresh_metadata(self, *, existing=False, reference=None) -> MetadataSnapshot:
        """Capture a new snapshot and make it current. A failure keeps nothing partial."""
        if self.metadata_client is not None:
            # Activate a coherent, validated release as a unit; never mix source files.
            self.metadata = release_snapshot(self.metadata_client.sync(existing=existing, reference=reference))
        elif existing:
            self.metadata = with_fresh_deployment(self.metadata)
        else:
            self.metadata = capture_local(self._configured)
        return self.metadata

    def refresh_deployment(self):
        self.metadata = with_fresh_deployment(self.metadata)

    def _catalog_snapshot(self) -> AlgorithmCatalog:
        return self.metadata.catalog()

    def planner_context(self):
        profile = self.deployment_registry.profile
        return {
            "sites": [{"name": s.name, "aliases": s.aliases, "database_labels": s.database_labels} for s in profile.sites],
            "dataset_selection_required": True,
            "semantic_variables_not_data_availability": [
                {"name": v.canonical_name, "aliases": v.aliases, "datatype": v.algorithm_datatype}
                for v in self.semantic_catalog.variables if v.algorithm_datatype],
        }

    def describe_algorithms(self) -> AgentResponse:
        try:
            self.refresh_metadata()
            catalog = self._catalog_snapshot()
        except (CatalogError, PreflightError) as exc:
            return metadata_failure(exc)
        return self._catalog_response(catalog)

    def _catalog_response(self, catalog: AlgorithmCatalog) -> AgentResponse:
        return AgentResponse(
            status="algorithms", message="Registered capabilities; availability is not live verification.",
            catalog=catalog.reference, catalog_document=catalog.document,
            catalog_content=catalog.content,
            algorithms=catalog.knowledge()["algorithms"],
            suggested_algorithms=suggestions(catalog, self.deployment_registry),
            issues=[issue("authorization_unverified", severity="warning"), issue("runtime_unverified", severity="warning")],
        )

    @staticmethod
    def _readiness_issues(prepared):
        items = [issue("runtime_unverified", severity="warning")]
        if prepared.policy_decision.verification != "verified":
            items.append(issue("authorization_unverified", severity="warning"))
        for check in prepared.data_checks:
            if check.state == "unknown":
                code = {"expired": "data_metadata_expired", "not_listed": "data_not_listed"}.get(check.reason, "data_unverified")
                items.append(issue(code, severity="warning", field=f"{check.site}.{check.variable}", source=check.source))
        if any(check.evidence == "metadata_snapshot" for check in prepared.data_checks):
            items.append(issue("data_snapshot_only", severity="warning"))
        return items

    def respond(self, request: str, planner: ResearchPlanner, *, previous_plan=None, questions=None) -> AgentResponse:
        """The request boundary: planning and compilation share one fresh snapshot."""
        conversation = conversation_response(request, planner)
        if conversation is not None:
            return conversation

        def propose(catalog):
            declared = capabilities_of(planner, PlannerCapabilities)
            if declared.uses_context:
                planner.context = self.planner_context()
            if declared.uses_lookups:
                planner.lookups = ResearchLookups(self.metadata, catalog)
            return (planner.revise_plan(request, previous_plan, catalog, questions)
                    if previous_plan is not None else planner.create_plan(request, catalog))

        return self._checked_response(request, propose, previous_plan)

    def check_plan(self, plan: ResearchPlan) -> AgentResponse:
        """Check a complete plan exactly as a planned request is checked. Nothing is written or submitted."""
        return self._checked_response(plan.purpose, lambda catalog: plan.model_copy(deep=True), None)

    def _checked_response(self, request, propose, previous_plan) -> AgentResponse:
        try:
            self.refresh_metadata()
            catalog = self._catalog_snapshot()
        except (CatalogError, PreflightError, OSError, ValueError) as exc:
            return metadata_failure(exc)
        context = {"catalog": catalog.reference, "catalog_document": catalog.document, "catalog_content": catalog.content}
        plan = None
        try:
            plan = propose(catalog)
            if plan.request_type == "catalog_query":
                return self._catalog_response(catalog)
            plan = protect_requirements(request, plan, previous_plan)
            prepared = self._prepare(plan, catalog)
            readiness = self._readiness_issues(prepared)
            if previous_plan is not None and plan.database_label and plan.database_label == previous_plan.database_label:
                readiness.append(issue("dataset_retained", severity="warning", field="database_label",
                                       requested_value=plan.database_label))
            return AgentResponse(status="ready", message="Task prepared for review, not executed. Review the checks below before any real submission.",
                                 research_plan=plan, prepared_task=prepared, issues=readiness, **context)
        except PLANNING_ERRORS as exc:
            return planning_failure(exc, catalog=catalog, plan=plan, deployment=self.deployment_registry)

    def prepare(self, plan: ResearchPlan) -> PreparedTask:
        self.refresh_metadata()
        return self._prepare(plan, self._catalog_snapshot())

    def _prepare(self, plan: ResearchPlan, catalog: AlgorithmCatalog) -> PreparedTask:
        metadata = self.metadata
        plan = protect_requirements(plan.purpose, plan)
        if plan.request_type != "analysis":
            raise TaskValidationError("Catalog questions cannot be compiled into tasks")
        try:
            algorithm = catalog.select(plan.analysis_type, plan.algorithm_id)
        except AmbiguousAlgorithmError as exc:
            raise ClarificationRequiredError(ClarificationRequest(
                research_plan=plan, catalog=catalog.reference,
                questions=[ClarificationQuestion(
                    field="algorithm_id", question=str(exc), candidates=exc.candidates,
                )],
            )) from exc
        questions: list[ClarificationQuestion] = []
        if algorithm.execution:
            for name in algorithm.execution.parameters_schema.get("required", []):
                schema = algorithm.execution.parameters_schema["properties"][name]
                if name not in plan.parameters and "default" not in schema:
                    questions.append(ClarificationQuestion(
                        field=f"parameters.{name}",
                        question=f"Please specify {name}: {schema.get('description', schema.get('type', 'value'))}",
                    ))
        variables = self._resolve_variables(plan.variables, "variables", questions)
        stratification_variables = self._resolve_variables(
            plan.stratification_variables,
            "stratification_variables",
            questions,
        )
        sites = self._resolve_sites(plan.sites, questions)
        candidates = self._database_candidates(sites, algorithm) if sites and not any(q.field == "sites" for q in questions) else []
        if plan.database_label:
            registered_labels = {label for site in metadata.deployment_registry.profile.sites for label in site.database_labels}
            if plan.database_label not in registered_labels:
                questions.append(ClarificationQuestion(field="database_label", requested_value=plan.database_label,
                    question="Which registered database did you mean?", candidates=candidates))
            sites = [site.model_copy(update={"database_label": plan.database_label}) for site in sites]
        else:
            question = "Which dataset should this study use? A deployment default is not a researcher selection."
            if sites and not any(q.field == "sites" for q in questions) and not candidates:
                question += " No common dataset is declared for the selected hospitals, central node and algorithm; ask the administrator to verify the catalog or explicitly revise the study."
            questions.append(ClarificationQuestion(field="database_label", question=question, candidates=candidates))

        if not plan.variables:
            questions.append(
                ClarificationQuestion(
                    field="variables",
                    question="Which variables should be included in the analysis?",
                )
            )
        if not plan.sites:
            questions.append(
                ClarificationQuestion(
                    field="sites",
                    question="Which configured hospitals or data sites should participate?",
                    candidates=[site.name for site in metadata.deployment_registry.profile.sites],
                )
            )
        # Missing dataset selection must not conceal a known unsupported analysis.
        # Other unresolved references still receive their targeted clarification.
        if plan.unsupported_requirements and not any(
            q.field != "database_label" or q.requested_value for q in questions
        ):
            raise UnsupportedAnalysisError(
                "Unmet research requirements: " + "; ".join(plan.unsupported_requirements)
            )
        if questions:
            raise ClarificationRequiredError(
                ClarificationRequest(research_plan=plan, questions=questions, catalog=catalog.reference)
            )

        resolved = ResolvedResearchPlan(
            analysis_type=plan.analysis_type,
            parameters=plan.parameters,
            variables=variables,
            stratification_variables=stratification_variables,
            sites=sites,
            database_label=plan.database_label,
            purpose=plan.purpose,
        )
        task_spec = metadata.task_compiler.compile(resolved, algorithm)
        try:
            result_adapter(task_spec.result_adapter)
        except KeyError:
            raise TaskValidationError("The algorithm's result adapter is not installed",
                                      code="result_adapter_missing") from None
        task_spec.catalog = catalog.reference
        execution = algorithm.execution
        task_spec.execution_template = metadata.template_registry.select(
            execution.template_id if execution else "joshi_vantage6_userclient",
            execution.template_version if execution else None,
        )
        if task_spec.execution_template.vantage6_version != task_spec.connection.version:
            raise PreflightError("Execution template and deployment versions differ")
        task_spec.registry_release = metadata.registry_reference
        task_spec.local_metadata_hashes = dict(metadata.local_metadata_hashes)
        # Verified data evidence or a real policy makes the task managed: it then needs fresh checks.
        task_spec.requires_managed_execution = (
            capabilities_of(metadata.data_checker, DataCapabilities).verifies_availability
            or not capabilities_of(self.policy_checker, PolicyCapabilities).development_allow_all
        )
        decision = enforce_policy(self.policy_checker, task_spec, action="prepare")
        try:
            data_checks = metadata.data_checker.check(resolved, task_spec)
            expected_checks = {(site.canonical_name, variable.canonical_name) for site in resolved.sites for variable in resolved.variables}
            by_key = {}
            for check in data_checks:
                check = DataAvailabilityCheck.model_validate(check)
                key = (check.site, check.variable)
                if key not in expected_checks or key in by_key:
                    raise ValueError("Unexpected or duplicated preflight result")
                by_key[key] = check
            data_checks = [by_key.get((site, variable)) or DataAvailabilityCheck(site=site, variable=variable, state="unknown")
                           for site, variable in sorted(expected_checks)]
        except PreflightError as exc:
            logger.info("Data availability metadata could not be used: %s", exc)
            raise PreflightError("Data availability checker unavailable") from exc
        except Exception as exc:
            incident = log_incident(logger, "Data availability checker failed unexpectedly", exc)
            raise PreflightError("Data availability checker unavailable", incident_id=incident) from exc
        unavailable = [check for check in data_checks if check.state == "unavailable"]
        if unavailable:
            raise DataUnavailableError(unavailable)
        return PreparedTask(
            confirmation_id=task_confirmation_id(task_spec),
            research_plan=plan,
            resolved_plan=resolved,
            task_spec=task_spec,
            policy_decision=decision,
            data_checks=data_checks,
        )

    def _database_candidates(self, sites, algorithm):
        profile = self.deployment_registry.profile
        required_ids = {profile.central_organisation_id, *(site.organisation_id for site in sites)}
        registered = {site.organisation_id: set(site.database_labels) for site in profile.sites}
        labels = set.intersection(*(registered.get(org, set()) for org in required_ids))
        if algorithm.execution:
            scope = next((entry for entry in algorithm.availability if entry.deployment == profile.name), None)
            if scope is None or not required_ids <= set(scope.organisation_ids):
                return []
            labels &= set(scope.database_labels)
        return sorted(labels)

    def execute(
        self,
        prepared: PreparedTask,
        confirmation_id: str,
    ) -> ExecutionRecord:
        self.refresh_metadata()
        record = self._execute(prepared, confirmation_id, self._catalog_snapshot())
        self._runtime_policy(prepared.task_spec, action="release_result", task_id=record.task_id)
        return record

    def _bind_executor_policy(self):
        """Give a trusted worker the policy file from which it repeats the same decisions."""
        if capabilities_of(self.executor, ExecutorCapabilities).trusted_worker:
            config = capabilities_of(self.policy_checker, PolicyCapabilities).worker_config
            if config is None:
                raise PreflightError("This policy provider has no trusted worker configuration")
            self.executor.policy_config = config

    def _runtime_policy(self, task, *, action, task_id=None):
        # The trusted worker owns runtime decisions for file-configured policies:
        # only it has the authenticated Vantage6 subject and recorded task ID.
        if (capabilities_of(self.executor, ExecutorCapabilities).trusted_worker
                and capabilities_of(self.policy_checker, PolicyCapabilities).worker_decides_runtime):
            return
        enforce_policy(self.policy_checker, task, action=action, task_id=task_id)

    def _execute(self, prepared, confirmation_id, current):
        expected = task_confirmation_id(prepared.task_spec)
        if prepared.confirmation_id != expected:
            raise TaskValidationError(
                "Prepared task content changed after its confirmation ID was created", code="confirmation_invalid",
            )
        if confirmation_id != expected:
            raise TaskValidationError(
                "Researcher confirmation does not match the exact prepared task", code="confirmation_invalid",
            )
        if self.algorithm_catalog.path:
            if not same_catalog(prepared.task_spec.catalog, current.reference):
                raise TaskValidationError("Algorithm catalog changed or is missing; prepare and confirm a new task", code="catalog_changed")
        # Recheck permissions and data, including for non-file-backed catalogs.
        refreshed = self._prepare(prepared.research_plan, current)
        if refreshed.task_spec.requires_managed_execution and any(check.state == "unknown" for check in refreshed.data_checks):
            raise PreflightError("Managed task requires current data availability evidence", code="data_verification_required")
        if task_confirmation_id(refreshed.task_spec) != expected:
            raise TaskValidationError("Task no longer matches the catalog and deployment contracts", code="confirmation_invalid")
        if not prepared.policy_decision.allowed:
            raise PolicyDeniedError("The task was denied by the policy checker")
        self._runtime_policy(refreshed.task_spec, action="submit")
        self._runtime_policy(refreshed.task_spec, action="read_result")
        self._bind_executor_policy()
        try:
            submitted = self.executor.submit(prepared.task_spec)
        except Exception as exc:
            raise backend_error(exc, stage="submit") from exc
        try:
            record = self.executor.wait_for_results(submitted.task_id)
        except Exception as exc:
            raise backend_error(exc, stage="result", task_id=submitted.task_id) from exc
        if record.status == ExecutionStatus.failed:
            raise BackendExecutionError("backend_task_failed", stage="result", task_id=submitted.task_id)
        if record.status != ExecutionStatus.completed:
            raise BackendExecutionError("backend_incomplete", stage="result", task_id=submitted.task_id)
        return record

    def execute_response(self, prepared: PreparedTask, confirmation_id: str) -> AgentResponse:
        """User-facing execution boundary; never return raw backend exceptions."""
        catalog = None
        record = None
        try:
            self.refresh_metadata()
            catalog = self._catalog_snapshot()
            record = self._execute(prepared, confirmation_id, catalog)
            return self._completed_response(prepared, record, catalog.reference)
        except EXECUTION_ERRORS as exc:
            return execution_failure(exc, catalog=catalog, prepared=prepared, record=record,
                                     deployment=self.deployment_registry)

    def _completed_response(self, prepared, record, catalog_reference):
        self._runtime_policy(prepared.task_spec, action="release_result", task_id=record.task_id)
        simulated = capabilities_of(self.executor, ExecutorCapabilities).simulated
        if not simulated:
            try:
                adapter = result_adapter(prepared.task_spec.result_adapter)
            except KeyError:
                raise BackendExecutionError("result_adapter_missing", stage="result", task_id=record.task_id) from None
            try:
                adapter.check(record.result, prepared.resolved_plan, self.semantic_catalog)
            except (ValueError, TypeError, KeyError) as exc:
                raise BackendExecutionError("result_invalid", stage="result", task_id=record.task_id) from exc
        message = ("Mock execution completed; no real Vantage6 task was submitted." if simulated
                   else "The backend returned a completed task. Scientific result validation is separate.")
        return AgentResponse(status="completed", message=message, catalog=catalog_reference,
                             execution=record, submission_state="completed")

    def recover_response(self, prepared: PreparedTask) -> AgentResponse:
        """Retrieve an already submitted task; never recompile or submit it again."""
        task = prepared.task_spec
        try:
            if prepared.confirmation_id != task_confirmation_id(task):
                raise BackendExecutionError("recovery_mismatch", stage="preflight")
            self.refresh_metadata(existing=True, reference=task.registry_release)
            profile = self.deployment_registry.profile
            actual = (profile.name, profile.server_url, profile.server_port, profile.server_api, profile.collaboration_id)
            original = (task.connection.environment, task.connection.server_url, task.connection.server_port,
                        task.connection.server_api, task.collaboration_id)
            if actual != original or not capabilities_of(self.executor, ExecutorCapabilities).recovery:
                raise BackendExecutionError("recovery_mismatch", stage="preflight")
            # Keep a policy hook on result retrieval. Fresh data inventory gates NEW
            # computations, not read-only recovery of an existing server task.
            self._runtime_policy(task, action="read_result")
            self._bind_executor_policy()
            try:
                record = self.executor.recover(task)
            except Exception as exc:
                raise backend_error(exc, stage="result", task_id=self.executor.task_id) from exc
            return self._completed_response(prepared, record, task.catalog)
        except RECOVERY_ERRORS as exc:
            return recovery_failure(exc, task_id=getattr(self.executor, "task_id", None))

    def _resolve_variables(
        self,
        queries: list[str],
        field: str,
        questions: list[ClarificationQuestion],
    ) -> list[ResolvedVariable]:
        resolved = []
        for query in queries:
            try:
                resolved.append(self.semantic_catalog.resolve(query))
            except (AmbiguousVariableError, VariableNotFoundError) as exc:
                questions.append(
                    ClarificationQuestion(
                        field=field,
                        requested_value=query,
                        question=f"Which schema variable did you mean by '{query}'?",
                        candidates=getattr(exc, "candidates", []),
                    )
                )
        return resolved

    def _resolve_sites(
        self,
        queries: list[str],
        questions: list[ClarificationQuestion],
    ) -> list[ResolvedSite]:
        resolved = []
        seen = set()
        for query in queries:
            try:
                site = self.deployment_registry.resolve(query)
            except (AmbiguousSiteError, SiteNotFoundError) as exc:
                questions.append(
                    ClarificationQuestion(
                        field="sites",
                        requested_value=query,
                        question=f"Which configured site did you mean by '{query}'?",
                        candidates=exc.candidates,
                    )
                )
                continue
            if site.organisation_id not in seen:
                resolved.append(site)
                seen.add(site.organisation_id)
        return resolved
