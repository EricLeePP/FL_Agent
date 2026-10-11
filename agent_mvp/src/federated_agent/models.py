"""Strict data contracts shared by all agent components."""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator


# Name of an installed result adapter (result_adapters.py); registry JSON only names one.
ADAPTER_NAME_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceReference(StrictModel):
    kind: str
    uri: str
    version: str | None = None


class FieldProvenance(StrictModel):
    field: str
    source: SourceReference


class ResearchPlan(StrictModel):
    request_type: Literal["analysis", "catalog_query"] = "analysis"
    analysis_type: str
    algorithm_id: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    unsupported_requirements: list[str] = Field(default_factory=list)
    variables: list[str] = Field(default_factory=list)
    stratification_variables: list[str] = Field(default_factory=list)
    sites: list[str] = Field(default_factory=list)
    database_label: str | None = None
    purpose: str = Field(min_length=1)

    @field_validator("database_label")
    @classmethod
    def normalize_database_label(cls, value):
        return value.strip() or None if value is not None else None

    @field_validator("variables", "stratification_variables", "sites")
    @classmethod
    def normalize_names(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        return list(dict.fromkeys(cleaned))


class CatalogVariable(StrictModel):
    canonical_name: str
    class_id: str
    schema_datatype: str
    algorithm_datatype: Literal["numerical", "categorical"] | None
    aliases: list[str] = Field(default_factory=list)


class ResolvedVariable(StrictModel):
    requested_name: str
    canonical_name: str
    class_id: str
    datatype: Literal["numerical", "categorical"]
    source: SourceReference


class DeploymentSite(StrictModel):
    name: str
    organisation_id: int
    aliases: list[str] = Field(default_factory=list)
    database_labels: list[str] = Field(min_length=1)

    @field_validator("organisation_id")
    @classmethod
    def validate_organisation_id(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Organisation IDs must be positive integers")
        return value


class DeploymentProfile(StrictModel):
    name: str
    registry_version: str
    sandbox: bool
    vantage6_version: str
    server_url: str
    server_port: int
    server_api: str
    collaboration_id: int
    central_organisation_id: int
    database_label: str
    sites: list[DeploymentSite] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_profile(self) -> "DeploymentProfile":
        if self.server_port <= 0:
            raise ValueError("Server port must be positive")
        if self.collaboration_id <= 0 or self.central_organisation_id <= 0:
            raise ValueError("Collaboration and central organisation IDs must be positive")
        organisation_ids = [site.organisation_id for site in self.sites]
        if len(organisation_ids) != len(set(organisation_ids)):
            raise ValueError("Deployment site organisation IDs must be unique")
        missing_database = [
            site.name
            for site in self.sites
            if self.database_label not in site.database_labels
        ]
        if missing_database:
            raise ValueError(
                f"Database label '{self.database_label}' is unavailable at sites: "
                f"{', '.join(missing_database)}"
            )
        return self


class ResolvedSite(StrictModel):
    requested_name: str
    canonical_name: str
    organisation_id: int
    database_label: str | None = None
    source: SourceReference


class ResolvedResearchPlan(StrictModel):
    analysis_type: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    variables: list[ResolvedVariable]
    stratification_variables: list[ResolvedVariable] = Field(default_factory=list)
    sites: list[ResolvedSite]
    database_label: str | None = None
    purpose: str


class CatalogReference(StrictModel):
    path: str
    schema_version: str
    catalog_version: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    # Hash of the parsed JSON value: independent of file location, line endings and layout.
    content_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class InputBinding(StrictModel):
    source: Literal["variables_by_class", "organisation_ids", "parameter", "constant"]
    parameter: str | None = None
    value: Any = None

    @model_validator(mode="after")
    def validate_binding(self) -> "InputBinding":
        if (self.source == "parameter") != bool(self.parameter):
            raise ValueError("Only parameter bindings must name a parameter")
        if self.source == "constant" and "value" not in self.model_fields_set:
            raise ValueError("Constant bindings require an explicit value (possibly null)")
        if self.source != "constant" and self.value is not None:
            raise ValueError("Only constant bindings may contain value")
        return self


class AlgorithmExecution(StrictModel):
    adapter: str
    input_bindings: dict[str, InputBinding]
    parameters_schema: dict[str, Any]
    kwargs_schema: dict[str, Any]
    result_adapter: str = Field(default="raw_json", pattern=ADAPTER_NAME_PATTERN)
    template_id: str = "joshi_vantage6_userclient"
    template_version: str | None = None


class AlgorithmAvailability(StrictModel):
    deployment: str
    organisation_ids: list[int] = Field(min_length=1)
    database_labels: list[str] = Field(min_length=1)
    verification: Literal["declared", "sandbox_verified"] = "declared"
    evidence: str = Field(min_length=1)

    @field_validator("organisation_ids")
    @classmethod
    def validate_ids(cls, value: list[int]) -> list[int]:
        if any(item <= 0 for item in value) or len(value) != len(set(value)):
            raise ValueError("Availability requires unique positive organisation IDs")
        return value


class AlgorithmConstraints(StrictModel):
    min_sites: int = Field(default=1, ge=1)
    min_sites_for_numerical: int = Field(default=1, ge=1)


class AlgorithmContract(StrictModel):
    algorithm_id: str
    display_name: str
    algorithm_version: str
    vantage6_version: str
    analysis_types: list[str]
    function: str
    image: str
    accepted_arguments: list[str]
    status: Literal["enabled", "draft", "disabled"] = "enabled"
    description: str = ""
    request_keywords: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    constraints: AlgorithmConstraints = Field(default_factory=AlgorithmConstraints)
    availability: list[AlgorithmAvailability] = Field(default_factory=list)
    execution: AlgorithmExecution
    catalog_source: SourceReference | None = None
    supports_stratification: bool = False
    source: SourceReference
    metadata_source: SourceReference | None = None
    contract_warnings: list[str] = Field(default_factory=list)


class DatabaseSpec(StrictModel):
    label: str
    query: str | None = None
    sheet_name: str | None = None


class Vantage6Connection(StrictModel):
    environment: str
    sandbox: bool
    version: str
    server_url: str
    server_port: int
    server_api: str


class RegistryReference(StrictModel):
    release_id: str
    sequence: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ExecutionTemplateReference(StrictModel):
    id: str
    version: str
    runtime_adapter: str
    export_asset: str
    export_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    adapter_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    poll_interval_seconds: float = Field(gt=0, le=60)
    vantage6_version: str
    # 2: newline-normalized hashes of the worker sources.
    digest_version: int = 2


class Vantage6TaskSpec(StrictModel):
    algorithm_id: str
    algorithm_version: str
    image: str
    connection: Vantage6Connection
    collaboration_id: int
    central_organisation_id: int
    name: str
    description: str
    input: dict[str, Any]
    databases: list[DatabaseSpec]
    provenance: list[FieldProvenance] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    catalog: CatalogReference | None = None
    result_adapter: str = Field(default="descriptive_statistics_v1", pattern=ADAPTER_NAME_PATTERN)
    requires_managed_execution: bool = False
    registry_release: RegistryReference | None = None
    execution_template: ExecutionTemplateReference | None = None
    local_metadata_hashes: dict[str, str] = Field(default_factory=dict)
    fingerprint_version: int = 2

    def to_client_payload(self) -> dict[str, Any]:
        return {
            "collaboration": self.collaboration_id,
            "organizations": [self.central_organisation_id],
            "name": self.name,
            "description": self.description,
            "image": self.image,
            "input_": self.input,
            "databases": [
                database.model_dump(exclude_none=True) for database in self.databases
            ],
        }


class PolicyDecision(StrictModel):
    allowed: StrictBool
    verification: Literal["not_checked", "verified"] = "not_checked"
    reasons: list[str] = Field(default_factory=list)
    obligations: list[str] = Field(default_factory=list)
    policy_version: str | None = None
    decision_id: str | None = None


class ClarificationQuestion(StrictModel):
    field: str
    question: str
    requested_value: str | None = None
    candidates: list[str] = Field(default_factory=list)


class ClarificationRequest(StrictModel):
    status: Literal["needs_clarification"] = "needs_clarification"
    research_plan: ResearchPlan
    questions: list[ClarificationQuestion] = Field(min_length=1)
    catalog: CatalogReference | None = None


class PreparedTask(StrictModel):
    confirmation_id: str
    research_plan: ResearchPlan
    resolved_plan: ResolvedResearchPlan
    task_spec: Vantage6TaskSpec
    policy_decision: PolicyDecision
    data_checks: list["DataAvailabilityCheck"] = Field(default_factory=list)


class DataAvailabilityCheck(StrictModel):
    site: str
    variable: str
    state: Literal["available", "unavailable", "unknown"]
    evidence: Literal["not_checked", "metadata_snapshot", "live_check"] = "not_checked"
    source: SourceReference | None = None
    reason: Literal["not_checked", "not_listed", "expired", "reported"] = "not_checked"

    @model_validator(mode="after")
    def validate_evidence(self):
        if self.state != "unknown" and (self.evidence == "not_checked" or self.source is None):
            raise ValueError("Known data availability requires evidence and a source")
        return self


class ResearcherIssue(StrictModel):
    code: str
    severity: Literal["error", "warning", "question"] = "error"
    message: str
    field: str | None = None
    requested_value: str | None = None
    next_steps: list[str] = Field(default_factory=list)
    candidates: list[str] = Field(default_factory=list)
    source: SourceReference | None = None
    constraint: str | None = None
    # Correlates an unexpected failure with the full record in the protected log.
    incident_id: str | None = None


class AlgorithmSuggestion(StrictModel):
    algorithm_id: str
    display_name: str
    version: str
    capabilities: list[str]
    limitations: list[str]
    scope: Literal["declared_for_selection", "select_sites_first"]
    verification: Literal["declared", "sandbox_verified"]
    authorization: Literal["not_verified"] = "not_verified"
    data_availability: Literal["not_verified"] = "not_verified"
    replaces_original_request: Literal[False] = False
    catalog: CatalogReference | None = None


class AgentResponse(StrictModel):
    status: Literal["conversation", "ready", "needs_clarification", "unsupported", "invalid_request", "catalog_error", "planner_error", "algorithms", "data_unavailable", "permission_denied", "preflight_error", "execution_failed", "completed", "report_failed"]
    message: str
    # Local study state can retain a rejected proposal without echoing unchecked
    # parameter values into public error responses or narrator facts.
    research_plan: ResearchPlan | None = Field(default=None, exclude=True)
    catalog: CatalogReference | None = None
    catalog_document: dict[str, Any] | None = None
    catalog_content: str | None = Field(default=None, exclude=True)
    prepared_task: PreparedTask | None = None
    clarification: ClarificationRequest | None = None
    algorithms: list[dict[str, Any]] = Field(default_factory=list)
    issues: list[ResearcherIssue] = Field(default_factory=list)
    suggested_algorithms: list[AlgorithmSuggestion] = Field(default_factory=list)
    submission_state: Literal["not_submitted", "unknown", "submitted", "completed", "failed"] = "not_submitted"
    execution: "ExecutionRecord | None" = None
    report_status: Literal["not_requested", "pending", "completed", "failed"] = "not_requested"


class ExecutionStatus(str, Enum):
    submitted = "submitted"
    running = "running"
    completed = "completed"
    failed = "failed"


class ExecutionRecord(StrictModel):
    task_id: str
    status: ExecutionStatus
    result: Any | None = None
    error: str | None = None


class ParsedResult(StrictModel):
    numerical: list[dict[str, Any]] = Field(default_factory=list)
    categorical: list[dict[str, Any]] = Field(default_factory=list)
    numerical_present: bool = False
    categorical_present: bool = False
