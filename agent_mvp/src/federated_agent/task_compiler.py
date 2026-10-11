"""Compile a resolved research plan into a versioned Vantage6 task."""

from copy import deepcopy
import json

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from .algorithm_catalog import SUPPORTED_ADAPTERS
from .confirmation import CURRENT_FINGERPRINT_VERSION
from .deployment import DeploymentRegistry
from .errors import TaskValidationError
from .models import (
    AlgorithmContract,
    DatabaseSpec,
    FieldProvenance,
    ResolvedResearchPlan,
    SourceReference,
    Vantage6Connection,
    Vantage6TaskSpec,
)


VANTAGE6_CLIENT_SOURCE = SourceReference(
    kind="vantage6_client_contract",
    uri="https://docs.vantage6.ai/en/version-4.15.0/user/pyclient.html",
    version="4.15.0",
)


class Vantage6TaskCompiler:
    def __init__(self, deployment_registry: DeploymentRegistry) -> None:
        self.deployment_registry = deployment_registry

    def compile(
        self,
        plan: ResolvedResearchPlan,
        algorithm: AlgorithmContract,
    ) -> Vantage6TaskSpec:
        profile = self.deployment_registry.profile
        self._validate_versions(profile.vantage6_version, algorithm.vantage6_version)
        label = plan.database_label
        if not label or not label.strip():
            raise TaskValidationError("A dataset must be explicitly selected before compilation", code="missing_database",
                                      field="database_label")
        minimum = algorithm.constraints.min_sites
        if any(variable.datatype == "numerical" for variable in plan.variables):
            minimum = max(minimum, algorithm.constraints.min_sites_for_numerical)
        if len({site.organisation_id for site in plan.sites}) < minimum:
            raise TaskValidationError("Registered site-count constraint not met", code="site_count_unsupported",
                                      field="sites", constraint=f"This variable/algorithm combination requires at least {minimum} participating sites. " + "; ".join(algorithm.limitations))
        required_ids = {profile.central_organisation_id, *(site.organisation_id for site in plan.sites)}
        covered_ids = {site.organisation_id for site in profile.sites if label in site.database_labels}
        if not required_ids <= covered_ids:
            raise TaskValidationError("Database not registered at all required sites", code="unknown_database",
                                      field="database_label", constraint=f"Requested database: {label}. Check the selected and central sites.")
        if plan.stratification_variables:
            raise TaskValidationError(
                "The approved algorithm contract does not yet support confirmed "
                "stratification ranges or category selections", code="unsupported_stratification",
            )

        kwargs = self._bind_inputs(plan, algorithm)
        unknown_arguments = set(kwargs) - set(algorithm.accepted_arguments)
        if unknown_arguments:
            raise TaskValidationError(
                f"Task contains unsupported arguments: {sorted(unknown_arguments)}", code="invalid_parameters",
            )

        input_ = {"method": algorithm.function, "kwargs": kwargs}
        provenance = self._provenance(plan, algorithm)
        warnings = list(algorithm.contract_warnings)
        if algorithm.execution:
            scope = next(scope for scope in algorithm.availability if scope.deployment == profile.name)
            warnings.append(f"Algorithm availability: {scope.verification}. {scope.evidence}")
        return Vantage6TaskSpec(
            algorithm_id=algorithm.algorithm_id,
            algorithm_version=algorithm.algorithm_version,
            image=algorithm.image,
            connection=Vantage6Connection(
                environment=profile.name,
                sandbox=profile.sandbox,
                version=profile.vantage6_version,
                server_url=profile.server_url,
                server_port=profile.server_port,
                server_api=profile.server_api,
            ),
            collaboration_id=profile.collaboration_id,
            central_organisation_id=profile.central_organisation_id,
            name=f"Agent-generated {algorithm.display_name}",
            description=plan.purpose,
            input=input_,
            databases=[DatabaseSpec(label=label)],
            provenance=provenance,
            warnings=warnings,
            result_adapter=algorithm.execution.result_adapter,
            fingerprint_version=CURRENT_FINGERPRINT_VERSION,
        )

    def _bind_inputs(self, plan: ResolvedResearchPlan, algorithm: AlgorithmContract) -> dict:
        execution = algorithm.execution
        if execution.adapter not in SUPPORTED_ADAPTERS:
            raise TaskValidationError(
                f"Algorithm is registered but requires a new adapter: {execution.adapter}", code="adapter_unavailable",
            )
        profile = self.deployment_registry.profile
        scopes = [scope for scope in algorithm.availability if scope.deployment == profile.name]
        if not scopes:
            raise TaskValidationError("Algorithm is not declared for this deployment", code="algorithm_deployment_unavailable")
        scope = scopes[0]
        required_sites = {profile.central_organisation_id, *[site.organisation_id for site in plan.sites]}
        if not required_sites <= set(scope.organisation_ids):
            missing = required_sites - set(scope.organisation_ids)
            names = [site.name for site in profile.sites if site.organisation_id in missing]
            if profile.central_organisation_id in missing:
                names.append("central execution organization")
            raise TaskValidationError("Algorithm is not declared for every selected/central organisation", code="algorithm_site_unavailable",
                                      field="sites", constraint="Not declared at: " + ", ".join(names))
        label = plan.database_label
        if label not in scope.database_labels:
            raise TaskValidationError("Algorithm is not declared for the selected database label", code="algorithm_database_unavailable",
                                      field="database_label", constraint=f"Not declared for: {label}")
        parameters = deepcopy(plan.parameters)
        for name, schema in execution.parameters_schema.get("properties", {}).items():
            if name not in parameters and "default" in schema:
                parameters[name] = deepcopy(schema["default"])
        self._validate_json(parameters, execution.parameters_schema, "Research parameters")
        kwargs = {}
        for name, binding in execution.input_bindings.items():
            if binding.source == "variables_by_class":
                value = {v.class_id: {"datatype": v.datatype} for v in plan.variables}
            elif binding.source == "organisation_ids":
                value = [site.organisation_id for site in plan.sites]
            elif binding.source == "parameter":
                value = parameters[binding.parameter]
            else:
                value = binding.value
            kwargs[name] = deepcopy(value)
        self._validate_json(kwargs, execution.kwargs_schema, "Algorithm kwargs")
        return kwargs

    @staticmethod
    def _validate_json(value: dict, schema: dict, label: str) -> None:
        try:
            json.dumps(value, allow_nan=False)
            Draft202012Validator(schema).validate(value)
        except ValidationError as exc:
            field = ".".join(str(part) for part in exc.absolute_path) or "parameters"
            constraint = f"{exc.validator}: {json.dumps(exc.validator_value, ensure_ascii=True)}"
            if exc.validator == "additionalProperties":
                constraint = "Allowed parameter names: " + (", ".join(exc.schema.get("properties", {})) or "none")
            raise TaskValidationError(f"{label}: {exc.message}", code="invalid_parameters", field=field, constraint=constraint) from exc
        except (ValueError, TypeError) as exc:
            raise TaskValidationError(f"{label} must contain finite JSON values", code="invalid_parameters") from exc

    def _organisation_argument(algorithm: AlgorithmContract) -> str:
        supported = [
            name
            for name in (
                "organisations_to_include",
                "organisation_ids",
                "organization_ids",
            )
            if name in algorithm.accepted_arguments
        ]
        if len(supported) != 1:
            raise TaskValidationError(
                "The algorithm contract must define exactly one organisation selector"
            )
        return supported[0]

    @staticmethod
    def _validate_versions(deployment_version: str, algorithm_version: str) -> None:
        def major_minor(value: str) -> tuple[str, str]:
            parts = value.split(".")
            if len(parts) < 2 or not all(part.isdigit() for part in parts[:2]):
                raise TaskValidationError(f"Invalid Vantage6 version: {value}")
            return parts[0], parts[1]

        if major_minor(deployment_version) != major_minor(algorithm_version):
            raise TaskValidationError(
                "Algorithm and deployment Vantage6 major/minor versions differ: "
                f"{algorithm_version} vs {deployment_version}", code="version_mismatch", field="vantage6_version",
                constraint=f"Algorithm requires {algorithm_version}; deployment declares {deployment_version}",
            )

    def _provenance(
        self,
        plan: ResolvedResearchPlan,
        algorithm: AlgorithmContract,
    ) -> list[FieldProvenance]:
        deployment_source = self.deployment_registry.source
        items = [
            FieldProvenance(field="algorithm_id", source=algorithm.source),
            FieldProvenance(field="algorithm_version", source=algorithm.source),
            FieldProvenance(field="image", source=algorithm.catalog_source or deployment_source),
            FieldProvenance(field="input.method", source=algorithm.source),
            FieldProvenance(field="input", source=VANTAGE6_CLIENT_SOURCE),
            FieldProvenance(field="collaboration_id", source=deployment_source),
            FieldProvenance(
                field="central_organisation_id", source=deployment_source
            ),
            FieldProvenance(field="connection", source=deployment_source),
            FieldProvenance(field="databases", source=SourceReference(kind="research_plan", uri="research_plan.database_label")),
        ]
        bindings = algorithm.execution.input_bindings
        variable_arguments = [name for name, binding in bindings.items() if binding.source == "variables_by_class"]
        for name, binding in bindings.items():
            source = algorithm.catalog_source or algorithm.source
            if binding.source == "organisation_ids":
                source = deployment_source
            elif binding.source == "parameter" and binding.parameter in plan.parameters:
                source = SourceReference(kind="research_plan", uri=f"research_plan.parameters.{binding.parameter}")
            items.append(FieldProvenance(field=f"input.kwargs.{name}", source=source))
        for name in variable_arguments:
            for variable in plan.variables:
                items.append(
                    FieldProvenance(field=f"input.kwargs.{name}.{variable.class_id}", source=variable.source)
                )
        if algorithm.metadata_source:
            items.append(
                FieldProvenance(
                    field="algorithm_store_metadata",
                    source=algorithm.metadata_source,
                )
            )
        return items
