"""Load the file-backed algorithm catalog: verified executable contracts with explicit input bindings."""

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError as SchemaValidationError
from pydantic import Field, model_validator

from .errors import (
    AmbiguousAlgorithmError, CatalogError, UnsupportedAnalysisError,
)
from .hashing import canonical_json_sha256
from .models import AlgorithmContract, CatalogReference, SourceReference, StrictModel


SUPPORTED_ADAPTERS = {"vantage6_kwargs_v1"}


def _check_schema(schema: dict) -> None:
    if schema.get("$schema", "https://json-schema.org/draft/2020-12/schema") != "https://json-schema.org/draft/2020-12/schema":
        raise ValueError("Catalog v1 uses JSON Schema draft 2020-12")
    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        raise ValueError("Input schemas must be closed objects (additionalProperties: false)")
    # Catalog validation never resolves network references or executes extensions.
    def check_references(value: Any) -> None:
        if isinstance(value, dict):
            if any(key in value for key in ("$ref", "$dynamicRef")):
                raise ValueError("Catalog v1 requires inline schemas, without $ref")
            for child in value.values():
                check_references(child)
        elif isinstance(value, list):
            for child in value:
                check_references(child)

    check_references(schema)
    Draft202012Validator.check_schema(schema)


class AlgorithmRegistryDocument(StrictModel):
    schema_version: Literal["1.0"]
    catalog_version: str = Field(min_length=1)
    updated_at: str = Field(min_length=1)
    description: str
    algorithms: list[AlgorithmContract]

    @model_validator(mode="after")
    def validate_contracts(self) -> "AlgorithmRegistryDocument":
        ids = [item.algorithm_id for item in self.algorithms]
        if len(ids) != len(set(ids)):
            raise ValueError("Algorithm IDs must be unique; keep one active version per ID")
        for item in self.algorithms:
            if not item.algorithm_id or not item.analysis_types or not item.description:
                raise ValueError("Each entry requires an ID, analysis types and description")
            if not item.availability or item.execution is None:
                raise ValueError(f"{item.algorithm_id}: availability and execution are required")
            if item.catalog_source is not None:
                raise ValueError("catalog_source is computed by the loader, not authored")
            if not item.image or not item.function or not item.algorithm_version:
                raise ValueError("An image, method and algorithm version are required")
            # Mutable latest tags cannot identify the implementation contract.
            final_segment = item.image.rsplit("/", 1)[-1]
            if ":" not in final_segment or final_segment.endswith(":latest"):
                raise ValueError("Use a versioned image tag or digest, never implicit/latest")
            execution = item.execution
            _check_schema(execution.parameters_schema)
            _check_schema(execution.kwargs_schema)
            parameters = execution.parameters_schema.get("properties", {})
            bindings = execution.input_bindings
            if set(bindings) != set(item.accepted_arguments):
                raise ValueError("Bindings must exactly match accepted_arguments")
            if len(item.accepted_arguments) != len(set(item.accepted_arguments)):
                raise ValueError("accepted_arguments must not contain duplicates")
            if execution.adapter == "vantage6_kwargs_v1":
                sources = [binding.source for binding in bindings.values()]
                if "variables_by_class" not in sources or sources.count("organisation_ids") != 1:
                    raise ValueError("vantage6_kwargs_v1 must bind variables and exactly one site selector")
                if item.supports_stratification:
                    raise ValueError("vantage6_kwargs_v1 does not implement stratification")
            if set(execution.kwargs_schema.get("properties", {})) != set(bindings):
                raise ValueError("kwargs_schema properties must exactly match bindings")
            parameter_bindings = {
                binding.parameter for binding in bindings.values()
                if binding.source == "parameter"
            }
            if parameter_bindings != set(parameters):
                raise ValueError("Every researcher parameter must be declared and bound")
            required = set(execution.parameters_schema.get("required", []))
            if not required <= set(parameters):
                raise ValueError("Required parameters must be declared")
            for name, schema in parameters.items():
                if not isinstance(schema, dict):
                    raise ValueError("Research parameter schemas must be objects")
                if "default" in schema:
                    Draft202012Validator(schema).validate(schema["default"])
                elif name not in required:
                    raise ValueError("Optional parameters need an explicit default in v1")
            deployments = [scope.deployment for scope in item.availability]
            if len(deployments) != len(set(deployments)):
                raise ValueError("Use one availability entry per deployment")
        return self


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"Non-finite JSON number: {value}")


class AlgorithmCatalog:
    def __init__(
        self, algorithms: list[AlgorithmContract], *,
        path: Path | None = None, reference: CatalogReference | None = None,
        document: dict | None = None,
        content: str | None = None,
    ) -> None:
        self.algorithms = algorithms
        self.path = path
        self.reference = reference
        self.document = document
        self.content = content

    @classmethod
    def from_json(cls, path: str | Path, *, content: bytes | None = None) -> "AlgorithmCatalog":
        path = Path(path).resolve()
        try:
            content = path.read_bytes() if content is None else content
            raw = json.loads(
                content.decode("utf-8-sig"), object_pairs_hook=_unique_object,
                parse_constant=_reject_nonfinite,
            )
            json.dumps(raw, allow_nan=False)
            document = AlgorithmRegistryDocument.model_validate(raw)
        except (OSError, ValueError, SchemaError, SchemaValidationError) as exc:
            raise CatalogError(f"Cannot load algorithm catalog {path}: {exc}") from exc
        reference = CatalogReference(
            path=str(path), schema_version=document.schema_version,
            catalog_version=document.catalog_version,
            sha256=hashlib.sha256(content).hexdigest(),
            content_sha256=canonical_json_sha256(raw),
        )
        source = SourceReference(
            kind="algorithm_registry", uri=str(path), version=document.catalog_version,
        )
        algorithms = [
            item.model_copy(update={"catalog_source": source}, deep=True)
            for item in document.algorithms
        ]
        return cls(algorithms, path=path, reference=reference, document=raw, content=content.decode("utf-8"))

    def snapshot(self) -> "AlgorithmCatalog":
        """Read once per turn; never fall back to a previous copy on an error."""
        return self.from_json(self.path) if self.path else self

    def knowledge(self) -> dict:
        """Public planning knowledge, without connection details or credentials."""
        return {
            "catalog": self.reference.model_dump(exclude={"path"}) if self.reference else None,
            "algorithms": [
                {
                    "algorithm_id": item.algorithm_id,
                    "display_name": item.display_name,
                    "version": item.algorithm_version,
                    "status": item.status,
                    "analysis_types": item.analysis_types,
                    "description": item.description,
                    "request_keywords": item.request_keywords,
                    "capabilities": item.capabilities,
                    "limitations": item.limitations,
                    "constraints": item.constraints.model_dump(),
                    "availability": [scope.model_dump() for scope in item.availability],
                    "adapter_supported": bool(
                        item.execution and item.execution.adapter in SUPPORTED_ADAPTERS
                    ),
                    "parameters_schema": (
                        item.execution.parameters_schema if item.execution else {}
                    ),
                    "source": item.source.model_dump(),
                }
                for item in self.algorithms
            ],
        }

    def select(
        self, analysis_type: str, algorithm_id: str | None = None,
    ) -> AlgorithmContract:
        matches = [
            algorithm
            for algorithm in self.algorithms
            if analysis_type in algorithm.analysis_types
            and algorithm.status == "enabled"
            and (algorithm_id is None or algorithm.algorithm_id == algorithm_id)
        ]
        if not matches:
            raise UnsupportedAnalysisError(
                "No approved algorithm currently supports this analysis. "
                "A new algorithm development and review workflow is required."
            )
        if len(matches) > 1:
            raise AmbiguousAlgorithmError([item.algorithm_id for item in matches])
        return matches[0].model_copy(deep=True)
