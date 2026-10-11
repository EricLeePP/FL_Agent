"""Read the STRONG AYA JSON-LD schema and resolve research variables."""

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .errors import AmbiguousVariableError, TaskValidationError, VariableNotFoundError
from .models import CatalogVariable, ResolvedVariable, SourceReference


DEFAULT_ALIASES = {
    "age_at_initial_diagnosis": [
        "age at diagnosis",
        "diagnosis age",
        "initial diagnosis age",
    ],
    "survival_status": ["vital status", "patient survival status"],
    "survival_event_time": ["survival time", "time to survival event"],
}


def normalize_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


class SemanticCatalog:
    def __init__(
        self,
        variables: list[CatalogVariable],
        source: SourceReference,
        path: Path | None = None,
    ) -> None:
        self.variables = variables
        self.source = source
        self.path = path
        self._by_class = {item.class_id: item for item in variables}

    @classmethod
    def from_jsonld(cls, path: str | Path, *, content: bytes | None = None) -> "SemanticCatalog":
        path = Path(path).resolve()
        raw = json.loads((path.read_bytes() if content is None else content).decode("utf-8-sig"))
        try:
            schema_variables: dict[str, dict[str, Any]] = raw["schema"]["variables"]
        except (KeyError, TypeError) as exc:
            raise TaskValidationError(
                "Schema must contain a schema.variables object"
            ) from exc
        if not isinstance(schema_variables, dict) or not all(isinstance(value, dict) for value in schema_variables.values()):
            raise TaskValidationError("Schema variables must map names to definitions")

        variables = []
        for canonical_name, definition in schema_variables.items():
            class_id = definition.get("class")
            schema_datatype = definition.get("dataType")
            if not class_id or not schema_datatype:
                continue
            algorithm_datatype = cls._map_datatype(
                schema_datatype, definition.get("@type", "")
            )
            aliases = [canonical_name.replace("_", " ")]
            aliases.extend(DEFAULT_ALIASES.get(canonical_name, []))
            variables.append(
                CatalogVariable(
                    canonical_name=canonical_name,
                    class_id=class_id,
                    schema_datatype=schema_datatype,
                    algorithm_datatype=algorithm_datatype,
                    aliases=list(dict.fromkeys(aliases)),
                )
            )
        return cls(
            variables,
            source=SourceReference(
                kind="semantic_schema",
                uri=raw.get("@id") or str(Path(path).resolve()),
                version=raw.get("version"),
            ),
            path=path,
        )

    @staticmethod
    def _map_datatype(schema_datatype: str, schema_type: str) -> str | None:
        if schema_datatype == "continuous":
            return "numerical"
        if schema_datatype == "categorical":
            return "categorical"
        if schema_datatype == "standardised" and "Categorical" in schema_type:
            return "categorical"
        return None

    def resolve(self, query: str) -> ResolvedVariable:
        normalized_query = normalize_text(query)
        exact = [
            item
            for item in self.variables
            if normalized_query
            in {
                normalize_text(item.canonical_name),
                *(normalize_text(alias) for alias in item.aliases),
            }
        ]
        if len(exact) == 1:
            return self._to_resolved(query, exact[0])
        if len(exact) > 1:
            raise AmbiguousVariableError(query, [item.canonical_name for item in exact])

        all_scores: list[tuple[float, CatalogVariable]] = []
        for item in self.variables:
            names = [item.canonical_name, *item.aliases]
            score = max(
                SequenceMatcher(None, normalized_query, normalize_text(name)).ratio()
                for name in names
            )
            all_scores.append((score, item))
        all_scores.sort(key=lambda pair: pair[0], reverse=True)
        scored = [pair for pair in all_scores if pair[0] >= 0.72]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        if not scored:
            candidates = [
                item.canonical_name for score, item in all_scores[:5] if score >= 0.45
            ]
            raise VariableNotFoundError(query, candidates)
        if len(scored) > 1 and scored[0][0] - scored[1][0] < 0.05:
            raise AmbiguousVariableError(
                query, [item.canonical_name for _, item in scored[:5]]
            )
        # A similar name is a suggestion, not evidence of semantic equivalence.
        raise AmbiguousVariableError(query, [item.canonical_name for _, item in scored[:5]])

    def get_by_class(self, class_id: str) -> CatalogVariable | None:
        return self._by_class.get(class_id)

    def _to_resolved(self, query: str, item: CatalogVariable) -> ResolvedVariable:
        if item.algorithm_datatype is None:
            raise TaskValidationError(
                f"Variable '{item.canonical_name}' has unsupported datatype "
                f"'{item.schema_datatype}'", code="unsupported_variable_type", field=item.canonical_name,
            )
        return ResolvedVariable(
            requested_name=query,
            canonical_name=item.canonical_name,
            class_id=item.class_id,
            datatype=item.algorithm_datatype,
            source=self.source,
        )
