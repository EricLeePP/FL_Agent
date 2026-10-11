"""Pluggable data preflight; metadata evidence is never an authorization grant."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol

from pydantic import Field, model_validator

from .capabilities import DataCapabilities
from .errors import PreflightError
from .models import DataAvailabilityCheck, ResolvedResearchPlan, SourceReference, StrictModel, Vantage6TaskSpec


class DataAvailabilityChecker(Protocol):
    def check(self, plan: ResolvedResearchPlan, task: Vantage6TaskSpec) -> list[DataAvailabilityCheck]:
        ...


class UnverifiedDataChecker:
    capabilities = DataCapabilities(verifies_availability=False)

    def check(self, plan, task):
        return [DataAvailabilityCheck(site=site.canonical_name, variable=variable.canonical_name, state="unknown")
                for site in plan.sites for variable in plan.variables]


class SiteDataDeclaration(StrictModel):
    organisation_id: int = Field(gt=0)
    database_label: str = Field(min_length=1)
    variables: dict[str, Literal["available", "unavailable", "unknown"]]
    complete_inventory: bool = False


class DataAvailabilityDocument(StrictModel):
    schema_version: Literal["1.0"]
    version: str = Field(min_length=1)
    deployment: str = Field(min_length=1)
    observed_at: datetime
    expires_at: datetime
    sites: list[SiteDataDeclaration]

    @model_validator(mode="after")
    def validate_evidence(self):
        keys = [(site.organisation_id, site.database_label) for site in self.sites]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate organization/database declarations")
        if self.observed_at.tzinfo is None or self.expires_at.tzinfo is None:
            raise ValueError("Evidence timestamps require explicit timezones")
        if self.expires_at <= self.observed_at:
            raise ValueError("Evidence expiry must follow observation")
        return self


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate metadata key")
        result[key] = value
    return result


class FileDataAvailabilityChecker:
    """Read an operator-maintained snapshot on every check, including execution."""

    capabilities = DataCapabilities(inventory_file=True)

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.content = None

    def load(self):
        return self._read_snapshot()[1]

    def _read_snapshot(self):
        # Keep the evidence hash and parsed declarations from the same bytes.
        try:
            raw = self.path.read_bytes() if self.content is None else self.content
            document = DataAvailabilityDocument.model_validate(json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_keys))
        except (OSError, ValueError) as exc:
            raise PreflightError("Data availability metadata could not be validated") from exc
        return raw, document

    def check(self, plan, task):
        raw, document = self._read_snapshot()
        if document.deployment != task.connection.environment:
            raise PreflightError("Data availability metadata belong to a different deployment")
        now = datetime.now(timezone.utc)
        if document.observed_at > now:
            raise PreflightError("Data availability evidence is dated in the future")
        source = SourceReference(kind="data_availability_snapshot", uri=str(self.path),
                                 version=f"{document.version}@{hashlib.sha256(raw).hexdigest()}")
        entries = {(site.organisation_id, site.database_label): site for site in document.sites}
        checks = []
        for site in plan.sites:
            entry = entries.get((site.organisation_id, site.database_label))
            for variable in plan.variables:
                state = entry.variables.get(variable.class_id, "unavailable" if entry.complete_inventory else "unknown") if entry else "unknown"
                reason = "reported" if entry and (variable.class_id in entry.variables or entry.complete_inventory) else "not_listed"
                if document.expires_at <= now:
                    state = "unknown"
                    reason = "expired"
                checks.append(DataAvailabilityCheck(site=site.canonical_name, variable=variable.canonical_name,
                              state=state, evidence="metadata_snapshot", source=source, reason=reason))
        return checks
