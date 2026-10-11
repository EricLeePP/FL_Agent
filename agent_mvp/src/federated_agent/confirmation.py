"""Bind researcher confirmation to the exact compiled task payload.

The material contains no file locations: catalog and metadata are identified by
content hashes, and code by the newline-normalized hash of the worker sources.
"""

import hashlib
import json

from .models import CatalogReference, Vantage6TaskSpec

CURRENT_FINGERPRINT_VERSION = 2


def _material(task_spec: Vantage6TaskSpec) -> dict:
    material = {
        "fingerprint_version": 2,
        "connection": task_spec.connection.model_dump(mode="json"),
        "client_payload": task_spec.to_client_payload(),
        "result_adapter": task_spec.result_adapter,
    }
    if task_spec.catalog:
        material["catalog"] = {"schema_version": task_spec.catalog.schema_version,
                               "catalog_version": task_spec.catalog.catalog_version,
                               "content_sha256": task_spec.catalog.content_sha256}
    if task_spec.requires_managed_execution:
        material["requires_managed_execution"] = True
    if task_spec.registry_release:
        material["registry_release"] = task_spec.registry_release.model_dump(mode="json")
    if task_spec.execution_template:
        material["execution_template"] = task_spec.execution_template.model_dump(mode="json")
    if task_spec.local_metadata_hashes:
        material["local_metadata_hashes"] = task_spec.local_metadata_hashes
    return material


def task_confirmation_material(task_spec: Vantage6TaskSpec) -> dict:
    if task_spec.fingerprint_version != CURRENT_FINGERPRINT_VERSION:
        raise ValueError(f"Unsupported confirmation fingerprint version {task_spec.fingerprint_version}; prepare the task again")
    return _material(task_spec)


def task_confirmation_id(task_spec: Vantage6TaskSpec) -> str:
    canonical = json.dumps(
        task_confirmation_material(task_spec),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def same_catalog(prepared: CatalogReference | None, current: CatalogReference | None) -> bool:
    """Compare catalogs by content when both references carry a content hash."""
    if prepared is None or current is None:
        return prepared == current
    if prepared.content_sha256 and current.content_sha256:
        return ((prepared.schema_version, prepared.catalog_version, prepared.content_sha256)
                == (current.schema_version, current.catalog_version, current.content_sha256))
    return prepared == current
