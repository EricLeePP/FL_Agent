"""Immutable metadata snapshots: every operation reads one consistent set.

A snapshot holds the semantic catalog, deployment, algorithm catalog, execution
templates and data-availability checker that one request, preparation or
execution sees together. Capturing never changes an existing snapshot. The agent
replaces its current snapshot as a whole, so a task is never compiled against a
mixture of old and new files.
"""

from copy import copy
from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .algorithm_catalog import AlgorithmCatalog
from .capabilities import DataCapabilities, capabilities_of
from .data_availability import FileDataAvailabilityChecker, UnverifiedDataChecker
from .deployment import DeploymentRegistry
from .errors import CatalogError, PreflightError, TaskValidationError
from .execution_templates import TemplateRegistry
from .hashing import content_sha256
from .result_parser import ResultParser
from .semantic_catalog import SemanticCatalog
from .task_compiler import Vantage6TaskCompiler


@dataclass(frozen=True)
class MetadataSnapshot:
    semantic_catalog: SemanticCatalog
    deployment_registry: DeploymentRegistry
    algorithm_catalog: AlgorithmCatalog
    template_registry: TemplateRegistry = field(default_factory=TemplateRegistry)
    data_checker: Any = field(default_factory=UnverifiedDataChecker)
    registry_reference: Any = None
    local_metadata_hashes: Mapping[str, str] = field(default_factory=dict)
    captured: bool = False
    """True when the files were read together; otherwise the catalog is read on use."""

    @cached_property
    def task_compiler(self) -> Vantage6TaskCompiler:
        return Vantage6TaskCompiler(self.deployment_registry)

    @cached_property
    def result_parser(self) -> ResultParser:
        return ResultParser(self.semantic_catalog)

    def catalog(self) -> AlgorithmCatalog:
        """The algorithm catalog of this snapshot; never a copy from an earlier one."""
        return self.algorithm_catalog if self.captured else self.algorithm_catalog.snapshot()


def capture_local(configured: MetadataSnapshot) -> MetadataSnapshot:
    """Read every configured file once, validate them together and return a new snapshot.

    In-memory collaborators without a file are kept as configured.
    """
    paths = {"templates": configured.template_registry.path}
    if configured.semantic_catalog.path:
        paths["semantics"] = configured.semantic_catalog.path
    if configured.algorithm_catalog.path:
        paths["algorithms"] = configured.algorithm_catalog.path
    if configured.deployment_registry.source.kind == "deployment_registry":
        paths["deployment"] = Path(configured.deployment_registry.source.uri)
    if capabilities_of(configured.data_checker, DataCapabilities).inventory_file:
        paths["availability"] = configured.data_checker.path
    raw = {}
    try:
        for role, path in paths.items():
            try:
                raw[role] = Path(path).read_bytes()
            except OSError as exc:
                if role == "algorithms":
                    raise CatalogError("Cannot read algorithm catalog") from exc
                raise
        algorithms = (AlgorithmCatalog.from_json(paths["algorithms"], content=raw["algorithms"])
                      if "algorithms" in raw else configured.algorithm_catalog)
        semantic = (SemanticCatalog.from_jsonld(paths["semantics"], content=raw["semantics"])
                    if "semantics" in raw else configured.semantic_catalog)
        deployment = (DeploymentRegistry.from_json(paths["deployment"], content=raw["deployment"])
                      if "deployment" in raw else configured.deployment_registry)
        templates = TemplateRegistry(paths["templates"], content=raw["templates"])
        templates.load()
        checker = configured.data_checker
        if "availability" in raw:
            checker = copy(checker)
            checker.content = raw["availability"]
            if checker.load().deployment != deployment.profile.name:
                raise ValueError("Data inventory belongs to another deployment")
        # Individual files have no group commit. Reject a change during capture;
        # use signed registry releases when atomic multi-file publication is needed.
        if any(Path(path).read_bytes() != raw[role] for role, path in paths.items()):
            raise ValueError("Local metadata changed during capture")
        hashes = MappingProxyType({role: content_sha256(content) for role, content in raw.items()})
    except CatalogError:
        raise
    except (OSError, ValueError, KeyError, TypeError, TaskValidationError) as exc:
        raise PreflightError("Local metadata snapshot could not be validated") from exc
    return MetadataSnapshot(semantic, deployment, algorithms, templates, checker,
                            local_metadata_hashes=hashes, captured=True)


def release_snapshot(release, *, validate=True) -> MetadataSnapshot:
    """A snapshot of one verified registry release; never mixed with local files."""
    templates = TemplateRegistry(release.path("templates"))
    if validate:
        templates.load()
    return MetadataSnapshot(
        SemanticCatalog.from_jsonld(release.path("semantics")),
        DeploymentRegistry.from_json(release.path("deployment")),
        AlgorithmCatalog.from_json(release.path("algorithms")),
        templates, FileDataAvailabilityChecker(release.path("availability")),
        registry_reference=release.reference, captured=validate)


def with_fresh_deployment(snapshot: MetadataSnapshot) -> MetadataSnapshot:
    """Reread a file-backed deployment profile; everything else stays as captured."""
    registry = snapshot.deployment_registry
    if registry.source.kind != "deployment_registry":
        return snapshot
    try:
        fresh = DeploymentRegistry.from_json(registry.source.uri)
    except (OSError, ValueError) as exc:
        raise PreflightError("Deployment metadata unavailable") from exc
    return replace(snapshot, deployment_registry=fresh)
