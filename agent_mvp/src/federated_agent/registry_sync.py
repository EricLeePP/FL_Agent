"""Signed, coherent metadata releases; atomic activation with anti-rollback checks."""

import base64
import hashlib
import os
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4
from pydantic import Field

from .models import StrictModel, RegistryReference, DeploymentProfile
from .algorithm_catalog import AlgorithmCatalog
from .data_availability import DataAvailabilityDocument
from .semantic_catalog import SemanticCatalog
from .execution_templates import TemplateRegistry
from .errors import PreflightError
from .trusted_http import fetch, strict_json, validate_url

ROLES = {"algorithms": "algorithms.json", "deployment": "deployment.json",
         "availability": "availability.json", "semantics": "semantics.jsonld", "templates": "templates.json"}


def canonical(value):
    import json
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


class Release(StrictModel):
    schema_version: Literal["1.0"]
    release_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,100}$")
    sequence: int = Field(gt=0, strict=True)
    deployment: str
    created_at: datetime
    expires_at: datetime
    documents: dict[str, dict]


class Envelope(StrictModel):
    release: Release
    signature: str | None = None


class LocalConnection(StrictModel):
    server_url: str
    server_port: int = Field(gt=0, le=65535)
    server_api: str
    collaboration_id: int = Field(gt=0)


class RegistrySource(StrictModel):
    schema_version: Literal["1.0"]
    source: str
    deployment: str
    cache_dir: str
    connection: LocalConnection
    trusted_public_key: str | None = None
    allow_unsigned_local: bool = False
    token_env: str | None = None
    timeout_seconds: float = Field(default=5, gt=0, le=30)


def verify_envelope(raw, config, *, existing=False):
    envelope = Envelope.model_validate(strict_json(raw))
    release = envelope.release
    now = datetime.now(timezone.utc)
    if (release.created_at.tzinfo is None or release.expires_at.tzinfo is None
            or release.created_at >= release.expires_at or release.created_at > now):
        raise ValueError("Invalid release timestamps")
    if not existing and release.expires_at <= now:
        raise ValueError("Release expired")
    if set(release.documents) != set(ROLES) or release.deployment != config.deployment:
        raise ValueError("Wrong release roles or deployment")
    inventory = DataAvailabilityDocument.model_validate(release.documents["availability"])
    if release.expires_at > inventory.expires_at:
        raise ValueError("Release validity cannot outlive its data inventory")
    # Sign the original JSON value, not a model's reformatted datetime string.
    signed = strict_json(raw)["release"]
    digest = hashlib.sha256(canonical(signed)).hexdigest()
    if config.trusted_public_key:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        key = Ed25519PublicKey.from_public_bytes(base64.b64decode(config.trusted_public_key, validate=True))
        key.verify(base64.b64decode(envelope.signature or "", validate=True), canonical(signed))
    else:
        is_remote = config.source.startswith(("http://", "https://"))
        local = not is_remote or validate_url(config.source).hostname in {"127.0.0.1", "localhost", "::1"}
        if not config.allow_unsigned_local or not local or envelope.signature is not None:
            raise ValueError("A pinned publisher key is required")
    return release, RegistryReference(release_id=release.release_id, sequence=release.sequence, sha256=digest)


def validate_documents(directory, config):
    profile = DeploymentProfile.model_validate(strict_json((directory / ROLES["deployment"]).read_bytes()))
    if profile.name != config.deployment or profile.collaboration_id != config.connection.collaboration_id:
        raise ValueError("Wrong deployment/collaboration")
    algorithms = AlgorithmCatalog.from_json(directory / ROLES["algorithms"])
    semantics = SemanticCatalog.from_jsonld(directory / ROLES["semantics"])
    classes = {v.class_id for v in semantics.variables}
    inventory = DataAvailabilityDocument.model_validate(strict_json((directory / ROLES["availability"]).read_bytes()))
    if inventory.deployment != profile.name or inventory.observed_at > datetime.now(timezone.utc):
        raise ValueError("Wrong or future-dated data inventory")
    sites = {s.organisation_id: set(s.database_labels) for s in profile.sites}
    for entry in inventory.sites:
        if entry.database_label not in sites.get(entry.organisation_id, set()) or not set(entry.variables) <= classes:
            raise ValueError("Inventory references unknown hospitals, datasets or semantic classes")
    templates = TemplateRegistry(directory / ROLES["templates"])
    templates.load()
    for algorithm in algorithms.algorithms:
        for scope in algorithm.availability:
            if scope.deployment == profile.name:
                if any(org not in sites or not set(scope.database_labels) <= sites[org] for org in scope.organisation_ids):
                    raise ValueError("Algorithm scope references unknown hospitals or datasets")
        if algorithm.execution:
            templates.select(algorithm.execution.template_id, algorithm.execution.template_version)
    return profile


class RegistrySnapshot:
    def __init__(self, directory, reference):
        self.directory, self.reference = directory, reference

    def path(self, role):
        return self.directory / ROLES[role]


class RegistryClient:
    def __init__(self, config_path):
        self.config_path = Path(config_path).resolve()

    def sync(self, *, existing=False, reference=None):
        from .storage import write_json
        from .storage import FileLease
        try:
            config = RegistrySource.model_validate(strict_json(self.config_path.read_bytes()))
            cache = (self.config_path.parent / config.cache_dir).resolve()
            cache.mkdir(parents=True, exist_ok=True)
            authority = hashlib.sha256(canonical(config.model_dump())).hexdigest()
            with closing(FileLease(cache / ".registry.lock")):
                pointer = cache / "current.json"
                previous = strict_json(pointer.read_bytes()) if pointer.exists() else None
                if previous and previous["authority"] != authority:
                    raise ValueError("Use a separate cache for each registry authority and local route")
                if existing:
                    if not previous:
                        raise ValueError("No verified registry cache for recovery")
                    target = reference or RegistryReference.model_validate({key: previous[key] for key in
                        ("release_id", "sequence", "sha256")})
                    target = RegistryReference.model_validate(target)
                    raw = (cache / "releases" / target.sha256 / "release.json").read_bytes()
                elif config.source.startswith(("https://", "http://")):
                    token = os.environ.get(config.token_env) if config.token_env else None
                    if config.token_env and not token:
                        raise ValueError("Registry token unavailable")
                    raw = fetch(config.source, token=token, timeout=config.timeout_seconds)
                else:
                    if "://" in config.source:
                        raise ValueError("Unsupported registry transport")
                    path = (self.config_path.parent / config.source).resolve()
                    with path.open("rb") as handle:
                        raw = handle.read(4 * 1024 * 1024 + 1)
                if len(raw) > 4 * 1024 * 1024:
                    raise ValueError("Registry release too large")
                release, verified_reference = verify_envelope(raw, config, existing=existing)
                if existing and (verified_reference != target or target.sequence > previous["sequence"]):
                    raise ValueError("Historical release does not match the recorded reference")
                reference = verified_reference
                if not existing and previous and (reference.sequence < previous["sequence"] or (
                    reference.sequence == previous["sequence"] and reference.sha256 != previous["sha256"]
                )):
                    raise ValueError("Registry rollback or same-version mutation rejected")
                directory = cache / "releases" / reference.sha256
                documents = dict(release.documents)
                documents["deployment"] = {**documents["deployment"], **config.connection.model_dump()}
                if directory.exists():
                    for role, filename in ROLES.items():
                        if (directory / filename).read_bytes() != canonical(documents[role]):
                            raise ValueError("Cached release content changed")
                    validate_documents(directory, config)
                else:
                    staging = cache / ("staging-" + uuid4().hex)
                    staging.mkdir()
                    for role, filename in ROLES.items():
                        (staging / filename).write_bytes(canonical(documents[role]))
                    (staging / "release.json").write_bytes(raw)
                    validate_documents(staging, config)
                    directory.parent.mkdir(exist_ok=True)
                    staging.rename(directory)
                if not existing:
                    write_json(pointer, {**reference.model_dump(), "authority": authority})
                return RegistrySnapshot(directory, reference)
        except Exception:
            # Keep the old cache for explicit historical retrieval, but never
            # silently authorize new work with it after a failed refresh.
            raise PreflightError("Registry synchronization failed; no new task may use stale metadata") from None
