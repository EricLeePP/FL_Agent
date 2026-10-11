"""Versioned descriptors bind installed code/assets, not remote executable text."""
from pathlib import Path
from typing import Literal
from pydantic import Field, model_validator
from .models import StrictModel, ExecutionTemplateReference
from .runtime_adapters import runtime_adapter
from .trusted_http import strict_json
from .errors import PreflightError
from .resources import EXECUTION_TEMPLATES
from .hashing import files_sha256, normalized_text_sha256

ASSETS = Path(__file__).parent / "templates"
DEFAULT_REGISTRY = EXECUTION_TEMPLATES


class TemplateDescriptor(StrictModel):
    id: str = Field(min_length=1)
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    runtime_adapter: str
    export_asset: str = Field(pattern=r"^[A-Za-z0-9_-]+\.py\.tpl$")
    vantage6_version: Literal["4.15.0"]
    poll_interval_seconds: float = Field(default=3, gt=0, le=60)


class TemplateDocument(StrictModel):
    schema_version: Literal["1.0"]
    defaults: dict[str, str]
    templates: list[TemplateDescriptor]

    @model_validator(mode="after")
    def validate_defaults(self):
        keys = [(t.id, t.version) for t in self.templates]
        if len(set(keys)) != len(keys) or any(item not in keys for item in self.defaults.items()):
            raise ValueError("Duplicate templates or unknown default versions")
        return self


def asset_path(name):
    path = (ASSETS / name).resolve()
    if path.parent != ASSETS.resolve() or not path.is_file():
        raise ValueError("Template asset is not installed")
    return path


# The code that runs in the worker process. Changes elsewhere do not affect confirmations.
WORKER_SOURCES = ("runs/worker.py", "runs/submission.py", "runtime_adapters.py", "adapters/vantage6.py")
DIGEST_VERSION = 2


def adapter_digest():
    base = Path(__file__).parent
    return files_sha256(base / name for name in WORKER_SOURCES)


def template_digest(asset):
    return normalized_text_sha256(asset_path(asset).read_text(encoding="utf-8"))


def bind_descriptor(descriptor):
    runtime_adapter(descriptor.runtime_adapter)
    return ExecutionTemplateReference(**descriptor.model_dump(), export_sha256=template_digest(descriptor.export_asset),
                                      adapter_sha256=adapter_digest(), digest_version=DIGEST_VERSION)


def validate_installed(reference):
    if reference.digest_version != DIGEST_VERSION:
        raise ValueError("The task was prepared with an earlier worker version; prepare it again")
    expected = bind_descriptor(TemplateDescriptor.model_validate(reference.model_dump(
        exclude={"export_sha256", "adapter_sha256", "digest_version"})))
    if expected != reference:
        raise ValueError("Installed template code differs from the reviewed task")
    return runtime_adapter(reference.runtime_adapter)


class TemplateRegistry:
    def __init__(self, path=DEFAULT_REGISTRY, *, content=None):
        self.path = Path(path).resolve()
        self.content = content

    def load(self):
        document = TemplateDocument.model_validate(strict_json(self.path.read_bytes() if self.content is None else self.content))
        for descriptor in document.templates:
            bind_descriptor(descriptor)
        return document

    def select(self, template_id="joshi_vantage6_userclient", version=None):
        try:
            document = self.load()
            version = version or document.defaults[template_id]
            descriptor = next(t for t in document.templates if (t.id, t.version) == (template_id, version))
            return bind_descriptor(descriptor)
        except (OSError, ValueError, KeyError, StopIteration):
            raise PreflightError("Execution template unavailable or incompatible") from None
