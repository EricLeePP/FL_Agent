"""Configuration file schema and the resolved, read-only runtime configuration.

A configuration file is optional. Command-line flags override environment
variables, which override the file, which overrides the bundled defaults.
Secrets never appear here: the file names environment variables or key files.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import Field

from .models import StrictModel


class MetadataSettings(StrictModel):
    algorithm_catalog: str | None = None
    semantic_schema: str | None = None
    deployment: str | None = None
    data_availability: str | None = None
    execution_templates: str | None = None
    registry_source: str | None = None


class PolicySettings(StrictModel):
    config: str | None = None


class LLMSettings(StrictModel):
    provider: Literal["demo", "deepseek", "local"] | None = None
    runtime: Literal["sdk", "json"] | None = None
    model: str | None = None
    endpoint: str | None = None
    api_key_env: str | None = Field(default=None, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    api_key_file: str | None = None
    stream: bool | None = None
    tool_lookups: bool = False
    max_turns: int = Field(default=6, ge=2, le=20)


class BackendSettings(StrictModel):
    kind: Literal["mock", "vantage6"] | None = None
    confirm_synthetic_unencrypted_sandbox: bool = False
    timeout_seconds: int | None = Field(default=None, gt=0)


class LoggingSettings(StrictModel):
    verbosity: int = Field(default=0, ge=0, le=2)
    file: str | None = None


class FedAgentSettings(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    output: str | None = None
    metadata: MetadataSettings = Field(default_factory=MetadataSettings)
    policy: PolicySettings = Field(default_factory=PolicySettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    backend: BackendSettings = Field(default_factory=BackendSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "demo"
    runtime: str = "json"
    model: str | None = None
    local_endpoint: str | None = None
    deepseek_endpoint: str | None = None
    local_api_key_env: str = "LOCAL_LLM_API_KEY"
    deepseek_api_key_env: str = "DEEPSEEK_API_KEY"
    deepseek_api_key_file: Path | None = None
    stream: bool = True
    tool_lookups: bool = False
    max_turns: int = 6


@dataclass(frozen=True)
class AppConfig:
    """Everything the application needs, resolved once at startup and never mutated."""

    output: Path
    backend: str
    confirm_synthetic_unencrypted_sandbox: bool
    timeout_seconds: int
    algorithm_catalog: Path
    semantic_schema: Path
    deployment: Path
    data_availability: Path | None = None
    execution_templates: Path | None = None
    registry_source: Path | None = None
    policy_config: Path | None = None
    llm: LLMConfig = field(default_factory=LLMConfig)
    verbosity: int = 0
    log_file: Path | None = None
    source: Path | None = None

    @property
    def managed_execution(self) -> bool:
        """New submissions need fresh data/policy checks when either source is configured."""
        return self.policy_config is not None or self.data_availability is not None
