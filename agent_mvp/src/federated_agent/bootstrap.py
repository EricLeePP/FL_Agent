"""Resolve command-line flags, environment variables and an optional configuration
file into one read-only AppConfig.

This is the only module that decides where configuration and default metadata
come from. Adapters and services receive already-resolved values.
"""

import os
from pathlib import Path

from pydantic import ValidationError

from .config import AppConfig, FedAgentSettings, LLMConfig
from .resources import (
    ALGORITHM_CATALOG, DEFAULT_POLICY, DEMO_DEPLOYMENT, SANDBOX_DATA_AVAILABILITY, SANDBOX_DEPLOYMENT,
    SEMANTIC_SCHEMA,
)
from .trusted_http import strict_json

CONFIG_ENV = "FED_AGENT_CONFIG"
DEEPSEEK_ENDPOINT = "https://api.deepseek.com/chat/completions"


class ConfigurationError(ValueError):
    """The configuration file or flag combination cannot be used."""


def load_settings(path: str | Path | None = None) -> tuple[FedAgentSettings, Path | None]:
    """Read the configuration file named by the argument or FED_AGENT_CONFIG, if any."""
    if path is None and os.environ.get(CONFIG_ENV):
        path = os.environ[CONFIG_ENV]
    if path is None:
        return FedAgentSettings(), None
    source = Path(path).expanduser().resolve()
    try:
        raw = strict_json(source.read_bytes())
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f"Cannot read configuration file {source}: {exc}") from None
    try:
        return FedAgentSettings.model_validate(raw), source
    except ValidationError as exc:
        fields = ", ".join(".".join(str(part) for part in error["loc"]) for error in exc.errors())
        raise ConfigurationError(f"Configuration file {source} has invalid fields: {fields}") from None


def _from_file(value: str | None, source: Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute() and source is not None:
        path = source.parent / path
    return path.resolve()


def _first(*values):
    return next((value for value in values if value is not None), None)


def build_app_config(args, settings: FedAgentSettings | None = None, source: Path | None = None) -> AppConfig:
    """Merge explicit flags (non-None on args) over the environment, file and defaults."""
    settings = settings or FedAgentSettings()
    meta, llm_file, backend_file = settings.metadata, settings.llm, settings.backend
    flag = lambda name: getattr(args, name, None)  # noqa: E731

    backend = _first(flag("backend"), backend_file.kind, "mock")
    catalog = _first(flag("catalog"), _from_file(meta.algorithm_catalog, source))
    deployment = _first(flag("deployment_config"), _from_file(meta.deployment, source))
    availability = _first(flag("data_availability_config"), _from_file(meta.data_availability, source))
    templates = _first(flag("template_registry"), _from_file(meta.execution_templates, source))
    registry = _first(flag("registry_source"), _from_file(meta.registry_source, source))
    policy = _first(flag("policy_config"), _from_file(settings.policy.config, source))
    if registry is not None and any(value is not None for value in (catalog, deployment, availability, templates)):
        raise ConfigurationError("--registry-source owns all metadata; do not mix it with individual registry overrides")

    timeout = _first(flag("timeout_seconds"), backend_file.timeout_seconds, 600)
    if timeout <= 0:
        raise ConfigurationError("--timeout-seconds must be positive")
    if backend == "vantage6":
        deployment = deployment or SANDBOX_DEPLOYMENT
        availability = availability or SANDBOX_DATA_AVAILABILITY
        policy = policy or DEFAULT_POLICY

    provider = _first(flag("provider"), os.environ.get("LLM_PROVIDER"), llm_file.provider, "demo")
    file_endpoint = llm_file.endpoint
    llm = LLMConfig(
        provider=provider,
        runtime=_first(flag("agent_runtime"), llm_file.runtime, "json"),
        model=_first(flag("model"), llm_file.model),
        local_endpoint=_first(flag("llm_endpoint"), os.environ.get("LOCAL_LLM_ENDPOINT"),
                              file_endpoint if provider == "local" else None),
        deepseek_endpoint=_first(flag("deepseek_endpoint"), os.environ.get("DEEPSEEK_ENDPOINT"),
                                 file_endpoint if provider == "deepseek" else None, DEEPSEEK_ENDPOINT),
        local_api_key_env=_first(flag("api_key_env"), llm_file.api_key_env if provider == "local" else None,
                                 "LOCAL_LLM_API_KEY"),
        deepseek_api_key_env=_first(flag("deepseek_api_key_env"),
                                    llm_file.api_key_env if provider == "deepseek" else None, "DEEPSEEK_API_KEY"),
        deepseek_api_key_file=_first(flag("deepseek_api_key_file"), _from_file(llm_file.api_key_file, source)),
        stream=_first(flag("stream"), llm_file.stream, True),
        tool_lookups=llm_file.tool_lookups,
        max_turns=llm_file.max_turns,
    )
    return AppConfig(
        output=Path(_first(flag("output"), _from_file(settings.output, source), Path("output"))),
        backend=backend,
        confirm_synthetic_unencrypted_sandbox=bool(flag("confirm_synthetic_unencrypted_sandbox")
                                                   or backend_file.confirm_synthetic_unencrypted_sandbox),
        timeout_seconds=timeout,
        algorithm_catalog=catalog or ALGORITHM_CATALOG,
        semantic_schema=_from_file(meta.semantic_schema, source) or SEMANTIC_SCHEMA,
        deployment=deployment or DEMO_DEPLOYMENT,
        data_availability=availability,
        execution_templates=templates,
        registry_source=registry,
        policy_config=policy,
        llm=llm,
        verbosity=max(flag("verbose") or 0, settings.logging.verbosity),
        log_file=_first(flag("log_file"), _from_file(settings.logging.file, source)),
        source=source,
    )


def apply_to_args(args, config: AppConfig) -> None:
    """Write the effective values back for command-line code that still reads ``args``."""
    args.output = config.output
    args.backend = config.backend
    args.confirm_synthetic_unencrypted_sandbox = config.confirm_synthetic_unencrypted_sandbox
    args.timeout_seconds = config.timeout_seconds
    args.catalog = config.algorithm_catalog
    args.deployment_config = config.deployment
    args.data_availability_config = config.data_availability
    args.template_registry = config.execution_templates
    args.registry_source = config.registry_source
    args.policy_config = config.policy_config
    args.provider = config.llm.provider
    args.agent_runtime = config.llm.runtime
    args.model = config.llm.model
    args.llm_endpoint = config.llm.local_endpoint
    args.deepseek_endpoint = config.llm.deepseek_endpoint
    args.api_key_env = config.llm.local_api_key_env
    args.deepseek_api_key_env = config.llm.deepseek_api_key_env
    args.deepseek_api_key_file = config.llm.deepseek_api_key_file
    args.stream = config.llm.stream
