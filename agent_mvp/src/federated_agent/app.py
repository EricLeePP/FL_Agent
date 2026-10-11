"""Composition root: build the application once from a resolved AppConfig."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import os

from .algorithm_catalog import AlgorithmCatalog
from .bootstrap import ConfigurationError
from .config import AppConfig, LLMConfig
from .credentials import CredentialError, load_deepseek_api_key
from .data_availability import FileDataAvailabilityChecker, UnverifiedDataChecker
from .deployment import DeploymentRegistry
from .errors import CatalogError, PreflightError
from .execution_templates import TemplateRegistry
from .executor import MockVantage6Executor
from .metadata import MetadataSnapshot, release_snapshot
from .planner import DeepSeekJSONProvider, DemoResearchPlanner, LLMResearchPlanner, OpenAICompatibleJSONProvider
from .policy import AllowAllPolicyChecker, FilePolicyChecker, PolicyConfiguration
from .registry_sync import RegistryClient
from .result_adapters import installed as installed_result_adapters
from .resources import ALGORITHM_CATALOG, DEMO_DEPLOYMENT, SEMANTIC_SCHEMA
from .semantic_catalog import SemanticCatalog
from .service import FederatedResearchAgent
from .trusted_http import strict_json


def build_agent(
    deployment_config: str | Path = DEMO_DEPLOYMENT,
    algorithm_catalog_path: str | Path = ALGORITHM_CATALOG,
    data_availability_path: str | Path | None = None,
    *, policy_config=None, registry_source=None, template_registry=None, registry_recovery=False,
    semantic_schema_path: str | Path = SEMANTIC_SCHEMA,
) -> FederatedResearchAgent:
    """Assemble the research agent from metadata locations."""
    metadata_client = RegistryClient(registry_source) if registry_source else None
    if metadata_client:
        # Every operation syncs again; this first release only fixes the starting point.
        metadata = release_snapshot(metadata_client.sync(existing=registry_recovery), validate=False)
    else:
        metadata = MetadataSnapshot(
            semantic_catalog=SemanticCatalog.from_jsonld(semantic_schema_path),
            deployment_registry=DeploymentRegistry.from_json(deployment_config),
            algorithm_catalog=AlgorithmCatalog.from_json(algorithm_catalog_path),
            template_registry=TemplateRegistry(template_registry) if template_registry else TemplateRegistry(),
            data_checker=(FileDataAvailabilityChecker(data_availability_path) if data_availability_path
                          else UnverifiedDataChecker()),
        )
    return FederatedResearchAgent(
        metadata,
        policy_checker=FilePolicyChecker(policy_config) if policy_config else AllowAllPolicyChecker(),
        executor=MockVantage6Executor(),
        metadata_client=metadata_client,
    )


def build_planner(llm: LLMConfig, *, interactive=False, ask_secret=None):
    """Build the research planner for ``llm``; ConfigurationError explains an unusable setup.

    The model runtime is the only difference between "json" (JSON mode over plain
    HTTP) and "sdk" (the Agents SDK with a proposal tool): both use the same prompts
    and the same validators. ``ask_secret`` may prompt for a DeepSeek key in an
    interactive session; keys never come from researcher chat text.
    """
    if llm.provider == "demo":
        return DemoResearchPlanner()
    if llm.provider == "deepseek":
        try:
            api_key = load_deepseek_api_key(llm.deepseek_api_key_file, llm.deepseek_api_key_env)
        except CredentialError as exc:
            raise ConfigurationError(str(exc)) from None
        if not api_key and interactive and ask_secret is not None:
            api_key = ask_secret("DeepSeek API key (local input, not saved): ")
        if not api_key:
            raise ConfigurationError(f"Supply --deepseek-api-key-file or set {llm.deepseek_api_key_env} "
                                     "before using --provider deepseek")
        provider = DeepSeekJSONProvider(api_key=api_key, endpoint=llm.deepseek_endpoint,
                                        model=llm.model or os.environ.get("DEEPSEEK_MODEL") or DeepSeekJSONProvider.DEFAULT_MODEL)
    else:
        if not llm.local_endpoint:
            raise ConfigurationError("--llm-endpoint is required for --provider local")
        model = llm.model or os.environ.get("LOCAL_LLM_MODEL")
        if not model:
            raise ConfigurationError("--model is required for --provider local")
        provider = OpenAICompatibleJSONProvider(endpoint=llm.local_endpoint, model=model,
                                                api_key=os.environ.get(llm.local_api_key_env))
    if llm.tool_lookups and llm.runtime != "sdk":
        raise ConfigurationError("llm.tool_lookups needs the sdk runtime (--agent-runtime sdk)")
    if llm.runtime == "sdk":
        try:
            from .sdk_runtime import AgentsSDKProvider
        except ImportError:
            raise ConfigurationError("SDK runtime is not installed. Install the project's [sdk] extra or "
                                     "explicitly use --agent-runtime json.") from None
        try:
            provider = AgentsSDKProvider.from_provider(provider)
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from None
    return LLMResearchPlanner(provider, tool_lookups=llm.tool_lookups, max_turns=llm.max_turns)


@dataclass
class FedAgentApp:
    config: AppConfig
    agent: FederatedResearchAgent
    issues: list[str] = field(default_factory=list)


def build_application(config: AppConfig, *, registry_recovery: bool = False) -> FedAgentApp:
    agent = build_agent(
        config.deployment, config.algorithm_catalog, config.data_availability,
        policy_config=config.policy_config, registry_source=config.registry_source,
        template_registry=config.execution_templates, registry_recovery=registry_recovery,
        semantic_schema_path=config.semantic_schema,
    )
    return FedAgentApp(config=config, agent=agent)


def package_version() -> str:
    try:
        return version("federated-research-agent")
    except PackageNotFoundError:
        return "unknown"


def describe(config: AppConfig) -> dict:
    """A credential-free summary of the effective configuration and its health.

    It loads metadata independently so that problems are reported, not raised.
    """
    issues = []
    algorithms = []
    try:
        catalog = AlgorithmCatalog.from_json(config.algorithm_catalog)
        algorithms = [{"algorithm_id": item.algorithm_id, "version": item.algorithm_version, "status": item.status,
                       "adapter": item.execution.adapter if item.execution else None,
                       "result_adapter": item.execution.result_adapter if item.execution else None,
                       "result_adapter_installed": (item.execution.result_adapter in installed_result_adapters()
                                                    if item.execution else None)}
                      for item in catalog.algorithms]
        catalog_version = catalog.reference.catalog_version
    except CatalogError:
        catalog_version = None
        issues.append("algorithm catalog cannot be loaded or validated")
    inventory = None
    if config.data_availability is not None:
        try:
            document = FileDataAvailabilityChecker(config.data_availability).load()
            expired = document.expires_at <= datetime.now(timezone.utc)
            inventory = {"version": document.version, "deployment": document.deployment,
                         "expires_at": document.expires_at.isoformat(), "expired": expired}
            if expired:
                issues.append("data availability snapshot has expired; new managed submissions are blocked")
        except PreflightError:
            issues.append("data availability snapshot cannot be loaded or validated")
    policy = {"mode": "allow_all", "version": "built-in development policy", "config": None}
    if config.policy_config is not None:
        try:
            parsed = PolicyConfiguration.model_validate(strict_json(Path(config.policy_config).read_bytes()))
            policy = {"mode": parsed.mode, "version": parsed.version, "config": str(config.policy_config)}
        except (OSError, ValueError):
            policy = {"mode": "invalid", "version": None, "config": str(config.policy_config)}
            issues.append("policy configuration cannot be loaded or validated")
    llm = config.llm
    key_env = llm.deepseek_api_key_env if llm.provider == "deepseek" else llm.local_api_key_env
    if llm.tool_lookups and (llm.provider == "demo" or llm.runtime != "sdk"):
        issues.append("llm.tool_lookups has no effect without a model on the sdk runtime")
    return {
        "version": package_version(),
        "config_file": str(config.source) if config.source else None,
        "output": str(config.output),
        "metadata": {
            "algorithm_catalog": str(config.algorithm_catalog), "catalog_version": catalog_version,
            "semantic_schema": str(config.semantic_schema), "deployment": str(config.deployment),
            "data_availability": inventory, "execution_templates": str(config.execution_templates)
            if config.execution_templates else "bundled default",
            "registry_source": str(config.registry_source) if config.registry_source else None,
        },
        "planner": {"provider": llm.provider, "runtime": llm.runtime, "model": llm.model,
                    "endpoint": llm.deepseek_endpoint if llm.provider == "deepseek" else llm.local_endpoint,
                    "api_key_env": key_env if llm.provider != "demo" else None,
                    "api_key_file_configured": llm.deepseek_api_key_file is not None,
                    "streaming": llm.stream, "tool_lookups": llm.tool_lookups, "max_turns": llm.max_turns},
        "backend": {"kind": config.backend, "synthetic_sandbox_confirmed": config.confirm_synthetic_unencrypted_sandbox,
                    "timeout_seconds": config.timeout_seconds},
        "policy": policy,
        "managed_execution": config.managed_execution,
        "algorithms": algorithms,
        "issues": issues,
    }
