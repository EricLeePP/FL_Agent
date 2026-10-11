"""Read-only MCP server: catalog, deployment and plan checks for other agents.

Run it over stdio, for example from an MCP client's server configuration:

    python -m federated_agent.mcp_server --config fed-agent.json

Every tool reads fresh metadata, as every command-line request does. No tool
writes files, asks for credentials, returns data or submits a task: execution
stays in a researcher's own confirmed session. Only stdio is offered, because a
network transport would first need an authentication layer.
"""

import argparse
import logging
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:  # mcp 2.x
    from mcp.server.mcpserver import MCPServer
    from mcp.types import ToolAnnotations
except ImportError:
    try:  # mcp 1.x, untested
        from mcp.server.fastmcp import FastMCP as MCPServer
        from mcp.types import ToolAnnotations
    except ImportError:
        MCPServer = ToolAnnotations = None

from .app import build_application
from .bootstrap import CONFIG_ENV, ConfigurationError, build_app_config, load_settings
from .errors import CatalogError, PreflightError
from .lookups import ResearchLookups
from .models import ResearchPlan
from .narration import reply_facts
from .observability import configure_logging, log_incident
from .prompts import MCP_SERVER_PROMPT
from .services.responses import metadata_failure

logger = logging.getLogger(__name__)
MISSING_EXTRA = "The MCP server needs the mcp extra: python -m pip install 'federated-research-agent[mcp]'"



def build_server(agent):
    """An MCP server whose tools only read; ``agent`` is a FederatedResearchAgent."""
    if MCPServer is None:
        raise RuntimeError(MISSING_EXTRA)
    READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
    # The server's own default logs every library at INFO; the agent's logging is configured separately.
    server = MCPServer("fed-agent", instructions=MCP_SERVER_PROMPT, log_level="WARNING")
    lock = threading.Lock()

    def guarded(operation):
        # One call at a time, so every call sees exactly one metadata snapshot.
        with lock:
            try:
                return operation()
            except (CatalogError, PreflightError) as exc:
                response = metadata_failure(exc)
                return {"status": response.status, "message": response.message}
            except Exception as exc:
                incident = log_incident(logger, "MCP tool failed unexpectedly", exc)
                return {"status": "internal_error", "incident_id": incident,
                        "message": "The tool failed. The server log has the details for this incident."}

    def lookups():
        return ResearchLookups(agent.refresh_metadata())

    @server.tool(annotations=READ_ONLY, structured_output=True)
    def list_algorithms() -> dict[str, Any]:
        """List registered algorithms with their versions, capabilities and limitations.

        Registration is not proof that an algorithm is deployed or available now.
        """
        return guarded(lambda: lookups().list_algorithms())

    @server.tool(annotations=READ_ONLY, structured_output=True)
    def list_sites() -> dict[str, Any]:
        """List configured hospitals with their aliases and dataset labels. A dataset must always be chosen explicitly."""
        return guarded(lambda: lookups().list_sites())

    @server.tool(annotations=READ_ONLY, structured_output=True)
    def resolve_variable(name: str) -> dict[str, Any]:
        """Map a researcher's wording to a semantic-catalog variable. Near matches come back as candidates, never as a guess."""
        return guarded(lambda: lookups().resolve_variable(name))

    @server.tool(annotations=READ_ONLY, structured_output=True)
    def resolve_site(name: str) -> dict[str, Any]:
        """Map a hospital name or alias to a configured hospital and its dataset labels."""
        return guarded(lambda: lookups().resolve_site(name))

    @server.tool(annotations=READ_ONLY, structured_output=True)
    def check_plan(variables: list[str], sites: list[str], database_label: str | None = None,
                   analysis_type: str = "descriptive_statistics", algorithm_id: str | None = None,
                   stratification_variables: list[str] | None = None, parameters: dict | None = None,
                   unsupported_requirements: list[str] | None = None, purpose: str = "") -> dict[str, Any]:
        """Check a research plan exactly as the agent checks one before its task review.

        Returns the checked status (ready, needs_clarification, unsupported, ...), the issues and any
        clarification questions. Nothing is written or submitted; the researcher must still prepare,
        review and confirm the task in the agent.
        """
        def check():
            # MCP has already validated the argument types against this signature.
            plan = ResearchPlan(analysis_type=analysis_type, algorithm_id=algorithm_id, variables=variables,
                                stratification_variables=stratification_variables or [], sites=sites,
                                database_label=database_label, parameters=parameters or {},
                                unsupported_requirements=unsupported_requirements or [],
                                purpose=purpose or "Plan checked through MCP")
            response = agent.check_plan(plan)
            return reply_facts(response) or {"status": response.status, "message": response.message}
        return guarded(check)

    return server


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m federated_agent.mcp_server",
                                     description="Read-only Fed Agent MCP server over stdio.")
    parser.add_argument("--config", type=Path, help=f"JSON configuration file (or set {CONFIG_ENV})")
    parser.add_argument("-v", "--verbose", action="count", default=None, help="Log more detail to stderr")
    parser.add_argument("--log-file", type=Path, help="Also write a full debug log to this file")
    args = parser.parse_args(argv)
    if MCPServer is None:
        parser.exit(2, MISSING_EXTRA + "\n")
    try:
        settings, source = load_settings(args.config)
        config = build_app_config(SimpleNamespace(verbose=args.verbose, log_file=args.log_file), settings, source)
    except ConfigurationError as exc:
        parser.error(str(exc))
    # stdout carries the protocol; logs go to stderr and the optional file only.
    configure_logging(config.verbosity, config.log_file)
    try:
        agent = build_application(config).agent
    except (CatalogError, OSError, ValueError, PreflightError) as exc:
        parser.exit(2, metadata_failure(exc).message + "\n")
    build_server(agent).run("stdio")


if __name__ == "__main__":
    main()
