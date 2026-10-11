"""Command-line transport: parse arguments, then run one request or a saved study.

The rules live in services/ and in the agent; this module only translates
arguments, prints results and chooses exit codes.
"""

import argparse
import json
import logging
from getpass import getpass
from pathlib import Path

from .app import build_application, build_planner, describe
from .bootstrap import CONFIG_ENV, ConfigurationError, apply_to_args, build_app_config, load_settings
from .errors import CatalogError, PreflightError
from .models import PreparedTask
from .observability import configure_logging, log_incident
from .services import execution
from .services.artifacts import write_prepared, write_response
from .services.responses import error_response, metadata_failure
from .services.study import StudyChat, StudySettings
from .sessions import SessionError
from .terminal_ui import TerminalUI

logger = logging.getLogger(__name__)

EXIT_STATUSES = {"catalog_error", "planner_error", "invalid_request", "unsupported", "data_unavailable",
                 "permission_denied", "preflight_error"}


def _planner_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser, config):
    try:
        return build_planner(config.llm, interactive=bool(args.chat), ask_secret=getpass)
    except ConfigurationError as exc:
        parser.error(str(exc))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help=f"JSON configuration file (or set {CONFIG_ENV}); flags override it")
    parser.add_argument("--describe", action="store_true", help="Print the effective configuration, capabilities and health, then exit")
    parser.add_argument("-v", "--verbose", action="count", default=None, help="Log more detail to stderr (-v info, -vv debug)")
    parser.add_argument("--log-file", type=Path, help="Also write a full debug log to this file")
    parser.add_argument(
        "--request",
        default=(
            "Calculate descriptive statistics for age at initial diagnosis and "
            "survival status in hospital 1, hospital 2, and hospital 3."
        ),
    )
    parser.add_argument("--output", type=Path, help="Output directory (default: output)")
    parser.add_argument("--chat", action="store_true", help="Interactive requests, clarification, exact-task review and execution")
    parser.add_argument("--resume", help="Resume a saved session ID under --output; never auto-execute")
    parser.add_argument("--plain", action="store_true", help="Plain terminal input without cursor menus")
    parser.add_argument("--no-stream", dest="stream", action="store_const", const=False, default=None,
                        help="Disable SSE and the optional model explanation; keep deterministic feedback")
    parser.add_argument("--backend", choices=["mock", "vantage6"], help="Execution backend (default: mock)")
    parser.add_argument("--confirm-synthetic-unencrypted-sandbox", action="store_true", help="Allow only the local-tunnel synthetic 4.15 sandbox; not production")
    parser.add_argument("--timeout-seconds", type=int, help="Real execution wait limit (default: 600)")
    parser.add_argument("--catalog", type=Path, help="Live algorithm registry, read at every request")
    parser.add_argument("--registry-source", type=Path, help="Trusted registry source configuration; sync a coherent release at each request")
    parser.add_argument("--policy-config", type=Path, help="Local policy configuration, never supplied by the LLM or a remote metadata release")
    parser.add_argument("--template-registry", type=Path, help="Local versioned execution-template descriptors")
    parser.add_argument("--list-algorithms", action="store_true", help="List the current file-backed capabilities without calling an LLM")
    parser.add_argument("--validate-catalog", action="store_true", help="Validate registry contracts without running an analysis")
    parser.add_argument("--data-availability-config", type=Path, help="Optional operator-maintained data metadata snapshot; not an authorization file")
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--execute", action="store_true", help="Execute using the selected backend; requires --confirm")
    execution.add_argument("--execute-prepared", type=Path, help="Execute an exact prepared_task.json artifact")
    parser.add_argument("--confirm", help="Confirmation ID printed during the preparation step")
    parser.add_argument("--deployment-config", type=Path, help="Public deployment registry containing IDs and database labels")
    parser.add_argument("--provider", choices=["demo", "deepseek", "local"],
                        help="Planner backend; demo stays offline (default: LLM_PROVIDER or demo)")
    parser.add_argument("--llm-endpoint", help="Full URL of a local chat-completions endpoint (default: LOCAL_LLM_ENDPOINT)")
    parser.add_argument("--model", help="Model name expected by the selected provider")
    parser.add_argument("--agent-runtime", choices=["sdk", "json"],
                        help="Model runtime: json (JSON mode over plain HTTP, the default here) or sdk (the Agents SDK, "
                             "the default of start_agent.py).")
    parser.add_argument("--api-key-env", help="Environment variable containing the optional local LLM API key (default: LOCAL_LLM_API_KEY)")
    parser.add_argument("--deepseek-api-key-file", type=Path,
                        help="UTF-8 file containing a bare key; overrides DEEPSEEK_API_KEY_FILE and the key environment variable")
    parser.add_argument("--deepseek-api-key-env", help="Environment variable containing the DeepSeek API key (default: DEEPSEEK_API_KEY)")
    parser.add_argument("--deepseek-endpoint", help="DeepSeek chat-completions endpoint (default: DEEPSEEK_ENDPOINT or the public endpoint)")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings, source = load_settings(args.config)
        config = build_app_config(args, settings, source)
    except ConfigurationError as exc:
        parser.error(str(exc))
    configure_logging(config.verbosity, config.log_file)
    if args.describe:
        print(json.dumps(describe(config), indent=2))
        return
    apply_to_args(args, config)
    try:
        _run(args, parser, config)
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception as exc:
        incident = log_incident(logger, "Unexpected failure in the command line", exc)
        parser.exit(2, f"Unexpected internal error (incident {incident}). The log contains the details.\n")


def _run(args, parser, config) -> None:
    if args.resume and not args.chat:
        parser.error("--resume requires --chat")
    if args.chat and (args.execute or args.execute_prepared or args.confirm):
        parser.error("--chat uses interactive confirmation; do not combine with execution flags")
    if args.backend == "vantage6" and not args.confirm_synthetic_unencrypted_sandbox:
        parser.error("Real execution requires --confirm-synthetic-unencrypted-sandbox")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        agent = build_application(config, registry_recovery=bool(args.resume)).agent
    except (CatalogError, OSError, ValueError, PreflightError) as exc:
        response = metadata_failure(exc)
        write_response(response, args.output)
        parser.exit(2, response.message + "\n")

    if args.list_algorithms or args.validate_catalog:
        response = agent.describe_algorithms()
        write_response(response, args.output)
        print(response.model_dump_json(indent=2, exclude={"catalog_document"}))
        if response.status != "algorithms":
            parser.exit(2)
        return

    if args.chat:
        planner = _planner_from_args(args, parser, config)
        try:
            run_chat(agent, planner, args)
        except SessionError as exc:
            parser.exit(2, str(exc) + "\n")
        return

    if args.execute_prepared:
        try:
            prepared = PreparedTask.model_validate_json(args.execute_prepared.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            write_response(error_response("invalid_request", "confirmation_invalid"), args.output)
            parser.exit(2)
    else:
        prepared = _prepare_request(agent, args, parser, config)
        if prepared is None:
            return

    if not args.execute and not args.execute_prepared:
        print("Review prepared_task.json and run it with --execute-prepared and --confirm <confirmation-id>.")
        return
    if not args.confirm:
        parser.error("Execution requires --confirm <confirmation-id>")
    _execute(agent, args, parser, prepared)


def _prepare_request(agent, args, parser, config):
    """Answer --request; return the prepared task, or None when there is nothing to execute."""
    planner = _planner_from_args(args, parser, config)
    try:
        response = agent.respond(args.request, planner)
    except PreflightError:
        response = error_response("preflight_error", "preflight_error")
    write_response(response, args.output)
    if response.status == "needs_clarification":
        clarification_path = args.output / "clarification.json"
        clarification_path.write_text(response.clarification.model_dump_json(indent=2), encoding="utf-8")
        print(f"Clarification required: {clarification_path.resolve()}")
        return None
    if response.status != "ready":
        print(f"Detailed response: {(args.output / 'response.json').resolve()}")
        if response.status in EXIT_STATUSES:
            parser.exit(2)
        return None
    prepared = response.prepared_task
    prepared_path = write_prepared(prepared, args.output)
    print("Generated files:")
    print(f"  Prepared task: {prepared_path.resolve()}")
    print(f"  Vantage6 task spec: {(args.output / 'task_spec.json').resolve()}")
    print(f"  Researcher-side script: {(args.output / 'run_vantage6_task.py').resolve()}")
    print(f"Confirmation ID: {prepared.confirmation_id}")
    return prepared


def _execute(agent, args, parser, prepared):
    prepared_dir = args.execute_prepared.resolve().parent if args.execute_prepared else args.output
    execution.configure_executor(agent, args.backend, prepared_dir, confirmed=args.confirm_synthetic_unencrypted_sandbox,
                                 timeout_seconds=args.timeout_seconds)
    outcome = agent.execute_response(prepared, args.confirm)
    write_response(outcome, args.output)
    if outcome.status != "completed":
        parser.exit(2, outcome.message + "\n")
    outcome = execution.finish_execution(
        agent, prepared, outcome, args.output, args.backend,
        persist=lambda item: write_response(item, args.output, display=item.report_status != "pending"))
    if outcome.status == "report_failed":
        parser.exit(2, outcome.message + "\n")


def run_chat(agent, planner, args, read=None, ui=None):
    ui = ui or TerminalUI(read=read, plain=getattr(args, "plain", False), stream=getattr(args, "stream", True))
    StudyChat(agent, planner, ui, StudySettings.from_args(args)).run()


if __name__ == "__main__":
    main()
