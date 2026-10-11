"""Evaluate planning only; this script never authenticates to or submits Vantage6."""

import argparse
import json
import os
import hashlib
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from federated_agent.app import build_agent
from federated_agent.resources import SOURCE_CHECKOUT
from federated_agent.resources import SANDBOX_DATA_AVAILABILITY, SANDBOX_DEPLOYMENT
from federated_agent.credentials import CredentialError, load_deepseek_api_key
from federated_agent.planner import DeepSeekJSONProvider, LLMResearchPlanner, OpenAICompatibleJSONProvider


@contextmanager
def fixture_clock(value):
    """Planning benchmark only; the runtime CLI exposes no clock override."""
    if value is None:
        yield
        return
    from unittest.mock import patch
    with patch("federated_agent.data_availability.datetime") as clock:
        clock.now.return_value = value
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["deepseek", "local"], default="deepseek")
    parser.add_argument("--agent-runtime", choices=["json", "sdk"], default="json")
    parser.add_argument("--stream", action="store_true", help="Evaluate the streamed planner transport")
    parser.add_argument("--model")
    parser.add_argument("--endpoint")
    parser.add_argument("--deepseek-api-key-file", type=Path)
    parser.add_argument("--data-availability-config", type=Path,
                        default=SANDBOX_DATA_AVAILABILITY)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--output", type=Path, default=SOURCE_CHECKOUT / "output-evaluation")
    parser.add_argument("--fixture-time", help="ISO timestamp with timezone for historical planning-fixture evaluation only; never enables execution")
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    fixture_time = None
    if args.fixture_time:
        try:
            fixture_time = datetime.fromisoformat(args.fixture_time.replace("Z", "+00:00"))
            if fixture_time.tzinfo is None:
                raise ValueError()
        except ValueError:
            parser.error("--fixture-time must be an ISO timestamp with an explicit timezone")
    if args.provider == "deepseek":
        try:
            key = load_deepseek_api_key(args.deepseek_api_key_file)
        except CredentialError as exc:
            parser.error(str(exc))
        if not key:
            parser.error("Supply --deepseek-api-key-file or set DEEPSEEK_API_KEY locally")
        provider = DeepSeekJSONProvider(key, model=args.model or os.environ.get("DEEPSEEK_MODEL") or DeepSeekJSONProvider.DEFAULT_MODEL)
    else:
        if not args.endpoint or not args.model:
            parser.error("Local provider requires --endpoint and --model")
        provider = OpenAICompatibleJSONProvider(args.endpoint, args.model, os.environ.get("LOCAL_LLM_API_KEY"))
    if args.agent_runtime == "sdk":
        from federated_agent.sdk_runtime import AgentsSDKProvider
        provider = AgentsSDKProvider.from_provider(provider)
    planner = LLMResearchPlanner(provider)
    if args.stream:
        planner.on_delta = lambda _: None
    agent = build_agent(SANDBOX_DEPLOYMENT,
        data_availability_path=args.data_availability_config)
    output = args.output / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    cases = json.loads((SOURCE_CHECKOUT / "examples/cli_requests.json").read_text(encoding="utf-8"))["cases"][:args.limit]
    records = []
    for case in cases:
        started = time.monotonic()
        with fixture_clock(fixture_time):
            response = agent.respond(case["request"], planner)
        passed = response.status in case["statuses"]
        if "clarification_fields" in case:
            fields = {question.field for question in response.clarification.questions} if response.clarification else set()
            passed &= set(case["clarification_fields"]) <= fields and response.prepared_task is None
        if response.prepared_task:
            resolved = response.prepared_task.resolved_plan
            if "sites" in case:
                passed &= set(case["sites"]) == {site.organisation_id for site in resolved.sites}
            if "variables" in case:
                passed &= set(case["variables"]) == {variable.class_id for variable in resolved.variables}
            if "database" in case:
                passed &= case["database"] == response.prepared_task.task_spec.databases[0].label
        records.append({"id": case["id"], "status": response.status, "passed": bool(passed),
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "issues": [item.code for item in response.issues]})
        (output / f"{case['id']}.json").write_text(response.model_dump_json(indent=2, exclude={"catalog_document"}), encoding="utf-8")
        print(f"{case['id']}: {response.status}; expectation {'PASS' if passed else 'FAIL'}")
        if response.status == "planner_error":
            print("Provider failure: stopped; remaining cases were NOT evaluated.")
            break
    report = {"provider": args.provider, "model": provider.model, "vantage6_submissions": 0,
              "agent_runtime": args.agent_runtime, "stream": args.stream,
              "runtime_version": getattr(provider, "runtime_version", None),
              "fixture_time": fixture_time.isoformat() if fixture_time else None,
              "fixture_time_is_not_current_hospital_evidence": fixture_time is not None,
              "catalog": agent.algorithm_catalog.snapshot().reference.model_dump(),
              "inventory_path": str(args.data_availability_config.resolve()),
              "inventory_sha256": hashlib.sha256(args.data_availability_config.read_bytes()).hexdigest(),
              "prompt_sha256": hashlib.sha256(LLMResearchPlanner.SYSTEM_PROMPT.encode()).hexdigest(),
              "requested_cases": len(cases), "evaluated_cases": len(records),
              "cases_sha256": hashlib.sha256((SOURCE_CHECKOUT / "examples/cli_requests.json").read_bytes()).hexdigest(),
              "passed": sum(record["passed"] for record in records), "records": records}
    (output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {output / 'evaluation.json'}")
    return 0 if len(records) == len(cases) and all(record["passed"] for record in records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
