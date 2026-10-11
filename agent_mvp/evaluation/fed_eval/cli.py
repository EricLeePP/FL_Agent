"""Command line: validate a dataset, list or pull models, run models and score results."""

import argparse
import json
import platform
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from . import cases as case_io
from . import variants
from .provider import EvalProvider, server_version, unload
from .runner import completed, run
from .world import fingerprint

HERE = Path(__file__).resolve().parents[1]
DEFAULT_ENDPOINT = "http://127.0.0.1:11435/v1/chat/completions"


def load_models(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))["models"]


def select_models(models, names):
    if not names:
        return models
    by_key = {key: m for m in models for key in (m["ref"], m.get("name"))}
    missing = [name for name in names if name not in by_key]
    if missing:
        raise SystemExit(f"Unknown model(s): {', '.join(missing)}")
    return [by_key[name] for name in dict.fromkeys(names)]  # in the order given


def installed(endpoint, attempts=10):
    """Model names on an Ollama server, which can be slow to answer while it loads a model from a slow drive."""
    base = endpoint.split("/v1/")[0]
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(base + "/api/tags", timeout=60) as response:
                return {item["name"] for item in json.loads(response.read())["models"]}
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(30)


def git_commit():
    """HEAD, marked "+dirty" when the working tree has uncommitted changes."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=HERE, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return head + ("+dirty" if dirty else "")


def command_validate(args):
    cases = case_io.load(args.dataset)
    problems = case_io.structure_problems(cases)
    if problems:
        print("\n".join(problems))
        return 2
    from .scoring import check_expectations
    from .world import build_world_agent
    agent = build_world_agent()
    failures = 0
    for case in cases:
        record = case_io.oracle_run(case, agent)
        mismatches = check_expectations(case, record)
        if record["harness_error"] or mismatches:
            failures += 1
            print(f"{case['id']}: {record['harness_error'] or '; '.join(mismatches)}")
    print(f"{len(cases) - failures} of {len(cases)} cases reach their expected outcome with a perfect planner")
    return 0 if failures == 0 else 2


def command_models(args):
    have = installed(args.endpoint)
    for model in load_models(args.models):
        mark = "present" if model["ref"] in have or f"{model['ref']}:latest" in have else "missing"
        print(f"{mark:8} {model['ref']:48} {model.get('download_gb', '?'):>6} GB  {model.get('role', '')}")
    return 0


def command_pull(args):
    base = args.endpoint.split("/v1/")[0]
    have = installed(args.endpoint)
    for model in select_models(load_models(args.models), args.model):
        if model["ref"] in have or f"{model['ref']}:latest" in have:
            print(f"present  {model['ref']}")
            continue
        print(f"pulling  {model['ref']}", flush=True)
        request = urllib.request.Request(base + "/api/pull", data=json.dumps({"model": model["ref"], "stream": True}).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
        last = None
        with urllib.request.urlopen(request, timeout=None) as response:
            for line in response:
                event = json.loads(line)
                if "error" in event:
                    print(f"failed   {model['ref']}: {event['error']}")
                    break
                status = event.get("status")
                if status != last and not status.startswith("pulling ") or status == "success":
                    print(f"         {status}", flush=True)
                last = status
    return 0


def command_run(args):
    cases = case_io.load(args.dataset)
    if args.categories:
        cases = [c for c in cases if c.get("category") in args.categories]
    if args.split:
        cases = [c for c in cases if c.get("split", "core") == args.split]
    if args.limit:
        cases = cases[:args.limit]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    repeats = len(seeds) if seeds else args.repeats
    for model in select_models(load_models(args.models), args.model):
        name = model.get("name", model["ref"])
        out = Path(args.out) / f"{name.replace('/', '_').replace(':', '_')}.jsonl"
        done = completed(out, cases)
        if all((case["id"], repeat) in done for repeat in range(repeats) for case in cases):
            print(f"== {name}: complete", flush=True)  # loading a model from a slow drive takes minutes
            continue
        if args.wait:
            while not {model["ref"], f"{model['ref']}:latest"} & installed(args.endpoint):
                print(f"waiting for {model['ref']} to finish downloading", flush=True)
                time.sleep(60)
        provider = EvalProvider(args.endpoint, model["ref"], options=model.get("options"), temperature=args.temperature,
                                structured_output=model.get("structured_output", "json_schema"),
                                max_tokens=model.get("max_tokens", 1200),
                                timeout_seconds=model.get("timeout_seconds", 1800))
        with variants.applied(args.variant):
            meta = {"model": model, "endpoint": args.endpoint, "server_version": server_version(args.endpoint),
                    "dataset": str(args.dataset), "dataset_sha256": case_io.digest(args.dataset), "cases": len(cases),
                    "repeats": repeats, "seeds": seeds, "temperature": args.temperature, "world": fingerprint(),
                    "variant": args.variant, "prompt_sha256": variants.current_prompt_sha256(), "git_commit": git_commit(),
                    "python": platform.python_version(), "started_at": datetime.now(timezone.utc).isoformat()}
            print(f"== {name}: {len(cases)} cases x {repeats}" + (f", variant {args.variant}" if args.variant else ""), flush=True)
            started = time.monotonic()
            run(cases, provider, out, repeats=repeats, seeds=seeds, progress=lambda text: print(text, flush=True))
        meta.update(finished_at=datetime.now(timezone.utc).isoformat(), wall_seconds=round(time.monotonic() - started, 1))
        meta_path = out.with_suffix(".meta.json")
        if meta_path.exists():  # a resumed run keeps when it first started and adds up its time
            earlier = json.loads(meta_path.read_text(encoding="utf-8"))
            meta.update(started_at=earlier["started_at"], wall_seconds=round(earlier["wall_seconds"] + meta["wall_seconds"], 1),
                        sessions=earlier.get("sessions", 1) + 1)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        unload(args.endpoint, model["ref"])
    return 0


def command_check(args):
    """One real case per model: catches thinking left on, truncation and context problems before a long run."""
    from .runner import run_case
    from .world import build_world_agent
    case = next(c for c in case_io.load(args.dataset) if c["category"] == "ready")
    have = installed(args.endpoint)
    agent = build_world_agent()
    failures = 0
    for model in select_models(load_models(args.models), args.model):
        if not {model["ref"], f"{model['ref']}:latest"} & have:
            print(f"missing  {model['ref']}")
            continue
        provider = EvalProvider(args.endpoint, model["ref"], options=model.get("options"),
                                structured_output=model.get("structured_output", "json_schema"), max_tokens=model.get("max_tokens", 1200),
                                timeout_seconds=model.get("timeout_seconds", 1800))
        record = run_case(agent, provider, case)
        calls = [c for turn in record["turns"] for c in turn["model_calls"]]
        status = record["turns"][-1]["responses"][-1]["status"] if record["turns"] and record["turns"][-1]["responses"] else None
        problems = []
        if any(c.get("reasoning_chars") for c in calls):
            problems.append("thinking is on")
        problems += sorted({c["error"] for c in calls if c.get("error")})
        if any(c.get("schema_valid") is False for c in calls):
            problems.append("schema invalid")
        failures += bool(problems)
        first = calls[0] if calls else {}
        print(f"{'ok' if not problems else 'PROBLEM':8} {model['ref']:30} status={status} calls={len(calls)} prompt={first.get('prompt_tokens')} "
              f"output={first.get('completion_tokens')} {first.get('seconds')}s {', '.join(problems)}", flush=True)
        unload(args.endpoint, model["ref"])
    return 0 if failures == 0 else 2


def command_replay(args):
    """Replay recorded outputs through a modified pipeline (no model calls) to measure a product change."""
    from .replay import ReplayProvider
    from .runner import load_records, run_case
    from .world import build_world_agent
    cases = {case["id"]: case for case in case_io.load(args.dataset)}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for path in sorted(Path(args.results).glob("*.jsonl")):
        if path.name.endswith(".scored.jsonl"):
            continue
        records = load_records(path, cases)
        with variants.applied(args.change), (Path(args.out) / path.name).open("w", encoding="utf-8") as out:
            agent = build_world_agent()  # inside the variants, which may change the deployment
            for record in records:
                replayed = run_case(agent, ReplayProvider(record), cases[record["case_id"]])
                replayed.update(case_id=record["case_id"], repeat=record["repeat"], seed=record.get("seed"))
                for turn in replayed["turns"]:  # replay time is meaningless: report the recorded model time
                    if turn.get("model_calls"):
                        turn["seconds"] = round(sum(call.get("seconds") or 0 for call in turn["model_calls"]), 3)
                if replayed["harness_error"] and not record.get("harness_error"):
                    # Runs before 2026-10-10 logged at most 4,000 characters of each output, so a few long
                    # outputs cannot be replayed; such a case keeps its measured outcome, marked as such.
                    replayed = {**record, "replay_fallback": replayed["harness_error"].strip().splitlines()[-1]}
                out.write(json.dumps(replayed, ensure_ascii=False) + "\n")
        meta_path = path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        meta["counterfactual"] = args.change
        (Path(args.out) / meta_path.name).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"replayed {path.stem}: {len(records)} records", flush=True)
    return 0


def command_score(args):
    from .scoring import score_results
    return score_results(args)


def command_redteam(args):
    """Planner errors that must never become a wrong task, run with and without the safeguards."""
    from .redteam import SAFEGUARDS, report, run_attacks
    rows = run_attacks(args.layers or SAFEGUARDS)
    print(report(rows))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fedeval", description="Evaluate planner LLMs for Fed Agent.")
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="OpenAI-compatible chat-completions URL")
    common.add_argument("--models", default=HERE / "models.json", type=Path)

    validate = sub.add_parser("validate", help="Check structure and labels with a perfect planner")
    validate.add_argument("--dataset", required=True, type=Path)
    validate.set_defaults(func=command_validate)

    models = sub.add_parser("models", parents=[common], help="List the comparison models and whether they are present")
    models.set_defaults(func=command_models)

    pull = sub.add_parser("pull", parents=[common], help="Download missing models through the Ollama server")
    pull.add_argument("--model", nargs="*", help="Names or references; default all")
    pull.set_defaults(func=command_pull)

    runner = sub.add_parser("run", parents=[common], help="Run models over a dataset")
    runner.add_argument("--dataset", required=True, type=Path)
    runner.add_argument("--model", nargs="*", help="Names or references; default all")
    runner.add_argument("--out", default=HERE / "results", type=Path)
    runner.add_argument("--repeats", type=int, default=1)
    runner.add_argument("--seeds", help="Comma-separated seeds, one run each (overrides --repeats)")
    runner.add_argument("--temperature", type=float, default=0.0)
    runner.add_argument("--categories", nargs="*")
    runner.add_argument("--split", help="Only cases of this split (core or hidden)")
    runner.add_argument("--limit", type=int)
    runner.add_argument("--wait", action="store_true", help="Wait for each model to be downloaded before running it")
    runner.add_argument("--variant", help="Comma-separated configuration variants to measure: quote_requirements "
                        "(prompt), grounding and ground_names (pipeline), site_aliases (deployment)")
    runner.set_defaults(func=command_run)

    check = sub.add_parser("check", parents=[common], help="Run one real case per model to verify its settings")
    check.add_argument("--dataset", default=HERE / "datasets" / "fed_agent_eval_smoke.jsonl", type=Path)
    check.add_argument("--model", nargs="*", help="Names or references; default all")
    check.set_defaults(func=command_check)

    replay = sub.add_parser("replay", help="Replay recorded outputs through a modified pipeline, without model calls")
    replay.add_argument("--dataset", required=True, type=Path)
    replay.add_argument("--results", required=True, type=Path)
    replay.add_argument("--change", default="grounding",
                        help="Comma-separated pipeline variants: grounding (drop model-written requirements that share "
                             "no word with the researcher's text), ground_names (drop hospitals and a dataset the "
                             "researcher never named), site_aliases (Chinese and variant hospital names as aliases)")
    replay.add_argument("--out", required=True, type=Path)
    replay.set_defaults(func=command_replay)

    score = sub.add_parser("score", help="Score result files and write a report")
    score.add_argument("--dataset", required=True, type=Path)
    score.add_argument("--results", required=True, type=Path, help="Directory of result .jsonl files")
    score.add_argument("--out", type=Path, help="Report directory (default: the results directory)")
    score.set_defaults(func=command_score)

    redteam = sub.add_parser("redteam", help="Run planner-error attacks against the safeguard layers")
    redteam.add_argument("--layers", help="Comma-separated variants to test (default: the full safeguard stack)")
    redteam.add_argument("--out", type=Path, help="Write the per-attack results as JSON")
    redteam.set_defaults(func=command_redteam)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
