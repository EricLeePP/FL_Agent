"""Credential-free run identity and provenance for the fixed execution workflow."""

import hashlib
from pathlib import Path

from .execution_templates import WORKER_SOURCES
from .storage import write_json
from .capabilities import PlannerCapabilities, capabilities_of
from .prompts import PLAN_PROMPT, UPDATE_PROMPT, prompt_sha256


TEMPLATE_ID = "joshi_vantage6_userclient"
TEMPLATE_VERSION = "1.0.0"


def write_manifest(output, prepared, planner, session, *, status="prepared", task_id=None):
    sources = Path(__file__).parent
    template_files = [*WORKER_SOURCES, "script_renderer.py", "task_compiler.py"]
    model = getattr(getattr(planner, "provider", None), "model", None)
    language_model = capabilities_of(planner, PlannerCapabilities).language_model
    runtime = {name: value if isinstance(value, str) else None
               for name in ("runtime_name", "runtime_version", "runtime_prompt_sha256")
               for value in [getattr(getattr(planner, "provider", None), name, None)]}
    write_json(Path(output) / "run_manifest.json", {
        "schema_version": "1.0",
        "session_id": session.session_id,
        "study_revision": session.revision,
        "confirmation_id": prepared.confirmation_id,
        "backend": session.backend,
        "status": status,
        "task_id": task_id,
        "template": {**(prepared.task_spec.execution_template.model_dump() if prepared.task_spec.execution_template else
                         {"id": TEMPLATE_ID, "version": TEMPLATE_VERSION}),
                     "implementation_sha256": {name: hashlib.sha256((sources / name).read_bytes()).hexdigest()
                                               for name in template_files}},
        "planner": {"type": type(planner).__name__,
                    "runtime": runtime["runtime_name"] or ("json" if language_model else "offline"),
                    "runtime_version": runtime["runtime_version"],
                    "runtime_prompt_sha256": runtime["runtime_prompt_sha256"],
                    "model": model if isinstance(model, str) else None,
                    "prompt_sha256": prompt_sha256(PLAN_PROMPT, UPDATE_PROMPT)
                    if language_model else None},
        "catalog": prepared.task_spec.catalog.model_dump() if prepared.task_spec.catalog else None,
        "registry_release": prepared.task_spec.registry_release.model_dump() if prepared.task_spec.registry_release else None,
        "local_metadata_hashes": prepared.task_spec.local_metadata_hashes,
        "provenance": [item.model_dump() for item in prepared.task_spec.provenance],
        "data_checks": [item.model_dump() for item in prepared.data_checks],
        "policy": prepared.policy_decision.model_dump(),
    })


def finish_manifest(output, response):
    from .storage import read_json
    path = Path(output) / "run_manifest.json"
    if path.exists():
        manifest = read_json(path)
        manifest["status"] = response.status
        manifest["report_status"] = response.report_status
        manifest["submission_state"] = response.submission_state
        manifest["task_id"] = response.execution.task_id if response.execution else manifest.get("task_id")
        write_json(path, manifest)
