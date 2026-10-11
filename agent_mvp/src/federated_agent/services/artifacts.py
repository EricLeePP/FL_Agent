"""Files that record a turn: the response, the prepared task and its review copies."""

from pathlib import Path

from ..feedback import render_researcher_response
from ..models import AgentResponse, PreparedTask
from ..script_renderer import render_vantage6_script


def write_response(response: AgentResponse, output: Path, *, display=True, emit=print) -> None:
    output = Path(output)
    (output / "response.json").write_text(
        response.model_dump_json(indent=2, exclude={"catalog_document"}), encoding="utf-8",
    )
    if response.catalog_content is not None:
        (output / "catalog_snapshot.json").write_bytes(response.catalog_content.encode("utf-8"))
    text = render_researcher_response(response)
    (output / "researcher_response.md").write_text(text, encoding="utf-8")
    if display:
        emit(text)


def write_planning_response(response: AgentResponse, output: Path) -> None:
    (Path(output) / "planning_response.json").write_text(
        response.model_dump_json(indent=2, exclude={"catalog_document"}), encoding="utf-8")


def write_prepared(prepared: PreparedTask, output: Path) -> Path:
    """Write the exact prepared task, its task specification and the review-only script."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    prepared_path = output / "prepared_task.json"
    prepared_path.write_text(prepared.model_dump_json(indent=2), encoding="utf-8")
    (output / "task_spec.json").write_text(prepared.task_spec.model_dump_json(indent=2), encoding="utf-8")
    (output / "run_vantage6_task.py").write_text(render_vantage6_script(prepared.task_spec), encoding="utf-8")
    return prepared_path
