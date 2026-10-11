"""Render the installed, version-bound review template with three literal slots."""
import inspect
import re
from pprint import pformat
from .confirmation import task_confirmation_id, task_confirmation_material
from .execution_templates import TemplateRegistry, asset_path, validate_installed
from .runs.submission import submit_task


def render_vantage6_script(task_spec):
    reference = task_spec.execution_template or TemplateRegistry().select()
    validate_installed(reference)
    template = asset_path(reference.export_asset).read_text(encoding="utf-8")
    slots = {
        "__CONFIRMATION_MATERIAL__": pformat(task_confirmation_material(task_spec), width=88, sort_dicts=False),
        "__CONFIRMATION_ID__": repr(task_confirmation_id(task_spec)),
        "__SUBMISSION_CODE__": inspect.getsource(submit_task).rstrip("\n"),
    }
    if any(template.count(slot) != 1 for slot in slots):
        raise ValueError("Template must contain exactly the three data slots")
    # A single substitution pass cannot interpret placeholder text inside user data.
    return re.sub("|".join(map(re.escape, slots)), lambda match: slots[match[0]], template)
