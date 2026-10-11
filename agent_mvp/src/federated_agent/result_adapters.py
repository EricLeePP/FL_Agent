"""Result adapters: how a completed task's result is checked and reported.

An algorithm's catalog entry names its adapter (``execution.result_adapter``).
Two adapters are built in, and an installed package can add more through the
``federated_agent.result_adapters`` entry-point group. Registry JSON only names
an adapter; the adapter itself is trusted code installed with the software, and
a plugin can never replace a built-in adapter. The agent never branches on an
algorithm's identity: it asks the named adapter.
"""

import logging
import re
from functools import lru_cache
from importlib.metadata import entry_points
from pathlib import Path
from typing import Protocol

from .models import ADAPTER_NAME_PATTERN
from .result_parser import ResultParser

logger = logging.getLogger(__name__)
ENTRY_POINT_GROUP = "federated_agent.result_adapters"
NAME_PATTERN = ADAPTER_NAME_PATTERN


class ResultAdapter(Protocol):
    name: str
    reports: bool
    """True when ``write_report`` produces a report for the researcher."""

    def check(self, raw, plan, semantic_catalog) -> None:
        """Raise ValueError, TypeError or KeyError when ``raw`` does not answer the resolved ``plan``."""

    def write_report(self, raw, output: Path, *, semantic_catalog, context: dict, chart: bool, emit) -> Path | None:
        """Write report files into ``output`` and return the main report, if any."""


class RawJSONResult:
    """Keep the raw result only; there is no generic way to check or summarize it."""

    name = "raw_json"
    reports = False

    def check(self, raw, plan, semantic_catalog):
        return None

    def write_report(self, raw, output, *, semantic_catalog, context, chart, emit):
        return None


class DescriptiveStatisticsResult:
    """STRONG AYA descriptive statistics: checked tables, an optional chart and an HTML report."""

    name = "descriptive_statistics_v1"
    reports = True

    def check(self, raw, plan, semantic_catalog):
        parser = ResultParser(semantic_catalog)
        parser.validate_requested(parser.parse(raw), plan)

    def write_report(self, raw, output, *, semantic_catalog, context, chart, emit):
        parser = ResultParser(semantic_catalog)
        parsed = parser.parse(raw or {})
        (output / "parsed_result.json").write_text(parsed.model_dump_json(indent=2), encoding="utf-8")
        image = None
        if chart:
            try:
                image = parser.write_chart(parsed, output / "result.png")
            except (ImportError, ValueError, OSError):
                emit("Chart unavailable; the checked tables and raw result remain available.")
        report = parser.write_html(parsed, output / "result.html", chart=image, context=context)
        for row in parsed.numerical:
            if row.get("statistic") in {"count", "mean", "na", "adjusted std"}:
                emit(f"  {row.get('variable_name', row['variable'])}: {row['statistic']} = {row['value']}")
        for row in parsed.categorical:
            emit(f"  {row.get('variable_name', row['variable'])}: {row['value']} = {row['count']}")
        return report


BUILT_IN = {adapter.name: adapter for adapter in (RawJSONResult(), DescriptiveStatisticsResult())}
_registered: dict = {}


def _validate(adapter):
    if not isinstance(getattr(adapter, "name", None), str) or not re.fullmatch(NAME_PATTERN, adapter.name):
        raise ValueError("A result adapter needs a lower-case identifier as its name")
    if not isinstance(getattr(adapter, "reports", None), bool):
        raise ValueError("A result adapter must state whether it reports")
    if not callable(getattr(adapter, "check", None)) or not callable(getattr(adapter, "write_report", None)):
        raise ValueError("A result adapter must implement check() and write_report()")
    if adapter.name in BUILT_IN:
        raise ValueError(f"Result adapter {adapter.name!r} is built in and cannot be replaced")
    return adapter


@lru_cache(maxsize=1)
def _plugins() -> dict:
    found = {}
    for entry in entry_points(group=ENTRY_POINT_GROUP):
        try:
            loaded = entry.load()
            adapter = _validate(loaded() if isinstance(loaded, type) else loaded)
        except Exception as exc:
            logger.warning("Ignoring result adapter plugin %r: %s", entry.name, exc)
            continue
        if adapter.name in found:
            logger.warning("Ignoring duplicate result adapter plugin %r", adapter.name)
            continue
        found[adapter.name] = adapter
    return found


def register(adapter) -> None:
    """Install ``adapter`` in this process, for embedding applications and tests."""
    _registered[_validate(adapter).name] = adapter


def installed() -> dict:
    """Every usable adapter by name; built-in adapters always win."""
    return {**_plugins(), **_registered, **BUILT_IN}


def result_adapter(name: str):
    """The installed adapter called ``name``; KeyError when it is not installed."""
    adapters = installed()
    if name not in adapters:
        raise KeyError(name)
    return adapters[name]
