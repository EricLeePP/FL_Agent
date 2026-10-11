"""Parse STRONG AYA result JSON and create a small self-contained report."""

import html
import json
import math
import os
from pathlib import Path
from typing import Any

from .models import ParsedResult
from .semantic_catalog import SemanticCatalog


def _column_json_to_records(value: str | dict[str, Any]) -> list[dict[str, Any]]:
    columns = json.loads(value) if isinstance(value, str) else value
    if not isinstance(columns, dict) or not all(isinstance(column, dict) for column in columns.values()):
        raise ValueError("Statistics must use the registered column-oriented schema")
    if not columns:
        return []
    row_ids = list(next(iter(columns.values())).keys())
    if any(set(values) != set(row_ids) for values in columns.values()):
        raise ValueError("Statistics columns have inconsistent row indices")
    return [
        {column: values.get(row_id) for column, values in columns.items()}
        for row_id in row_ids
    ]


class ResultParser:
    def __init__(self, semantic_catalog: SemanticCatalog) -> None:
        self.semantic_catalog = semantic_catalog

    def parse(self, raw: Any) -> ParsedResult:
        raw = self._unwrap_vantage6_result(raw)
        numerical = _column_json_to_records(
            raw.get("numerical_general_statistics", {})
        )
        categorical = _column_json_to_records(
            raw.get("categorical_general_statistics", {})
        )
        for row in [*numerical, *categorical]:
            variable = self.semantic_catalog.get_by_class(str(row.get("variable", "")))
            if variable:
                row["variable_name"] = variable.canonical_name
        return ParsedResult(numerical=numerical, categorical=categorical,
                            numerical_present="numerical_general_statistics" in raw,
                            categorical_present="categorical_general_statistics" in raw)

    @staticmethod
    def _unwrap_vantage6_result(raw: Any) -> dict[str, Any]:
        for _ in range(10):
            if isinstance(raw, str):
                raw = json.loads(raw)
            elif isinstance(raw, list):
                if len(raw) != 1:
                    raise ValueError("Expected exactly one central result")
                raw = raw[0]
            elif isinstance(raw, dict):
                if raw.get("error") or raw.get("errors"):
                    raise ValueError("Backend result contains an error")
                if "data" in raw:
                    raw = raw["data"]
                elif "result" in raw:
                    raw = raw["result"]
                elif {"numerical_general_statistics", "categorical_general_statistics"} & raw.keys():
                    return raw
                else:
                    raise ValueError("Result does not match the statistics contract")
            else:
                raise ValueError("Vantage6 result payload must resolve to an object")
        raise ValueError("Result nesting exceeds the contract limit")

    @staticmethod
    def validate_requested(result, resolved_plan):
        expected = {variable.class_id for variable in resolved_plan.variables}
        returned = {row.get("variable") for row in [*result.numerical, *result.categorical]}
        if not returned <= expected:
            raise ValueError("Unexpected variables in result")
        for variable in resolved_plan.variables:
            if not getattr(result, variable.datatype + "_present"):
                raise ValueError("Requested statistics table is missing; this is not a suppression declaration")
            rows = [row for row in (result.numerical if variable.datatype == "numerical" else result.categorical)
                    if row.get("variable") == variable.class_id]
            if variable.datatype == "numerical":
                by_statistic = {row.get("statistic"): row.get("value") for row in rows}
                if len(by_statistic) != len(rows):
                    raise ValueError("Duplicate numerical statistic")
                for field in ("count", "mean", "na", "outliers", "adjusted std"):
                    value = by_statistic.get(field)
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError("Missing or non-finite requested numerical statistic")
                    if field != "mean" and value < 0:
                        raise ValueError("Negative count or standard deviation")
            else:
                # An empty categorical table can be legitimate local suppression.
                if any(not {"variable", "value", "count"} <= row.keys() for row in rows):
                    raise ValueError("Categorical rows require variable, value and count")
                if len({row.get("value") for row in rows}) != len(rows):
                    raise ValueError("Duplicate category")
                for row in rows:
                    value = row.get("count")
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                        raise ValueError("Invalid category count")

    def write_chart(self, result, output_path):
        os.environ.setdefault("MPLCONFIGDIR", str(Path(output_path).resolve().parent / ".plot-cache"))
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        rows = result.categorical or [row for row in result.numerical if row.get("statistic") == "mean"]
        if not rows:
            return None
        labels = [f"{row.get('variable_name', row['variable'])}: {row['value']}" if result.categorical
                  else str(row.get("variable_name", row["variable"])) for row in rows]
        labels = [label.replace("_", " ") for label in labels]
        values = [row["count"] if result.categorical else row["value"] for row in rows]
        figure, ax = plt.subplots(figsize=(9, max(3, len(rows) * 0.6 + 1)))
        bars = ax.barh(labels, values, color="#187f78", height=0.5)
        ax.bar_label(bars, labels=[f"{value:g}" if result.categorical else f"{value:.2f}" for value in values], padding=5)
        ax.set_xlim(min(0, min(values) * 1.2), max(max(values) * 1.2, 1))
        ax.set_ylim(-0.5, len(rows) - 0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Returned frequency (suppressed categories may be absent)" if result.categorical else "Reported mean (variable-specific units)")
        ax.set_title("Federated aggregate result")
        ax.invert_yaxis()
        figure.tight_layout()
        path = Path(output_path)
        figure.savefig(path, dpi=150)
        plt.close(figure)
        return path

    def write_html(self, result: ParsedResult, output_path: str | Path, *, chart=None, context=None) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = "<dl>" + "".join(f"<dt>{html.escape(str(key))}</dt><dd>{html.escape(str(value))}</dd>" for key, value in (context or {}).items()) + "</dl>"
        sections = ([metadata] if context else []) + ([f'<img src="{html.escape(Path(chart).name)}" alt="Aggregate statistics" style="max-width:100%">'] if chart else []) + [
            self._table("Numerical statistics", result.numerical),
            self._table("Categorical statistics", result.categorical),
        ]
        document = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Federated result</title>
<style>
body{font-family:Arial,sans-serif;max-width:960px;margin:24px auto;padding:0 16px;color:#17202a;overflow-wrap:anywhere}
dl{display:grid;grid-template-columns:110px minmax(0,1fr);gap:8px}dt{font-weight:bold}dd{margin:0}
table{border-collapse:collapse;width:100%;margin:12px 0 28px;display:block;overflow-x:auto}
th,td{border:1px solid #ccd1d1;padding:8px;text-align:left}th{background:#eef2f3}
h1{font-size:26px}h2{font-size:19px;margin-top:28px}
</style></head><body><h1>Federated descriptive statistics</h1>
""" + "<p>Aggregate results only. Counts are non-missing observations, not necessarily unique patients. Categories may be suppressed. Federated quantiles are not exact pooled quantiles.</p>" + "\n".join(sections) + "</body></html>"
        path.write_text(document, encoding="utf-8")
        return path

    @staticmethod
    def _table(title: str, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return f"<h2>{html.escape(title)}</h2><p>No results.</p>"
        columns = list(dict.fromkeys(key for row in rows for key in row))
        header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
        body = "".join(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(row.get(column, '')))}</td>"
                for column in columns
            )
            + "</tr>"
            for row in rows
        )
        return f"<h2>{html.escape(title)}</h2><table><tr>{header}</tr>{body}</table>"
