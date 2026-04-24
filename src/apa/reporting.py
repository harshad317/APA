from __future__ import annotations

import dataclasses
import statistics
from pathlib import Path
from typing import Any

from apa.types import AggregateReportRow
from apa.utils import read_json, write_json


def discover_run_metrics(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("**/metrics.json")):
        try:
            payload = read_json(path)
            payload["_path"] = str(path)
            records.append(payload)
        except Exception:
            continue
    return records


def aggregate_records(records: list[dict[str, Any]]) -> list[AggregateReportRow]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in records:
        benchmark = str(row.get("benchmark", ""))
        method = str(row.get("method", ""))
        grouped.setdefault((benchmark, method), []).append(row)

    out: list[AggregateReportRow] = []
    for (benchmark, method), rows in sorted(grouped.items()):
        scores = [float(r.get("score", 0.0)) for r in rows]
        costs = [float((r.get("cost") or {}).get("usd_cost", 0.0)) for r in rows]
        canonical_flags = [bool(r.get("canonical_mode", False)) for r in rows]

        out.append(
            AggregateReportRow(
                benchmark=benchmark,
                method=method,
                score_mean=float(statistics.fmean(scores)) if scores else 0.0,
                score_count=len(scores),
                cost_mean=float(statistics.fmean(costs)) if costs else 0.0,
                canonical_mode_all=all(canonical_flags),
            )
        )
    return out


def to_markdown(rows: list[AggregateReportRow]) -> str:
    lines = [
        "| Benchmark | Method | Mean Score | Runs | Mean Cost (USD) | Canonical |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {benchmark} | {method} | {score:.4f} | {count} | {cost:.4f} | {canonical} |".format(
                benchmark=row.benchmark,
                method=row.method,
                score=row.score_mean,
                count=row.score_count,
                cost=row.cost_mean,
                canonical="yes" if row.canonical_mode_all else "no",
            )
        )
    return "\n".join(lines)


def generate_aggregate_report(root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    records = discover_run_metrics(root)
    rows = aggregate_records(records)
    markdown = to_markdown(rows)

    payload = {
        "root": str(root),
        "num_runs": len(records),
        "rows": [dataclasses.asdict(row) for row in rows],
        "markdown": markdown,
    }

    out_dir = output_dir or root
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "aggregate_report.json", payload)
    (out_dir / "aggregate_report.md").write_text(markdown + "\n", encoding="utf-8")

    return payload
