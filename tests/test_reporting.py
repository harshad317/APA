from __future__ import annotations

from pathlib import Path

from apa.reporting import generate_aggregate_report
from apa.utils import write_json


def test_aggregate_report_generation(tmp_path: Path):
    run_a = tmp_path / "runs" / "aime" / "apa"
    run_b = tmp_path / "runs" / "aime" / "gepa"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)

    write_json(
        run_a / "metrics.json",
        {
            "benchmark": "aime",
            "method": "apa",
            "score": 0.6,
            "canonical_mode": True,
            "cost": {"usd_cost": 1.25},
        },
    )
    write_json(
        run_b / "metrics.json",
        {
            "benchmark": "aime",
            "method": "gepa",
            "score": 0.8,
            "canonical_mode": True,
            "cost": {"usd_cost": 2.5},
        },
    )

    payload = generate_aggregate_report(root=tmp_path / "runs")

    assert payload["num_runs"] == 2
    assert len(payload["rows"]) == 2
    assert (tmp_path / "runs" / "aggregate_report.json").exists()
    assert (tmp_path / "runs" / "aggregate_report.md").exists()
