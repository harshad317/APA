"""Shared dataclasses and protocols."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RuntimeConfig:
    model: str
    seed: int
    smoke: bool = False
    dry_run: bool = False
    dataset_mode: str = "full"
    invocation_cap: int | None = None
    output_dir: Path | None = None


@dataclass(slots=True)
class CostSummary:
    input_tokens: int = 0
    output_tokens: int = 0
    usd_cost: float = 0.0


@dataclass(slots=True)
class RunResult:
    benchmark: str
    method: str
    seed: int
    score: float
    canonical_mode: bool
    retrieval_backend: str
    invocation_count: int
    cost: CostSummary = field(default_factory=CostSummary)
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AggregateReportRow:
    benchmark: str
    method: str
    score_mean: float
    score_count: int
    cost_mean: float
    canonical_mode_all: bool
