from __future__ import annotations

from pathlib import Path

import dspy

from apa.benchmarks.base import BenchmarkSpec, DatasetSplits
from apa.orchestrator import run_single
from apa.types import RuntimeConfig


def _build_toy_spec() -> BenchmarkSpec:
    class ToyProgram(dspy.Module):
        def forward(self, question: str):
            return dspy.Prediction(answer=question)

    def loader(runtime):
        del runtime
        ex = dspy.Example(question="q", answer="q").with_inputs("question")
        return DatasetSplits(train=[ex], val=[ex], test=[ex])

    def factory(runtime, retrieval=None):
        del runtime, retrieval
        return ToyProgram()

    def metric(example, pred, trace=None):
        del trace
        return float(example.answer == pred.answer)

    return BenchmarkSpec(
        key="toybench",
        name="ToyBench",
        description="toy",
        invocation_cap=5,
        dataset_loader=loader,
        program_factory=factory,
        metric=metric,
        retrieval_required=False,
    )


def test_run_single_dry_run(monkeypatch, tmp_path: Path):
    spec = _build_toy_spec()
    monkeypatch.setattr("apa.orchestrator.get_benchmark_spec", lambda key: spec)

    runtime = RuntimeConfig(
        model="openai/gpt-4.1-mini-2025-04-14",
        seed=0,
        dry_run=True,
        dataset_mode="tiny",
    )

    result = run_single(benchmark="toybench", method="apa", runtime=runtime, run_dir=tmp_path / "run")
    assert result.benchmark == "toybench"
    assert result.method == "apa"
    assert result.score == 0.0
