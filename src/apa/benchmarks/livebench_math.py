from __future__ import annotations

import random
from typing import Any

from apa.benchmarks.base import BenchmarkSpec, DatasetSplits, mode_size
from apa.constants import OFFICIAL_INVOCATION_CAPS
from apa.types import RuntimeConfig

from .livebench_math_utils.metric import calculate_livebench_score


def _load_splits(runtime: RuntimeConfig) -> DatasetSplits:
    from datasets import load_dataset
    import dspy

    rows = load_dataset("livebench/math")["test"]
    dataset = [
        dspy.Example(
            question=item["turns"][0],
            answer=item["ground_truth"],
            question_d=item,
        ).with_inputs("question")
        for item in rows
    ]
    random.Random(0).shuffle(dataset)

    n = len(dataset)
    train = dataset[: int(0.33 * n)]
    val = dataset[int(0.33 * n): int(0.66 * n)]
    test = dataset[int(0.66 * n):]

    t, v, s = mode_size(runtime)
    if t > 0:
        train, val, test = train[:t], val[:v], test[:s]

    return DatasetSplits(train=train, val=val, test=test)


def _program_factory(runtime: RuntimeConfig, retrieval: Any | None = None) -> Any:
    import dspy

    del runtime, retrieval

    class SolveLiveBenchMath(dspy.Signature):
        """Solve the problem and provide the answer in required format."""

        question = dspy.InputField()
        answer = dspy.OutputField()

    return dspy.ChainOfThought(SolveLiveBenchMath)


def _metric(example: Any, pred: Any, trace: Any | None = None) -> float:
    del trace
    score, _ = calculate_livebench_score(example["question_d"], str(pred.answer), debug=False)
    return float(score)


def _metric_with_feedback(example: Any, pred: Any, trace: Any | None = None) -> Any:
    import dspy

    del trace
    score, feedback = calculate_livebench_score(example["question_d"], str(pred.answer), debug=True)
    return dspy.Prediction(score=float(score), feedback=str(feedback))


def build_spec() -> BenchmarkSpec:
    return BenchmarkSpec(
        key="livebench_math",
        name="LiveBenchMathBench",
        description="LiveBench math subset",
        invocation_cap=OFFICIAL_INVOCATION_CAPS["livebench_math"],
        dataset_loader=_load_splits,
        program_factory=_program_factory,
        metric=_metric,
        metric_with_feedback=_metric_with_feedback,
        retrieval_required=False,
    )
