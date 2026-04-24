from __future__ import annotations

import random
from typing import Any

from apa.benchmarks.base import BenchmarkSpec, DatasetSplits, mode_size
from apa.constants import OFFICIAL_INVOCATION_CAPS
from apa.types import RuntimeConfig


def _load_splits(runtime: RuntimeConfig) -> DatasetSplits:
    from datasets import load_dataset
    import dspy

    train_split = [
        dspy.Example(
            problem=x["problem"],
            solution=x.get("solution", ""),
            answer=str(x["answer"]),
        ).with_inputs("problem")
        for x in load_dataset("AI-MO/aimo-validation-aime")["train"]
    ]
    random.Random(0).shuffle(train_split)

    test_split = [
        dspy.Example(problem=x["problem"], answer=str(x["answer"])).with_inputs("problem")
        for x in load_dataset("MathArena/aime_2025")["train"]
    ]

    mid = len(train_split) // 2
    train = train_split[:mid]
    val = train_split[mid:]
    test = test_split * 5

    t, v, s = mode_size(runtime)
    if t > 0:
        train, val, test = train[:t], val[:v], test[:s]

    return DatasetSplits(train=train, val=val, test=test)


def _program_factory(runtime: RuntimeConfig, retrieval: Any | None = None) -> Any:
    import dspy

    del runtime, retrieval

    class SolveAIME(dspy.Signature):
        """Solve the problem and output final integer answer only."""

        problem = dspy.InputField()
        answer = dspy.OutputField()

    return dspy.ChainOfThought(SolveAIME)


def _metric(example: Any, pred: Any, trace: Any | None = None) -> float:
    del trace
    try:
        gt = int(str(example["answer"]))
        pr = int(str(pred.answer).strip())
        return float(gt == pr)
    except Exception:
        return 0.0


def _metric_with_feedback(example: Any, pred: Any, trace: Any | None = None) -> Any:
    import dspy

    del trace
    gt = str(example["answer"])
    score = _metric(example, pred, None)
    text = f"Ground truth: {gt}."
    if score >= 1.0:
        text = "Correct. " + text
    else:
        text = f"Incorrect. Predicted '{pred.answer}'. " + text
    if "solution" in example and example["solution"]:
        text += " Use the reference solution to refine final-number extraction."
    return dspy.Prediction(score=score, feedback=text)


def build_spec() -> BenchmarkSpec:
    return BenchmarkSpec(
        key="aime",
        name="AIMEBench",
        description="AIME prompt optimization benchmark",
        invocation_cap=OFFICIAL_INVOCATION_CAPS["aime"],
        dataset_loader=_load_splits,
        program_factory=_program_factory,
        metric=_metric,
        metric_with_feedback=_metric_with_feedback,
        retrieval_required=False,
    )
