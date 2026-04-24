from __future__ import annotations

import random
from typing import Any

from apa.benchmarks.base import BenchmarkSpec, DatasetSplits, mode_size
from apa.constants import OFFICIAL_INVOCATION_CAPS
from apa.retrieval.base import Retriever
from apa.types import RuntimeConfig


def _count_unique_docs(example: dict[str, Any]) -> int:
    return len({fact["key"] for fact in example["supporting_facts"]})


def _load_splits(runtime: RuntimeConfig) -> DatasetSplits:
    from datasets import load_dataset
    import dspy

    rows = load_dataset("hover", trust_remote_code=True)["train"]
    filtered = []
    for row in rows:
        if _count_unique_docs(row) == 3:
            filtered.append(
                dspy.Example(
                    claim=row["claim"],
                    supporting_facts=row["supporting_facts"],
                    label=row["label"],
                ).with_inputs("claim")
            )
    random.Random(0).shuffle(filtered)

    train = filtered[:150]
    val = filtered[150:450]
    test = filtered[450:750]

    t, v, s = mode_size(runtime)
    if t > 0:
        train, val, test = train[:t], val[:v], test[:s]

    return DatasetSplits(train=train, val=val, test=test)


def _program_factory(runtime: RuntimeConfig, retrieval: Retriever | None = None) -> Any:
    import dspy

    del runtime

    if retrieval is None:
        raise ValueError("HoVer requires a retriever")

    def search(query: str, k: int, example: Any | None = None) -> Any:
        result = retrieval.search(query=query, k=k, example=example)
        return dspy.Prediction(passages=result.passages)

    class HoverMultiHop(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            self.k = 7
            self.create_query_hop2 = dspy.ChainOfThought("claim,summary_1->query")
            self.create_query_hop3 = dspy.ChainOfThought("claim,summary_1,summary_2->query")
            self.summarize1 = dspy.ChainOfThought("claim,passages->summary")
            self.summarize2 = dspy.ChainOfThought("claim,context,passages->summary")

        def forward(self, claim: str, example: Any | None = None) -> Any:
            hop1_docs = search(claim, k=self.k, example=example).passages
            summary_1 = self.summarize1(claim=claim, passages=hop1_docs).summary

            hop2_query = self.create_query_hop2(claim=claim, summary_1=summary_1).query
            hop2_docs = search(hop2_query, k=self.k, example=example).passages
            summary_2 = self.summarize2(claim=claim, context=summary_1, passages=hop2_docs).summary

            hop3_query = self.create_query_hop3(claim=claim, summary_1=summary_1, summary_2=summary_2).query
            hop3_docs = search(hop3_query, k=10, example=example).passages

            return dspy.Prediction(retrieved_docs=hop1_docs + hop2_docs + hop3_docs)

    return HoverMultiHop()


def _metric(example: Any, pred: Any, trace: Any | None = None) -> float:
    import dspy

    del trace
    gold_titles = set(map(dspy.evaluate.normalize_text, [fact["key"] for fact in example["supporting_facts"]]))
    found_titles = set(map(dspy.evaluate.normalize_text, [p.split(" | ")[0] for p in pred.retrieved_docs]))
    return float(gold_titles.issubset(found_titles))


def _metric_with_feedback(example: Any, pred: Any, trace: Any | None = None) -> Any:
    import dspy

    del trace
    gold_titles = set(map(dspy.evaluate.normalize_text, [fact["key"] for fact in example["supporting_facts"]]))
    found_titles = set(map(dspy.evaluate.normalize_text, [p.split(" | ")[0] for p in pred.retrieved_docs]))
    found = sorted(gold_titles.intersection(found_titles))
    missing = sorted(gold_titles.difference(found_titles))
    score = float(gold_titles.issubset(found_titles))
    feedback = f"Retrieved relevant: {found}. Missing: {missing}."
    return dspy.Prediction(score=score, feedback=feedback)


def _feedback_map_factory() -> dict[str, Any]:
    import dspy

    def _feedback(
        predictor_output: dict[str, Any],
        predictor_inputs: dict[str, Any],
        module_inputs: Any,
        module_outputs: Any,
        captured_trace: Any,
    ) -> dict[str, Any]:
        del predictor_output, predictor_inputs, captured_trace
        scored = _metric_with_feedback(module_inputs, dspy.Prediction(**module_outputs), None)
        return {"feedback_score": float(scored.score), "feedback_text": str(scored.feedback)}

    return {
        "create_query_hop2.predict": _feedback,
        "create_query_hop3.predict": _feedback,
        "summarize1.predict": _feedback,
        "summarize2.predict": _feedback,
    }


def build_spec() -> BenchmarkSpec:
    return BenchmarkSpec(
        key="hover",
        name="hoverBench",
        description="HoVer multi-hop evidence retrieval benchmark",
        invocation_cap=OFFICIAL_INVOCATION_CAPS["hover"],
        dataset_loader=_load_splits,
        program_factory=_program_factory,
        metric=_metric,
        metric_with_feedback=_metric_with_feedback,
        feedback_map_factory=_feedback_map_factory,
        retrieval_required=True,
    )
