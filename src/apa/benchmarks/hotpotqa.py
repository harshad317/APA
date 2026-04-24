from __future__ import annotations

import random
from typing import Any

from apa.benchmarks.base import BenchmarkSpec, DatasetSplits, mode_size
from apa.constants import OFFICIAL_INVOCATION_CAPS
from apa.retrieval.base import Retriever
from apa.types import RuntimeConfig


def _load_splits(runtime: RuntimeConfig) -> DatasetSplits:
    from datasets import load_dataset
    import dspy

    rows = load_dataset("hotpot_qa", "fullwiki", trust_remote_code=True)["train"]
    examples = [dspy.Example(**row).with_inputs("question") for row in rows]

    # Official protocol uses 150 / 300 / 300 sampled splits.
    random.Random(0).shuffle(examples)
    train = examples[:150]
    val = examples[150:450]
    test = examples[450:750]

    t, v, s = mode_size(runtime)
    if t > 0:
        train, val, test = train[:t], val[:v], test[:s]

    return DatasetSplits(train=train, val=val, test=test)


def _program_factory(runtime: RuntimeConfig, retrieval: Retriever | None = None) -> Any:
    import dspy

    del runtime

    if retrieval is None:
        raise ValueError("HotPotQA requires a retriever")

    def search(query: str, k: int, example: Any | None = None) -> Any:
        result = retrieval.search(query=query, k=k, example=example)
        return dspy.Prediction(passages=result.passages)

    class HotpotMultiHop(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            self.k = 7
            self.create_query_hop2 = dspy.ChainOfThought("question,summary_1->query")
            self.final_answer = dspy.ChainOfThought("question,summary_1,summary_2->answer")
            self.summarize1 = dspy.ChainOfThought("question,passages->summary")
            self.summarize2 = dspy.ChainOfThought("question,context,passages->summary")

        def forward(self, question: str, example: Any | None = None) -> Any:
            hop1_docs = search(question, k=self.k, example=example).passages
            summary_1 = self.summarize1(question=question, passages=hop1_docs).summary

            hop2_query = self.create_query_hop2(question=question, summary_1=summary_1).query
            hop2_docs = search(hop2_query, k=self.k, example=example).passages
            summary_2 = self.summarize2(question=question, context=summary_1, passages=hop2_docs).summary

            answer = self.final_answer(question=question, summary_1=summary_1, summary_2=summary_2).answer
            return dspy.Prediction(answer=answer, hop1_docs=hop1_docs, hop2_docs=hop2_docs)

    return HotpotMultiHop()


def _answer_match(prediction: str, answers: list[str] | str, frac: float = 1.0) -> bool:
    from dspy.dsp.utils import EM, F1

    if frac >= 1.0:
        return bool(EM(prediction, answers))
    return bool(F1(prediction, answers) >= frac)


def _textual_context(example: Any) -> str:
    context = example["context"]
    title_to_sentences = {title: sents for title, sents in zip(context["title"], context["sentences"])}
    useful_titles = set(example["supporting_facts"]["title"])
    lines = []
    for title in useful_titles:
        lines.append(f"{title} | {''.join(title_to_sentences.get(title, []))}")
    return "\n".join(lines)


def _metric(example: Any, pred: Any, trace: Any | None = None) -> float:
    del trace
    answers = example.answer if hasattr(example, "answer") else example["answer"]
    return float(_answer_match(str(pred.answer), answers, frac=1.0))


def _metric_with_feedback(example: Any, pred: Any, trace: Any | None = None) -> Any:
    import dspy

    del trace
    score = _metric(example, pred, None)
    if score >= 1.0:
        feedback = f"Answer '{pred.answer}' is correct.\n{_textual_context(example)}"
    else:
        feedback = f"Answer '{pred.answer}' is incorrect. Correct answer: {example.answer}.\n{_textual_context(example)}"
    return dspy.Prediction(score=score, feedback=feedback)


def _feedback_map_factory() -> dict[str, Any]:
    import dspy

    def _module_feedback(
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
        "create_query_hop2.predict": _module_feedback,
        "final_answer.predict": _module_feedback,
        "summarize1.predict": _module_feedback,
        "summarize2.predict": _module_feedback,
    }


def build_spec() -> BenchmarkSpec:
    return BenchmarkSpec(
        key="hotpotqa",
        name="HotpotQABench",
        description="HotPotQA multi-hop retrieval benchmark",
        invocation_cap=OFFICIAL_INVOCATION_CAPS["hotpotqa"],
        dataset_loader=_load_splits,
        program_factory=_program_factory,
        metric=_metric,
        metric_with_feedback=_metric_with_feedback,
        feedback_map_factory=_feedback_map_factory,
        retrieval_required=True,
    )
