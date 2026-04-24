from __future__ import annotations

from typing import Any

from apa.benchmarks.base import BenchmarkSpec, DatasetSplits, mode_size
from apa.constants import DEFAULT_MODEL, OFFICIAL_INVOCATION_CAPS
from apa.types import RuntimeConfig


_judge_cache: dict[str, Any] = {}
_runtime_model = DEFAULT_MODEL


def _load_splits(runtime: RuntimeConfig) -> DatasetSplits:
    from datasets import load_dataset
    import dspy

    rows = load_dataset("Columbia-NLP/PUPA", "pupa_new")["train"]
    examples = [
        dspy.Example(
            target_response=row["target_response"],
            user_query=row["user_query"],
            pii_str=row["pii_units"],
        ).with_inputs("user_query")
        for row in rows
    ]

    num_train, num_val, num_test = 111, 111, 221
    train_val = examples[: num_train + num_val]
    test = examples[num_train + num_val: num_train + num_val + num_test]

    train = train_val[:num_train]
    val = train_val[num_train:]

    t, v, s = mode_size(runtime)
    if t > 0:
        train, val, test = train[:t], val[:v], test[:s]

    return DatasetSplits(train=train, val=val, test=test)


def _program_factory(runtime: RuntimeConfig, retrieval: Any | None = None) -> Any:
    import dspy

    del retrieval

    global _runtime_model
    _runtime_model = runtime.model

    class CraftRedactedRequest(dspy.Signature):
        """Create a privacy-preserving request for an external LLM."""

        user_query = dspy.InputField()
        llm_request = dspy.OutputField()

    class RespondToQuery(dspy.Signature):
        """Respond to a user query using external hint response."""

        related_llm_request = dspy.InputField()
        related_llm_response = dspy.InputField()
        user_query = dspy.InputField()
        response = dspy.OutputField()

    class PAPILLON(dspy.Module):
        def __init__(self, untrusted_model: Any):
            super().__init__()
            self.craft_redacted_request = dspy.ChainOfThought(CraftRedactedRequest)
            self.respond_to_query = dspy.Predict(RespondToQuery)
            self.untrusted_model = untrusted_model

        def forward(self, user_query: str) -> Any:
            try:
                llm_request = self.craft_redacted_request(user_query=user_query).llm_request
                llm_response = self.untrusted_model(llm_request)[0]
                response = self.respond_to_query(
                    related_llm_request=llm_request,
                    related_llm_response=llm_response,
                    user_query=user_query,
                ).response
            except Exception:
                return dspy.Prediction(llm_request="", llm_response="", response="")
            return dspy.Prediction(llm_request=llm_request, llm_response=llm_response, response=response)

    untrusted_lm = dspy.LM(runtime.model, temperature=1.0)
    return PAPILLON(untrusted_model=untrusted_lm)


def _build_judge(model: str) -> Any:
    import dspy

    class JudgeQuality(dspy.Signature):
        """Compare quality of response A against response B."""

        user_query = dspy.InputField()
        response_A = dspy.InputField()
        response_B = dspy.InputField()
        judgment: bool = dspy.OutputField()

    class JudgeLeakage(dspy.Signature):
        """Count leaked PII units in prompt."""

        pii = dspy.InputField()
        prompt = dspy.InputField()
        num_pii_leaked: int = dspy.OutputField()

    class LLMJudge(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            self.quality_judge = dspy.ChainOfThought(JudgeQuality)
            self.fact_checker = dspy.ChainOfThought(JudgeLeakage)

        def forward(self, user_query: str, og_resp: str, new_resp: str, updated_query: str, pii_str: str) -> Any:
            j1 = self.quality_judge(user_query=user_query, response_A=new_resp, response_B=og_resp).judgment
            j2 = self.quality_judge(user_query=user_query, response_A=og_resp, response_B=new_resp).judgment
            quality = bool(j1 or (j1 == j2))
            pii = list(set(str(pii_str).split("||"))) if pii_str else []
            leakage = self.fact_checker(pii=pii, prompt=updated_query).num_pii_leaked
            leakage_score = float(leakage / max(1, len(pii)))
            return dspy.Prediction(quality=quality, leakage=leakage_score)

    judge = LLMJudge()
    judge.set_lm(dspy.LM(model, temperature=1.0))
    return judge


def _get_judge(model: str) -> Any:
    if model not in _judge_cache:
        _judge_cache[model] = _build_judge(model)
    return _judge_cache[model]


def _metric_with_feedback(example: Any, pred: Any, trace: Any | None = None) -> Any:
    import dspy

    judge = _get_judge(_runtime_model)
    metrics = judge(
        user_query=example.user_query,
        new_resp=pred.response,
        og_resp=example.target_response,
        updated_query=pred.llm_request,
        pii_str=example.pii_str,
    )
    overall = (float(metrics.quality) + (1.0 - float(metrics.leakage))) / 2.0
    score = float(overall >= 1.0) if trace is not None else float(overall)
    feedback = (
        f"Overall={overall:.2f}; quality={float(metrics.quality):.2f}; "
        f"privacy={1.0 - float(metrics.leakage):.2f}."
    )
    return dspy.Prediction(score=score, feedback=feedback)


def _metric(example: Any, pred: Any, trace: Any | None = None) -> float:
    return float(_metric_with_feedback(example, pred, trace).score)


def _extra_lm_collectors() -> list[Any]:
    return [judge.lm for judge in _judge_cache.values() if hasattr(judge, "lm")]


def build_spec() -> BenchmarkSpec:
    return BenchmarkSpec(
        key="pupa",
        name="Papillon",
        description="Privacy-Conscious Delegation benchmark",
        invocation_cap=OFFICIAL_INVOCATION_CAPS["pupa"],
        dataset_loader=_load_splits,
        program_factory=_program_factory,
        metric=_metric,
        metric_with_feedback=_metric_with_feedback,
        retrieval_required=False,
        metadata={"extra_lm_collectors": _extra_lm_collectors},
    )
