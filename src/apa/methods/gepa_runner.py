from __future__ import annotations

from typing import Any

from apa.benchmarks.base import BenchmarkSpec
from apa.types import RuntimeConfig
from apa.utils import call_program

from .base import CompileResult


class GEPAMethodRunner:
    key = "gepa"

    def compile(
        self,
        *,
        spec: BenchmarkSpec,
        student: Any,
        trainset: list[Any],
        valset: list[Any],
        runtime: RuntimeConfig,
        task_lm: Any,
        reflection_lm: Any,
    ) -> CompileResult:
        import dspy

        if runtime.dry_run:
            return CompileResult(program=student, compile_invocations=0, artifacts={"mode": "dry_run"})

        max_metric_calls = runtime.invocation_cap or spec.invocation_cap
        metric = self._build_gepa_metric(spec)

        teleprompter = dspy.GEPA(
            metric=metric,
            auto=None,
            max_metric_calls=max_metric_calls,
            reflection_minibatch_size=8,
            reflection_lm=reflection_lm,
            use_merge=True,
            max_merge_invocations=5,
            track_stats=True,
            seed=runtime.seed,
            log_dir=str(runtime.output_dir / "gepa_logs") if runtime.output_dir else None,
        )

        with dspy.context(lm=task_lm):
            optimized = teleprompter.compile(student, trainset=trainset, valset=valset)

        artifacts = {
            "max_metric_calls": max_metric_calls,
            "has_detailed_results": hasattr(optimized, "detailed_results"),
        }
        if hasattr(optimized, "detailed_results") and hasattr(optimized.detailed_results, "to_dict"):
            try:
                artifacts["detailed_results"] = optimized.detailed_results.to_dict()
            except Exception:
                artifacts["detailed_results"] = {"error": "failed_to_serialize"}

        compile_invocations = int(max_metric_calls)
        if "detailed_results" in artifacts:
            compile_invocations = int(artifacts["detailed_results"].get("total_metric_calls", max_metric_calls))

        return CompileResult(program=optimized, compile_invocations=compile_invocations, artifacts=artifacts)

    def predict(self, compiled_program: Any, example: Any) -> Any:
        return call_program(compiled_program, example)

    @staticmethod
    def _build_gepa_metric(spec: BenchmarkSpec):
        import dspy

        def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
            del pred_name, pred_trace

            if spec.metric_with_feedback is not None:
                scored = spec.metric_with_feedback(gold, pred, trace)
                if hasattr(scored, "score"):
                    feedback = getattr(scored, "feedback", None)
                    if feedback is None:
                        feedback = f"This trajectory got a score of {float(scored.score)}."
                    return dspy.Prediction(score=float(scored.score), feedback=str(feedback))
                if isinstance(scored, dict) and "score" in scored:
                    feedback = scored.get("feedback") or f"This trajectory got a score of {float(scored['score'])}."
                    return dspy.Prediction(score=float(scored["score"]), feedback=str(feedback))

            score = float(spec.metric(gold, pred, trace))
            return dspy.Prediction(score=score, feedback=f"This trajectory got a score of {score}.")

        return metric
