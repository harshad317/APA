from __future__ import annotations

from typing import Any

from apa.benchmarks.base import BenchmarkSpec
from apa.types import RuntimeConfig
from apa.utils import call_program

from .base import CompileResult


class MIPROv2MethodRunner:
    key = "miprov2"

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

        invocation_cap = runtime.invocation_cap or spec.invocation_cap
        num_trials = self._estimate_trials(invocation_cap, train_size=len(trainset), val_size=len(valset))

        teleprompter = dspy.MIPROv2(
            metric=spec.metric,
            prompt_model=reflection_lm,
            task_model=task_lm,
            auto="heavy",
            seed=runtime.seed,
            track_stats=True,
            verbose=False,
        )

        with dspy.context(lm=task_lm):
            optimized = teleprompter.compile(
                student,
                trainset=trainset,
                valset=valset,
                num_trials=num_trials,
                seed=runtime.seed,
                requires_permission_to_run=False,
            )

        artifacts = {
            "invocation_cap": invocation_cap,
            "num_trials": num_trials,
        }
        return CompileResult(program=optimized, compile_invocations=num_trials, artifacts=artifacts)

    def predict(self, compiled_program: Any, example: Any) -> Any:
        return call_program(compiled_program, example)

    @staticmethod
    def _estimate_trials(invocation_cap: int, train_size: int, val_size: int) -> int:
        full_eval_size = max(1, train_size + val_size)
        trials = max(1, invocation_cap // full_eval_size)
        return int(min(64, trials))
