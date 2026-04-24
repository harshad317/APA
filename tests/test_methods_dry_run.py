from __future__ import annotations

import dspy

from apa.benchmarks.base import BenchmarkSpec, DatasetSplits
from apa.methods.apa_runner import APAMethodRunner
from apa.methods.gepa_runner import GEPAMethodRunner
from apa.methods.mipro_runner import MIPROv2MethodRunner
from apa.types import RuntimeConfig


def _spec() -> BenchmarkSpec:
    class ToyProgram(dspy.Module):
        def forward(self, question: str):
            return dspy.Prediction(answer=question)

    def loader(_runtime):
        ex = dspy.Example(question="hello", answer="hello").with_inputs("question")
        return DatasetSplits(train=[ex], val=[ex], test=[ex])

    def factory(_runtime, retrieval=None):
        del retrieval
        return ToyProgram()

    def metric(example, pred, trace=None):
        del trace
        return float(pred.answer == example.answer)

    return BenchmarkSpec(
        key="toy",
        name="Toy",
        description="toy",
        invocation_cap=5,
        dataset_loader=loader,
        program_factory=factory,
        metric=metric,
        retrieval_required=False,
    )


def _runtime() -> RuntimeConfig:
    return RuntimeConfig(model="openai/gpt-4.1-mini-2025-04-14", seed=0, dry_run=True, dataset_mode="tiny")


def test_all_method_runners_support_dry_run_compile():
    spec = _spec()
    runtime = _runtime()
    splits = spec.dataset_loader(runtime)
    student = spec.program_factory(runtime)

    task_lm = dspy.LM(runtime.model, temperature=0.0, cache=True)
    reflection_lm = dspy.LM(runtime.model, temperature=1.0, cache=True)

    for runner in [APAMethodRunner(), GEPAMethodRunner(), MIPROv2MethodRunner()]:
        result = runner.compile(
            spec=spec,
            student=student,
            trainset=splits.train,
            valset=splits.val,
            runtime=runtime,
            task_lm=task_lm,
            reflection_lm=reflection_lm,
        )
        assert result.compile_invocations == 0
