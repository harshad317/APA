from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from apa.benchmarks.base import BenchmarkSpec
from apa.types import RuntimeConfig


@dataclass(slots=True)
class CompileResult:
    program: Any
    compile_invocations: int = 0
    artifacts: dict[str, Any] = field(default_factory=dict)


class MethodRunner(Protocol):
    key: str

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
        ...

    def predict(self, compiled_program: Any, example: Any) -> Any:
        ...
