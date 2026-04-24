from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from apa.types import RuntimeConfig


Example = Any


@dataclass(slots=True)
class DatasetSplits:
    train: list[Example]
    val: list[Example]
    test: list[Example]


class ProgramFactory(Protocol):
    def __call__(self, runtime: RuntimeConfig, retrieval: Any | None = None) -> Any:
        ...


MetricFn = Callable[[Any, Any, Any | None], float]
MetricWithFeedbackFn = Callable[[Any, Any, Any | None], Any]
FeedbackMapFactory = Callable[[], dict[str, Callable[..., dict[str, Any]]]]


@dataclass(slots=True)
class BenchmarkSpec:
    key: str
    name: str
    description: str
    invocation_cap: int
    dataset_loader: Callable[[RuntimeConfig], DatasetSplits]
    program_factory: ProgramFactory
    metric: MetricFn
    metric_with_feedback: MetricWithFeedbackFn | None = None
    feedback_map_factory: FeedbackMapFactory | None = None
    retrieval_required: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


def trim_dataset(dataset: list[Any], size: int, seed: int = 1) -> list[Any]:
    if len(dataset) <= size:
        return dataset
    import random

    rng = random.Random(seed)
    return rng.sample(dataset, size)


def mode_size(runtime: RuntimeConfig) -> tuple[int, int, int]:
    mode = runtime.dataset_mode.lower()
    if runtime.smoke or mode == "smoke":
        return 8, 8, 8
    if mode == "tiny":
        return 24, 24, 24
    if mode == "test":
        return 50, 50, 50
    return -1, -1, -1
