from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


Comparator = Callable[[float, float], bool]


def _cmp_gt(a: float, b: float) -> bool:
    return a > b


def _cmp_gte(a: float, b: float) -> bool:
    return a >= b


def _cmp_lt(a: float, b: float) -> bool:
    return a < b


def _cmp_lte(a: float, b: float) -> bool:
    return a <= b


def _cmp_eq(a: float, b: float) -> bool:
    return a == b


COMPARATORS: dict[str, Comparator] = {
    ">": _cmp_gt,
    ">=": _cmp_gte,
    "<": _cmp_lt,
    "<=": _cmp_lte,
    "==": _cmp_eq,
}


@dataclass(slots=True)
class Guard:
    feature: str
    op: str
    value: float

    def evaluate(self, features: dict[str, Any]) -> bool:
        if self.op not in COMPARATORS:
            return False
        raw = features.get(self.feature)
        if raw is None:
            return False
        try:
            return COMPARATORS[self.op](float(raw), float(self.value))
        except Exception:
            return False


@dataclass(slots=True)
class State:
    state_id: str
    template: str
    terminal: bool = False


@dataclass(slots=True)
class Transition:
    source: str
    target: str
    guard: Guard
    priority: int = 0

    def fires(self, features: dict[str, Any]) -> bool:
        return self.guard.evaluate(features)


@dataclass(slots=True)
class ExecutionStep:
    state_id: str
    prompt_fragment: str
    score: float
    features: dict[str, Any]
    next_state: str | None = None


@dataclass(slots=True)
class ExecutionTrace:
    steps: list[ExecutionStep] = field(default_factory=list)

    @property
    def final_score(self) -> float:
        return self.steps[-1].score if self.steps else 0.0

    @property
    def path(self) -> list[str]:
        return [step.state_id for step in self.steps]


@dataclass(slots=True)
class Automaton:
    states: dict[str, State]
    transitions: list[Transition]
    start_state: str

    def validate(self) -> None:
        if self.start_state not in self.states:
            raise ValueError(f"Unknown start state: {self.start_state}")
        for transition in self.transitions:
            if transition.source not in self.states:
                raise ValueError(f"Transition source does not exist: {transition.source}")
            if transition.target not in self.states:
                raise ValueError(f"Transition target does not exist: {transition.target}")

    def outgoing(self, state_id: str) -> list[Transition]:
        edges = [t for t in self.transitions if t.source == state_id]
        return sorted(edges, key=lambda t: t.priority)
