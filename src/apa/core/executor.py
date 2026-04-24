from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .automaton import Automaton, ExecutionStep, ExecutionTrace


class StepEvaluator(Protocol):
    def __call__(self, prompt_fragment: str, context: dict[str, Any], step_index: int) -> tuple[float, dict[str, Any]]:
        ...


@dataclass(slots=True)
class ExecutorConfig:
    max_steps: int = 3


class AutomatonExecutor:
    def __init__(self, automaton: Automaton, config: ExecutorConfig | None = None) -> None:
        self.automaton = automaton
        self.config = config or ExecutorConfig()
        self.automaton.validate()

    def run(self, context: dict[str, Any], evaluate_step: StepEvaluator) -> ExecutionTrace:
        trace = ExecutionTrace()
        state_id = self.automaton.start_state

        for step_idx in range(self.config.max_steps):
            state = self.automaton.states[state_id]
            score, features = evaluate_step(state.template, context, step_idx)
            next_state = None

            if not state.terminal:
                for transition in self.automaton.outgoing(state_id):
                    if transition.fires(features):
                        next_state = transition.target
                        break

            trace.steps.append(
                ExecutionStep(
                    state_id=state.state_id,
                    prompt_fragment=state.template,
                    score=score,
                    features=features,
                    next_state=next_state,
                )
            )

            if state.terminal or not next_state:
                break

            state_id = next_state

        return trace
