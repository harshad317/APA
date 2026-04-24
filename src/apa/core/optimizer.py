from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any, Callable

from .automaton import Automaton, Guard, State, Transition
from .executor import AutomatonExecutor, ExecutorConfig
from .mutation import mutate_automaton


@dataclass(slots=True)
class APAConfig:
    model_id: str | None = None
    seed: int = 0
    max_states: int = 6
    max_metric_calls: int = 200
    reflection_minibatch_size: int = 8
    max_steps_per_episode: int = 3
    population_limit: int = 24
    guard_delta: float = 0.15
    add_state_prob: float = 0.15
    mutate_guard_prob: float = 0.6
    mutate_text_prob: float = 0.9
    cache_path: str | None = None


@dataclass(slots=True)
class Candidate:
    automaton: Automaton
    train_score: float
    val_score: float
    metric_calls: int
    traces: list[dict[str, Any]] = field(default_factory=list)


Example = Any
Dataset = list[Example]
EvaluateEpisode = Callable[[Automaton, Example, int], tuple[float, dict[str, Any], list[str]]]


class APAOptimizer:
    """Evolutionary optimizer over prompt automatons with Pareto-like retention."""

    def __init__(self, config: APAConfig | None = None) -> None:
        self.config = config or APAConfig()
        self.rng = random.Random(self.config.seed)
        self.metric_calls = 0

    def compile(
        self,
        trainset: Dataset,
        valset: Dataset,
        evaluate_episode: EvaluateEpisode,
    ) -> Candidate:
        seed_automaton = self._seed_automaton()
        population: list[Candidate] = []

        seed_candidate = self._evaluate_candidate(seed_automaton, trainset, valset, evaluate_episode)
        population.append(seed_candidate)
        best = seed_candidate

        while self.metric_calls < self.config.max_metric_calls:
            parent = self._select_parent(population)
            child_automaton = mutate_automaton(
                parent.automaton,
                self.rng,
                config=_build_mutation_config(self.config),
            )
            child = self._evaluate_candidate(child_automaton, trainset, valset, evaluate_episode)
            population = self._retain(population + [child])
            if child.val_score >= best.val_score:
                best = child

            if self.metric_calls >= self.config.max_metric_calls:
                break

        return best

    def _evaluate_candidate(
        self,
        automaton: Automaton,
        trainset: Dataset,
        valset: Dataset,
        evaluate_episode: EvaluateEpisode,
    ) -> Candidate:
        train_batch = self._sample(trainset, self.config.reflection_minibatch_size)
        train_score, train_traces = self._evaluate_on_dataset(automaton, train_batch, evaluate_episode)
        val_score, val_traces = self._evaluate_on_dataset(automaton, valset, evaluate_episode)
        return Candidate(
            automaton=copy.deepcopy(automaton),
            train_score=train_score,
            val_score=val_score,
            metric_calls=self.metric_calls,
            traces=train_traces + val_traces,
        )

    def _evaluate_on_dataset(
        self,
        automaton: Automaton,
        dataset: Dataset,
        evaluate_episode: EvaluateEpisode,
    ) -> tuple[float, list[dict[str, Any]]]:
        if not dataset:
            return 0.0, []

        total = 0.0
        traces: list[dict[str, Any]] = []

        for ex_idx, example in enumerate(dataset):
            if self.metric_calls >= self.config.max_metric_calls:
                break
            score, features, path = evaluate_episode(automaton, example, ex_idx)
            self.metric_calls += 1
            total += score
            traces.append({"example_index": ex_idx, "score": score, "features": features, "path": path})

        denom = max(1, len(traces))
        return total / denom, traces

    def _sample(self, dataset: Dataset, k: int) -> Dataset:
        if len(dataset) <= k:
            return list(dataset)
        return self.rng.sample(dataset, k)

    def _select_parent(self, population: list[Candidate]) -> Candidate:
        by_val = sorted(population, key=lambda c: c.val_score, reverse=True)
        top = by_val[: min(4, len(by_val))]
        return self.rng.choice(top)

    def _retain(self, candidates: list[Candidate]) -> list[Candidate]:
        ranked = sorted(candidates, key=lambda c: (c.val_score, c.train_score), reverse=True)
        return ranked[: self.config.population_limit]

    def _seed_automaton(self) -> Automaton:
        states = {
            "draft": State(
                state_id="draft",
                template="Answer the task accurately. Keep the final response format strictly valid.",
                terminal=False,
            ),
            "verify": State(
                state_id="verify",
                template="Re-check constraints and produce the same answer only if consistent.",
                terminal=True,
            ),
            "finish": State(
                state_id="finish",
                template="Return the final answer concisely and in required format.",
                terminal=True,
            ),
        }
        transitions = [
            Transition(
                source="draft",
                target="verify",
                guard=Guard(feature="quality_proxy", op="<", value=0.65),
                priority=0,
            ),
            Transition(
                source="draft",
                target="finish",
                guard=Guard(feature="quality_proxy", op=">=", value=0.65),
                priority=1,
            ),
            Transition(
                source="verify",
                target="finish",
                guard=Guard(feature="quality_proxy", op=">=", value=0.0),
                priority=0,
            ),
        ]
        return Automaton(states=states, transitions=transitions, start_state="draft")


def build_episode_executor(automaton: Automaton, max_steps: int) -> AutomatonExecutor:
    return AutomatonExecutor(automaton=automaton, config=ExecutorConfig(max_steps=max_steps))


def _build_mutation_config(config: APAConfig):
    from .mutation import MutationConfig

    return MutationConfig(
        guard_delta=config.guard_delta,
        add_state_prob=config.add_state_prob,
        mutate_guard_prob=config.mutate_guard_prob,
        mutate_text_prob=config.mutate_text_prob,
        max_states=config.max_states,
    )
