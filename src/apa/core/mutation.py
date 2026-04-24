from __future__ import annotations

import copy
import random
from dataclasses import dataclass

from .automaton import Automaton, Guard, State, Transition


MUTATION_SUFFIXES = [
    "Be concise and deterministic.",
    "Explicitly check edge cases before finalizing.",
    "Prefer exact formatting over verbose prose.",
    "If uncertain, self-verify briefly before answering.",
]


@dataclass(slots=True)
class MutationConfig:
    guard_delta: float = 0.15
    add_state_prob: float = 0.15
    mutate_guard_prob: float = 0.6
    mutate_text_prob: float = 0.9
    max_states: int | None = None


def mutate_automaton(automaton: Automaton, rng: random.Random, config: MutationConfig | None = None) -> Automaton:
    cfg = config or MutationConfig()
    child = copy.deepcopy(automaton)

    state_ids = list(child.states.keys())

    if cfg.mutate_text_prob > rng.random() and state_ids:
        chosen = rng.choice(state_ids)
        suffix = rng.choice(MUTATION_SUFFIXES)
        child.states[chosen].template = _merge_fragment(child.states[chosen].template, suffix)

    if cfg.mutate_guard_prob > rng.random() and child.transitions:
        edge = rng.choice(child.transitions)
        direction = -1.0 if rng.random() < 0.5 else 1.0
        edge.guard.value = float(edge.guard.value + direction * cfg.guard_delta)

    can_add_state = cfg.max_states is None or len(state_ids) < cfg.max_states
    if cfg.add_state_prob > rng.random() and state_ids and can_add_state:
        new_state_id = f"state_{len(state_ids)}"
        anchor = rng.choice(state_ids)
        child.states[new_state_id] = State(
            state_id=new_state_id,
            template=child.states[anchor].template + "\nUse a verification-oriented style.",
            terminal=False,
        )
        child.transitions.append(
            Transition(
                source=anchor,
                target=new_state_id,
                guard=Guard(feature="quality_proxy", op="<", value=0.65),
                priority=99,
            )
        )

    child.validate()
    return child


def _merge_fragment(base: str, extra: str) -> str:
    if extra in base:
        return base
    if base.endswith("\n"):
        return base + extra
    return base + "\n" + extra
