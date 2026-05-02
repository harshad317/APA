"""
core/automaton.py
─────────────────
Defines the core Adaptive Prompt Automaton data structures:
  - StateConfig / State          : a node that holds a prompt template
  - TransitionConfig / Transition: a directed edge with a guard predicate
  - AutomatonConfig / Automaton  : the full FSA object with fitness tracking
"""
from __future__ import annotations

import uuid
import copy
from typing import Dict, List, Optional, Any, Tuple

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic config models  (serialisable / mutable)
# ──────────────────────────────────────────────────────────────────────────────

class StateConfig(BaseModel):
    state_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str
    template: str
    role: str = "user"          # "system" | "user" | "assistant"
    max_tokens: int = 256
    is_terminal: bool = False
    carry_context: bool = True  # carry previous LLM output as {context}
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"


class TransitionConfig(BaseModel):
    transition_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    source_state: str
    target_state: str
    guard_type: str = "threshold"   # "threshold" | "always"
    feature_name: str = "uncertainty_score"
    threshold: float = 0.5
    operator: str = ">"             # ">" | "<" | ">=" | "<=" | "==" | "always"
    priority: int = 0               # higher fires first among candidates

    class Config:
        extra = "allow"


class AutomatonConfig(BaseModel):
    automaton_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = "PromptAutomaton"
    start_state: str
    states: Dict[str, StateConfig]
    transitions: List[TransitionConfig] = Field(default_factory=list)
    max_steps: int = 6
    max_budget: int = 8             # max LLM calls per episode

    class Config:
        extra = "allow"


# ──────────────────────────────────────────────────────────────────────────────
# Runtime classes  (wraps configs with behaviour)
# ──────────────────────────────────────────────────────────────────────────────

class State:
    """A node in the prompt automaton FSA."""

    def __init__(self, config: StateConfig):
        self.config = config
        self.state_id = config.state_id
        self.name = config.name
        self.template = config.template
        self.is_terminal = config.is_terminal
        self.role = config.role

    # ------------------------------------------------------------------
    def render(self, task_input: str, context: str = "") -> str:
        """Fill the prompt template with runtime values."""
        try:
            return self.template.format(input=task_input, context=context)
        except (KeyError, IndexError):
            # Fallback: plain replacement
            return (
                self.template
                .replace("{input}", task_input)
                .replace("{context}", context)
            )

    def __repr__(self) -> str:
        return (
            f"State(id={self.state_id!r}, name={self.name!r}, "
            f"terminal={self.is_terminal})"
        )


class Transition:
    """A directed edge in the FSA with a guard predicate over a FeatureVector."""

    _OPS = {
        ">":  lambda v, t: v > t,
        "<":  lambda v, t: v < t,
        ">=": lambda v, t: v >= t,
        "<=": lambda v, t: v <= t,
        "==": lambda v, t: abs(v - t) < 1e-6,
    }

    def __init__(self, config: TransitionConfig):
        self.config = config
        self.source = config.source_state
        self.target = config.target_state
        self.priority = config.priority

    # ------------------------------------------------------------------
    def fires(self, features: Dict[str, float]) -> bool:
        """Return True if this transition's guard is satisfied."""
        if self.config.guard_type == "always" or self.config.operator == "always":
            return True
        val = features.get(self.config.feature_name, 0.0)
        op_fn = self._OPS.get(self.config.operator)
        if op_fn is None:
            return False
        return op_fn(val, self.config.threshold)

    def __repr__(self) -> str:
        return (
            f"Transition({self.source!r}→{self.target!r}, "
            f"{self.config.feature_name}{self.config.operator}"
            f"{self.config.threshold:.2f}, pri={self.priority})"
        )


class Automaton:
    """
    The Adaptive Prompt Automaton FSA.

    Holds a dictionary of States and a list of Transitions.
    Tracks per-episode statistics: path_counts, episodes_run, fitness.
    """

    def __init__(self, config: AutomatonConfig):
        self.config = config
        self.automaton_id = config.automaton_id
        self.name = config.name
        self.start_state = config.start_state

        self.states: Dict[str, State] = {
            sid: State(sc) for sid, sc in config.states.items()
        }
        self.transitions: List[Transition] = [
            Transition(tc) for tc in config.transitions
        ]

        # Fitness and diagnostics
        self.fitness: float = 0.0
        self.episodes_run: int = 0
        self.path_counts: Dict[Tuple[str, ...], int] = {}
        self.reward_history: List[float] = []

        # Behavioral fingerprint — per-probe-task score vector (Fix 5).
        # Populated by EvolutionarySearch._evaluate() when probe_tasks active.
        self.fingerprint: List[float] = []

        # Diversity selection scratch space (set by _diversity_aware_select).
        self._diversity_bonus:    float = 0.0
        self._augmented_fitness:  float = 0.0

    # ------------------------------------------------------------------
    def get_transitions_from(self, state_id: str) -> List[Transition]:
        """Return all transitions leaving state_id, sorted by priority desc."""
        candidates = [t for t in self.transitions if t.source == state_id]
        return sorted(candidates, key=lambda t: -t.priority)

    def get_state(self, state_id: str) -> Optional[State]:
        return self.states.get(state_id)

    # ------------------------------------------------------------------
    def record_path(self, path: List[str]) -> None:
        key = tuple(path)
        self.path_counts[key] = self.path_counts.get(key, 0) + 1
        self.episodes_run += 1

    def state_visit_entropy(self) -> float:
        """Shannon entropy of path distribution (higher = more diverse routing)."""
        import math
        total = sum(self.path_counts.values())
        if total == 0:
            return 0.0
        h = 0.0
        for cnt in self.path_counts.values():
            p = cnt / total
            h -= p * math.log2(p + 1e-12)
        return max(0.0, h)

    # ------------------------------------------------------------------
    def copy(self, copy_diagnostics: bool = False) -> "Automaton":
        """Deep-copy this automaton with a fresh ID."""
        new_config = self.config.copy(deep=True)
        new_config.automaton_id = str(uuid.uuid4())[:8]
        new_config.name = self.name + "_copy"
        child = Automaton(new_config)
        child.fitness     = self.fitness
        child.fingerprint = list(self.fingerprint)      # carry fingerprint across copy
        if copy_diagnostics:
            child.episodes_run    = self.episodes_run
            child.path_counts     = dict(self.path_counts)
            child.reward_history  = list(self.reward_history)
        return child

    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        return {
            "id": self.automaton_id,
            "name": self.name,
            "n_states": len(self.states),
            "n_transitions": len(self.transitions),
            "start_state": self.start_state,
            "fitness": round(self.fitness, 4),
            "episodes_run": self.episodes_run,
            "path_entropy": round(self.state_visit_entropy(), 4),
        }

    def __repr__(self) -> str:
        return (
            f"Automaton(id={self.automaton_id!r}, name={self.name!r}, "
            f"states={len(self.states)}, fitness={self.fitness:.3f})"
        )
