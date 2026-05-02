"""
core/executor.py
────────────────
AutomatonExecutor — runs one episode of the Adaptive Prompt Automaton.

Episode loop
────────────
  1. Start at the automaton's start_state.
  2. Render the current state's prompt template with task_input + carried context.
  3. Call the LLM API → response text + token count.
  4. Extract a FeatureVector from (task_input, response).
  5. Walk the transitions from the current state in priority order;
     fire the first one whose guard is satisfied.
  6. Repeat until: terminal state reached | budget exhausted | no firing transition.
  7. Return a fully-populated Episode object with path, steps, reward bookkeeping.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .automaton import Automaton, State
from .features import FeatureExtractor, FeatureVector


# ──────────────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ExecutionStep:
    """Record of one automaton step."""
    step:             int
    state_id:         str
    state_name:       str
    prompt:           str
    response:         str
    features:         Dict[str, float]
    transition_taken: Optional[str]     # "src→dst" or "TERMINAL" or None
    tokens_used:      int = 0


@dataclass
class Episode:
    """Full record of a single automaton episode."""
    episode_id:    str
    task_input:    str
    path:          List[str]           = field(default_factory=list)
    steps:         List[ExecutionStep] = field(default_factory=list)
    final_output:  str                 = ""
    total_tokens:  int                 = 0
    reward:        float               = 0.0
    success:       bool                = False
    # How the episode ended
    terminated_by: str                 = "budget"
    # "terminal_state" | "budget" | "no_transition" | "error"

    # ------------------------------------------------------------------
    def path_str(self) -> str:
        return " → ".join(self.path)

    def n_steps(self) -> int:
        return len(self.steps)

    def summary(self) -> Dict[str, Any]:
        return {
            "episode_id":    self.episode_id,
            "path":          self.path_str(),
            "n_steps":       self.n_steps(),
            "total_tokens":  self.total_tokens,
            "reward":        round(self.reward, 4),
            "terminated_by": self.terminated_by,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Executor
# ──────────────────────────────────────────────────────────────────────────────

class AutomatonExecutor:
    """
    Runs a single episode of a given Automaton against an LLM API.

    Parameters
    ----------
    automaton              : the Automaton FSA to execute
    llm_api                : object with .call(prompt, role, max_tokens) → (str, int)
    feature_extractor      : FeatureExtractor instance
    n_consistency_samples  : number of additional samples to draw per step for
                             self-consistency estimation (default 1 = disabled).
                             Each extra sample costs one LLM call against the budget.
    """

    def __init__(
        self,
        automaton:             Automaton,
        llm_api:               Any,
        feature_extractor:     FeatureExtractor,
        n_consistency_samples: int = 1,
    ):
        self.automaton              = automaton
        self.llm                    = llm_api
        self.extractor              = feature_extractor
        self.n_consistency_samples  = max(1, n_consistency_samples)

    # ------------------------------------------------------------------
    def run_episode(
        self,
        task_input:  str,
        episode_id:  str = "",
        verbose:     bool = False,
    ) -> Episode:
        """
        Execute one episode.

        Parameters
        ----------
        task_input  : the raw task / question string
        episode_id  : optional identifier for logging
        verbose     : if True, prints each step to stdout

        Returns
        -------
        Episode object with path, steps, final_output, token usage, etc.
        """
        if not episode_id:
            episode_id = str(uuid.uuid4())[:8]

        episode = Episode(episode_id=episode_id, task_input=task_input)

        current_state_id = self.automaton.start_state
        context          = ""
        budget_used      = 0
        max_budget       = self.automaton.config.max_budget
        max_steps        = self.automaton.config.max_steps

        for step_idx in range(max_steps):

            # ── Budget guard ──────────────────────────────────────────
            if budget_used >= max_budget:
                episode.terminated_by = "budget"
                break

            # ── Fetch current state ───────────────────────────────────
            state = self.automaton.get_state(current_state_id)
            if state is None:
                episode.terminated_by = "error"
                break

            episode.path.append(current_state_id)

            # ── Render prompt ─────────────────────────────────────────
            prompt = state.render(task_input, context)

            # ── Call LLM (primary response) ───────────────────────────
            response, tokens = self.llm.call(
                prompt, role=state.role, max_tokens=state.config.max_tokens
            )
            budget_used          += 1
            episode.total_tokens += tokens

            # ── Optional: draw extra samples for self-consistency ─────
            # Each sample costs one LLM call against the episode budget.
            consistency_samples: Optional[List[str]] = None
            if self.n_consistency_samples > 1:
                extra: List[str] = []
                for _ in range(self.n_consistency_samples - 1):
                    if budget_used >= max_budget:
                        break
                    s_resp, s_tok = self.llm.call(
                        prompt, role=state.role, max_tokens=state.config.max_tokens
                    )
                    budget_used          += 1
                    episode.total_tokens += s_tok
                    extra.append(s_resp)
                if extra:
                    consistency_samples = [response] + extra

            # ── Extract features ──────────────────────────────────────
            fvec: FeatureVector = self.extractor.extract(
                task_input  = task_input,
                llm_output  = response,
                samples     = consistency_samples,
                step        = step_idx,
            )

            # ── Build step record ─────────────────────────────────────
            exec_step = ExecutionStep(
                step             = step_idx,
                state_id         = current_state_id,
                state_name       = state.name,
                prompt           = prompt,
                response         = response,
                features         = fvec.to_dict(),
                transition_taken = None,
                tokens_used      = tokens,
            )

            # ── Update carry context ──────────────────────────────────
            if state.config.carry_context:
                context = response

            # ── Check terminal ────────────────────────────────────────
            if state.is_terminal:
                episode.final_output     = response
                episode.terminated_by    = "terminal_state"
                exec_step.transition_taken = "TERMINAL"
                episode.steps.append(exec_step)
                if verbose:
                    print(f"  [step {step_idx}] {state.name} → TERMINAL")
                break

            # ── Select transition ─────────────────────────────────────
            transitions = self.automaton.get_transitions_from(current_state_id)
            next_state_id: Optional[str] = None

            for t in transitions:
                if t.fires(fvec.features):
                    next_state_id               = t.target
                    exec_step.transition_taken   = f"{current_state_id}→{t.target}"
                    break

            episode.steps.append(exec_step)

            if verbose:
                arrow = exec_step.transition_taken or "NO_TRANSITION"
                print(
                    f"  [step {step_idx}] {state.name} | "
                    f"unc={fvec['uncertainty_score']:.2f} | "
                    f"conf={fvec['answer_confidence']:.2f} | → {arrow}"
                )

            if next_state_id is None:
                episode.terminated_by = "no_transition"
                if not episode.final_output:
                    episode.final_output = response
                break

            current_state_id = next_state_id

        # ── Ensure final_output is populated ─────────────────────────
        if not episode.final_output and episode.steps:
            episode.final_output = episode.steps[-1].response

        # ── Record path in automaton ──────────────────────────────────
        self.automaton.record_path(episode.path)

        return episode
