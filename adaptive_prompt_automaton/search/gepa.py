"""
search/gepa.py
──────────────
GEPA: Reflective Prompt Evolution
Based on: "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"
          arXiv 2507.19457 — Accepted at ICLR 2026 (Oral)

Core mechanism
──────────────
  1. Sample trajectories and collect Actionable Side Information (ASI):
       — feature signals: uncertainty, format, path taken, token cost
  2. Reflective LLM reads ASI from failures and produces a natural-language
     diagnosis + targeted fix proposal for the prompt.
  3. Apply the fix to the current best prompt template.
  4. Maintain a Pareto frontier across (reward ↑, token_cost ↓).
  5. Combine complementary fixes from the Pareto front into the next candidate.
  6. Repeat for N iterations.

Unlike evolutionary search (random mutations), GEPA uses *language-based
self-reflection* to produce semantically meaningful, targeted edits — which
is why it is far more sample-efficient than scalar-reward RL methods.

Key differences vs APA's EvolutionarySearch:
  • APA evolutionary: random word-swaps + threshold perturbation, no LLM feedback
  • GEPA: LLM-guided diagnosis → targeted fix → Pareto combination
  • Both operate on a single pool of prompts; neither branches at inference time.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from tqdm import tqdm
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from ..core.automaton import Automaton, AutomatonConfig, StateConfig
from ..core.executor import AutomatonExecutor, Episode
from ..core.features import FeatureExtractor


# ──────────────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ReflectionFeedback:
    """Structured output from one reflective LLM call."""
    iteration:         int
    failure_diagnosis: str
    proposed_fix:      str
    target_state:      str = "start"
    confidence:        float = 0.7


@dataclass
class ParetoCandidate:
    """
    One point on the Pareto frontier.
    Objectives: maximise avg_reward, minimise avg_tokens.
    """
    automaton:   Automaton
    avg_reward:  float
    avg_tokens:  float
    n_eval:      int = 0

    def dominates(self, other: "ParetoCandidate") -> bool:
        """Return True if self Pareto-dominates other."""
        return (
            self.avg_reward >= other.avg_reward
            and self.avg_tokens <= other.avg_tokens
            and (
                self.avg_reward > other.avg_reward
                or self.avg_tokens < other.avg_tokens
            )
        )


# ──────────────────────────────────────────────────────────────────────────────
# Reflection helpers
# ──────────────────────────────────────────────────────────────────────────────

# Patterns that map ASI signals to suggested fixes
_ASI_FIX_RULES: List[Tuple[str, str, str]] = [
    # (signal_name, condition, suggested_fix)
    ("uncertainty_score",     "high",   "Instruct the model to commit to a definitive answer and avoid hedging."),
    ("has_structured_format", "low",    "Ask for a structured response with numbered steps or clear sections."),
    ("output_length",         "low",    "Request a more detailed and thorough explanation."),
    ("output_length",         "high",   "Add a 'Be concise — answer in 2-3 sentences.' directive."),
    ("answer_confidence",     "low",    "Add: 'You are an expert. State your answer with confidence.'"),
    ("self_consistency",      "low",    "Prepend: 'Think carefully before answering. Verify your reasoning.'"),
    ("step_number",           "high",   "Consider decomposing: 'Break the problem into steps before answering.'"),
]

_REFLECTION_PROMPT_TEMPLATE = """\
You are an expert prompt engineer reviewing failed AI task completions.

FAILED EPISODES (with Actionable Side Information):
{asi_block}

DIAGNOSIS TASK:
Identify the single most important failure pattern across these episodes.
Propose ONE specific addition or rewording to fix the prompt.

Respond in this exact format:
DIAGNOSIS: <one-sentence description of the failure mode>
FIX: <exact text to append or replace in the prompt>"""


def _build_asi_block(episodes: List[Episode], reward_fn: Callable) -> str:
    lines = []
    for i, ep in enumerate(episodes[:4]):
        r    = reward_fn(ep)
        fvec = ep.steps[0].features if ep.steps else {}
        unc  = fvec.get("uncertainty_score", 0.0)
        conf = fvec.get("answer_confidence",  0.0)
        fmt  = fvec.get("has_structured_format", 0.0)
        lines.append(
            f"[{i+1}] reward={r:.2f}  uncertainty={unc:.2f}  "
            f"confidence={conf:.2f}  structured={fmt:.0f}  "
            f"path={ep.path_str()}  "
            f"response='{ep.final_output[:90]}'"
        )
    return "\n".join(lines)


def _parse_reflection(text: str) -> Tuple[str, str]:
    """Extract (diagnosis, fix) from the reflector's output."""
    diagnosis = "Output quality is poor; the prompt lacks sufficient specificity."
    fix       = "Be precise, direct, and avoid uncertain language."

    upper = text.upper()
    if "DIAGNOSIS:" in upper and "FIX:" in upper:
        try:
            d_start = upper.index("DIAGNOSIS:") + len("DIAGNOSIS:")
            f_start = upper.index("FIX:")      + len("FIX:")
            diagnosis = text[d_start:upper.index("FIX:")].strip().strip("- \n")
            fix       = text[f_start:].strip().strip("- \n").split("\n")[0]
        except ValueError:
            pass
    elif "step" in text.lower():
        fix = "Think step by step before responding."
    elif any(w in text.lower() for w in ("verify", "confirm", "check")):
        fix = "Verify your answer before stating it."
    elif "concise" in text.lower() or "brief" in text.lower():
        fix = "Be concise. Answer in 1-2 sentences."

    return diagnosis, fix


def _rule_based_fix(episodes: List[Episode], reward_fn: Callable) -> str:
    """
    Heuristic fix derived directly from ASI features — used when the
    reflector LLM output is not parseable.
    """
    agg: Dict[str, float] = {}
    for ep in episodes:
        fvec = ep.steps[0].features if ep.steps else {}
        for k, v in fvec.items():
            agg[k] = agg.get(k, 0.0) + v
    n = max(len(episodes), 1)
    avg = {k: v / n for k, v in agg.items()}

    if avg.get("uncertainty_score", 0) > 0.35:
        return "State your answer definitively. Avoid words like 'possibly', 'perhaps', or 'I'm not sure'."
    if avg.get("has_structured_format", 0) < 0.3:
        return "Structure your answer with numbered steps or clear headings."
    if avg.get("answer_confidence", 0) < 0.5:
        return "You are a domain expert. Answer with full confidence."
    return "Be thorough. Provide a complete, verified, well-reasoned answer."


# ──────────────────────────────────────────────────────────────────────────────
# GEPA Search
# ──────────────────────────────────────────────────────────────────────────────

class GEPASearch:
    """
    GEPA: Reflective Prompt Evolution.

    Parameters
    ----------
    initial_automaton    : seed Automaton (topology is kept; templates are evolved)
    llm_api              : LLM backend (used both as task LLM and reflector)
    feature_extractor    : FeatureExtractor
    reward_fn            : callable(Episode) → float
    n_iterations         : reflective evolution steps
    n_trajectory_samples : episodes sampled per iteration for ASI analysis
    failure_threshold    : episodes with reward < this are labelled "failures"
    n_eval_tasks         : tasks used to score each candidate
    seed                 : RNG seed
    """

    def __init__(
        self,
        initial_automaton:    Automaton,
        llm_api:              Any,
        feature_extractor:    FeatureExtractor,
        reward_fn:            Callable[[Episode], float],
        n_iterations:         int   = 10,
        n_trajectory_samples: int   = 6,
        failure_threshold:    float = 0.40,
        n_eval_tasks:         int   = 5,
        seed:                 int   = 42,
    ):
        self.initial_automaton  = initial_automaton
        self.llm                = llm_api
        self.extractor          = feature_extractor
        self.reward_fn          = reward_fn
        self.n_iterations       = n_iterations
        self.n_traj             = n_trajectory_samples
        self.fail_thresh        = failure_threshold
        self.n_eval             = n_eval_tasks
        self.rng                = random.Random(seed)

        self.pareto_frontier:  List[ParetoCandidate]   = []
        self.reflections:      List[ReflectionFeedback] = []
        self.history:          List[Dict[str, Any]]     = []
        self.best_automaton:   Optional[Automaton]       = None
        self.best_fitness:     float                     = -float("inf")

    # ------------------------------------------------------------------
    def _evaluate(self, aut: Automaton, tasks: List[str]) -> Tuple[float, float]:
        """Returns (avg_reward, avg_tokens)."""
        executor = AutomatonExecutor(aut, self.llm, self.extractor)
        sample   = self.rng.sample(tasks, min(self.n_eval, len(tasks)))
        rewards, tokens = [], []
        for i, t in enumerate(sample):
            ep = executor.run_episode(t, episode_id=f"gepa_eval_{i}")
            rewards.append(self.reward_fn(ep))
            tokens.append(ep.total_tokens)
        r = sum(rewards) / len(rewards) if rewards else 0.0
        t = sum(tokens)  / len(tokens)  if tokens  else 0.0
        return r, t

    # ------------------------------------------------------------------
    def _sample_trajectories(
        self, aut: Automaton, tasks: List[str], n: int
    ) -> List[Episode]:
        executor = AutomatonExecutor(aut, self.llm, self.extractor)
        sample   = self.rng.sample(tasks, min(n, len(tasks)))
        return [
            executor.run_episode(t, episode_id=f"gepa_traj_{i}")
            for i, t in enumerate(sample)
        ]

    # ------------------------------------------------------------------
    def _reflect(
        self, episodes: List[Episode], iteration: int
    ) -> ReflectionFeedback:
        """
        Build an ASI-enriched reflection prompt, call the LLM, parse the fix.
        Falls back to rule-based fix if parsing fails.
        """
        failures = sorted(
            [e for e in episodes if self.reward_fn(e) < self.fail_thresh],
            key=lambda e: self.reward_fn(e),
        )
        if not failures:
            failures = sorted(episodes, key=lambda e: self.reward_fn(e))[:3]

        asi_block = _build_asi_block(failures, self.reward_fn)
        prompt    = _REFLECTION_PROMPT_TEMPLATE.format(asi_block=asi_block)

        response, _ = self.llm.call(prompt, role="user", max_tokens=150)
        diagnosis, fix = _parse_reflection(response)

        # Fallback if fix is trivially short or generic
        if len(fix.split()) < 3:
            fix = _rule_based_fix(failures, self.reward_fn)

        return ReflectionFeedback(
            iteration         = iteration,
            failure_diagnosis = diagnosis,
            proposed_fix      = fix,
        )

    # ------------------------------------------------------------------
    def _apply_fix(self, aut: Automaton, fix: str, state_id: str = "start") -> Automaton:
        """Append the fix as an additional instruction to the target state."""
        child = aut.copy()
        if state_id in child.config.states:
            tmpl = child.config.states[state_id].template
            if fix.lower()[:30] not in tmpl.lower():
                child.config.states[state_id].template = (
                    tmpl.rstrip() + f"\n\nAdditional instruction: {fix}"
                )
        return child

    # ------------------------------------------------------------------
    def _update_pareto(self, cand: ParetoCandidate) -> None:
        self.pareto_frontier = [
            c for c in self.pareto_frontier if not cand.dominates(c)
        ]
        if not any(c.dominates(cand) for c in self.pareto_frontier):
            self.pareto_frontier.append(cand)

    # ------------------------------------------------------------------
    def _combine_pareto(self) -> Automaton:
        """
        Combine lessons from the Pareto front:
          — Take the template from the highest-reward member.
          — Take the transition thresholds from the most token-efficient member.
        """
        if not self.pareto_frontier:
            return self.initial_automaton.copy()

        best_r   = max(self.pareto_frontier, key=lambda c: c.avg_reward)
        best_eff = min(self.pareto_frontier, key=lambda c: c.avg_tokens)

        if best_r is best_eff:
            return best_r.automaton.copy()

        combined = best_r.automaton.copy()
        # Inherit transition thresholds from most-efficient member
        for i, t in enumerate(combined.config.transitions):
            if i < len(best_eff.automaton.config.transitions):
                t.threshold = best_eff.automaton.config.transitions[i].threshold
        return combined

    # ------------------------------------------------------------------
    def run(
        self,
        train_tasks: List[str],
        console:     Optional[Console] = None,
    ) -> Automaton:
        if console is None:
            console = Console()

        console.print(Panel(
            f"[bold yellow]GEPA: Reflective Prompt Evolution[/bold yellow]\n\n"
            f"  Iterations          : [green]{self.n_iterations}[/green]\n"
            f"  Trajectories / iter : [green]{self.n_traj}[/green]\n"
            f"  Failure threshold   : [green]{self.fail_thresh}[/green]\n"
            f"  Eval tasks / iter   : [green]{self.n_eval}[/green]\n"
            f"  Training tasks      : [green]{len(train_tasks)}[/green]\n\n"
            f"[dim]Ref: arXiv:2507.19457 — ICLR 2026 Oral[/dim]",
            title="[bold]GEPA Training[/bold]",
            border_style="yellow",
        ))

        current = self.initial_automaton.copy()

        # ── Initial evaluation ────────────────────────────────────────
        r0, t0 = self._evaluate(current, train_tasks)
        current.fitness = r0
        self._update_pareto(ParetoCandidate(current, r0, t0))
        if r0 > self.best_fitness:
            self.best_fitness   = r0
            self.best_automaton = current.copy()

        self.history.append(
            {"iteration": -1, "reward": r0, "tokens": t0,
             "pareto_size": 1, "diagnosis": "—", "fix": "—"}
        )

        # ── Reflective evolution loop ─────────────────────────────────
        iter_bar = tqdm(
            range(self.n_iterations),
            desc="  GEPA Iterations",
            colour="yellow",
        )

        for it in iter_bar:
            # 1. Sample trajectories (collect ASI)
            episodes = self._sample_trajectories(current, train_tasks, self.n_traj)

            # 2. Reflect on failures → proposed fix
            feedback = self._reflect(episodes, it)
            self.reflections.append(feedback)

            # 3. Apply fix to current template
            candidate = self._apply_fix(current, feedback.proposed_fix)

            # 4. Evaluate candidate
            new_r, new_t = self._evaluate(candidate, train_tasks)
            candidate.fitness = new_r

            # 5. Update Pareto frontier
            cand_obj = ParetoCandidate(candidate, new_r, new_t, n_eval=self.n_eval)
            self._update_pareto(cand_obj)

            # 6. Track global best
            if new_r > self.best_fitness:
                self.best_fitness   = new_r
                self.best_automaton = candidate.copy()

            # 7. Next candidate = Pareto combination
            current = self._combine_pareto()

            self.history.append({
                "iteration":   it,
                "reward":      new_r,
                "tokens":      new_t,
                "pareto_size": len(self.pareto_frontier),
                "diagnosis":   feedback.failure_diagnosis[:55],
                "fix":         feedback.proposed_fix[:55],
            })

            iter_bar.set_postfix({
                "reward":  f"{new_r:.3f}",
                "pareto":  len(self.pareto_frontier),
                "best":    f"{self.best_fitness:.3f}",
            })

            # Print table every 3 iterations
            if it % 3 == 2 or it == self.n_iterations - 1:
                self._print_iter_table(it, console)

        # ── Final summary ─────────────────────────────────────────────
        console.print(Panel(
            f"[bold green]GEPA Training Complete![/bold green]\n\n"
            f"  Best reward      : [cyan]{self.best_fitness:.4f}[/cyan]\n"
            f"  Pareto frontier  : [cyan]{len(self.pareto_frontier)} candidates[/cyan]\n"
            f"  Reflections made : [cyan]{len(self.reflections)}[/cyan]\n"
            f"  Total LLM calls  : [cyan]{self.llm.call_count}[/cyan]",
            title="[bold]GEPA Results[/bold]",
            border_style="yellow",
        ))

        return self.best_automaton or current

    # ------------------------------------------------------------------
    def _print_iter_table(self, it: int, console: Console) -> None:
        t = Table(
            title       = f"GEPA — Recent Iterations (up to iter {it})",
            box         = box.SIMPLE_HEAD,
            show_header = True,
        )
        t.add_column("Iter",    style="dim",    justify="center", width=6)
        t.add_column("Reward",  style="green",  justify="right",  width=8)
        t.add_column("Tokens",  style="cyan",   justify="right",  width=8)
        t.add_column("Pareto",               justify="center", width=7)
        t.add_column("Diagnosis (excerpt)", style="dim", max_width=45)

        for h in self.history[-5:]:
            t.add_row(
                str(h["iteration"]),
                f"{h['reward']:.4f}",
                f"{h['tokens']:.0f}",
                str(h["pareto_size"]),
                h["diagnosis"][:43],
            )
        console.print(t)
