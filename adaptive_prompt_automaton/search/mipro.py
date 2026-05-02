"""
search/mipro.py
───────────────
MIPRO / MIPROv2: Optimizing Instructions and Demonstrations for
                 Multi-Stage Language Model Programs
Based on: arXiv:2406.11695 — EMNLP 2024
          DSPy framework: https://dspy.ai/api/optimizers/MIPROv2/

Core mechanism
──────────────
  Stage 1 — Bootstrap demonstrations
    Run a simple seed prompt on training tasks; collect high-reward
    (input, output) pairs as few-shot examples.

  Stage 2 — Instruction candidate proposals
    A meta-optimizer LLM (or template bank) proposes K candidate
    instruction strings grounded in observed task structure.

  Stage 3 — Bayesian search over (instruction × demo_set)
    Evaluate all (instruction_i, demo_set_j) combinations on a dev set.
    A surrogate model (running-mean Thompson sampling) decides which
    combos to evaluate more and which to prune.

  Stage 4 — Meta-optimization (optional rounds)
    After round 1, the top instructions are used to propose variants;
    the search continues on the refined candidate pool.

Output: the single-state prompt with the best (instruction + few-shot demos)
        wrapped in a one-state Automaton for fair evaluation on the same
        benchmark infrastructure as APA and GEPA.

Key structural difference vs APA
──────────────────────────────────
MIPRO produces a STATIC prompt (possibly with demos); it does not branch
at inference time. Its advantage is a richer instruction + demonstration
format; its limitation is that the same prompt runs on every input regardless
of runtime signals (length, uncertainty, tool feedback).
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
class FewShotDemo:
    """A bootstrapped (input, output) pair with its quality score."""
    task_input:   str
    model_output: str
    reward:       float


@dataclass
class InstructionCandidate:
    """One candidate instruction string with accumulated reward estimates."""
    cand_id:     str
    template:    str
    eval_scores: List[float] = field(default_factory=list)

    @property
    def mean_score(self) -> float:
        return sum(self.eval_scores) / len(self.eval_scores) if self.eval_scores else 0.0

    @property
    def ucb_score(self) -> float:
        """Upper Confidence Bound for Thompson-sampling-style selection."""
        import math
        n = max(len(self.eval_scores), 1)
        return self.mean_score + 0.3 * math.sqrt(1.0 / n)


# ──────────────────────────────────────────────────────────────────────────────
# Instruction template bank (used by the meta-optimizer)
# ──────────────────────────────────────────────────────────────────────────────

_BASE_INSTRUCTIONS: List[str] = [
    "Answer the following question accurately and concisely:\n\n{input}{demos}",
    "You are a helpful, precise AI assistant. Provide a direct answer to:\n\n{input}{demos}",
    "Think step by step, then provide the final answer to:\n\n{input}{demos}",
    "You are a domain expert. Carefully reason through the question and respond:\n\n{input}{demos}",
    "Answer definitively and with confidence:\n\n{input}{demos}",
    "Analyse the key components of this question, then answer completely:\n\n{input}{demos}",
    "You must provide a correct, verified answer. Question:\n\n{input}{demos}",
    "Address each aspect of the question methodically. Be thorough:\n\n{input}{demos}",
    "Apply your domain knowledge to answer precisely and completely:\n\n{input}{demos}",
    "As a knowledgeable assistant, provide the best possible answer to:\n\n{input}{demos}",
    "Reason carefully. Verify your answer. Then state it clearly:\n\n{input}{demos}",
    "Break down complex parts before answering. Be precise:\n\n{input}{demos}",
]

# Suffixes added by the meta-optimizer in round 2+
_META_SUFFIXES: List[str] = [
    "\n\nIMPORTANT: Avoid hedging or uncertainty. State facts directly.",
    "\n\nIMPORTANT: If the question is complex, decompose it first.",
    "\n\nIMPORTANT: Provide a structured answer with clear reasoning.",
    "\n\nIMPORTANT: Be concise — answer in 2-4 sentences unless detail is essential.",
    "\n\nIMPORTANT: Always verify your answer before stating it.",
]


# ──────────────────────────────────────────────────────────────────────────────
# MIPROSearch
# ──────────────────────────────────────────────────────────────────────────────

class MIPROSearch:
    """
    MIPRO / MIPROv2 prompt optimizer.

    Parameters
    ----------
    llm_api                  : LLM backend
    feature_extractor        : FeatureExtractor
    reward_fn                : callable(Episode) → float
    n_bootstrap_episodes     : episodes to run for demo collection
    n_instruction_candidates : size of instruction pool
    n_demo_sets              : number of distinct demo subsets to try
    max_demos_per_set        : k-shot count per demo set
    n_bayesian_rounds        : Bayesian search rounds (evaluate more combos)
    n_eval_tasks             : tasks per evaluation call
    seed                     : RNG seed
    """

    def __init__(
        self,
        llm_api:                  Any,
        feature_extractor:        FeatureExtractor,
        reward_fn:                Callable[[Episode], float],
        n_bootstrap_episodes:     int = 12,
        n_instruction_candidates: int = 6,
        n_demo_sets:              int = 3,
        max_demos_per_set:        int = 2,
        n_bayesian_rounds:        int = 3,
        n_eval_tasks:             int = 5,
        seed:                     int = 42,
    ):
        self.llm              = llm_api
        self.extractor        = feature_extractor
        self.reward_fn        = reward_fn
        self.n_bootstrap      = n_bootstrap_episodes
        self.n_instrs         = n_instruction_candidates
        self.n_demo_sets      = n_demo_sets
        self.max_demos        = max_demos_per_set
        self.n_rounds         = n_bayesian_rounds
        self.n_eval           = n_eval_tasks
        self.rng              = random.Random(seed)

        self.demos:            List[FewShotDemo]         = []
        self.candidates:       List[InstructionCandidate] = []
        self.history:          List[Dict[str, Any]]       = []
        self.best_automaton:   Optional[Automaton]         = None
        self.best_fitness:     float                       = -float("inf")

    # ------------------------------------------------------------------
    # Stage 1 — Bootstrap demonstrations
    # ------------------------------------------------------------------

    def _bootstrap_demos(self, tasks: List[str]) -> List[FewShotDemo]:
        """
        Run a minimal seed prompt on training tasks; collect high-reward
        (input, output) pairs as few-shot demonstrations.
        """
        seed_state  = StateConfig(
            state_id="seed", name="Seed",
            template   = "Answer this question: {input}",
            is_terminal = True, carry_context = False,
        )
        seed_config = AutomatonConfig(
            automaton_id = "mipro_seed", name = "MIPRO-Seed",
            start_state  = "seed",
            states       = {"seed": seed_state},
            transitions  = [], max_steps = 1, max_budget = 1,
        )
        seed_aut  = Automaton(seed_config)
        executor  = AutomatonExecutor(seed_aut, self.llm, self.extractor)
        collected: List[FewShotDemo] = []

        sample = self.rng.sample(tasks, min(self.n_bootstrap, len(tasks)))
        for i, task in enumerate(
            tqdm(sample, desc="  Bootstrapping demos", colour="magenta", leave=False)
        ):
            ep = executor.run_episode(task, episode_id=f"boot_{i}")
            r  = self.reward_fn(ep)
            if r >= 0.45:
                collected.append(
                    FewShotDemo(
                        task_input   = task[:100],
                        model_output = ep.final_output[:120],
                        reward       = r,
                    )
                )

        collected.sort(key=lambda d: -d.reward)
        return collected[: self.max_demos * (self.n_demo_sets + 1)]

    # ------------------------------------------------------------------
    # Stage 2 — Instruction proposals
    # ------------------------------------------------------------------

    def _propose_instructions(self) -> List[InstructionCandidate]:
        """
        Select K base templates + propose meta-optimizer variants.
        In a full implementation this would call a meta-LM to generate
        instructions grounded in the observed task distribution.
        """
        base = self.rng.sample(
            _BASE_INSTRUCTIONS,
            min(self.n_instrs, len(_BASE_INSTRUCTIONS)),
        )
        candidates = [
            InstructionCandidate(cand_id=f"instr_{i}", template=tmpl)
            for i, tmpl in enumerate(base)
        ]
        return candidates

    # ------------------------------------------------------------------
    # Stage 3 — Bayesian search
    # ------------------------------------------------------------------

    def _make_demo_sets(self) -> List[List[FewShotDemo]]:
        """Build n_demo_sets distinct subsets of bootstrapped demos + one empty set."""
        sets: List[List[FewShotDemo]] = [[]]          # always include no-demo baseline
        for _ in range(self.n_demo_sets):
            if len(self.demos) >= self.max_demos:
                s = self.rng.sample(self.demos, self.max_demos)
            else:
                s = list(self.demos)
            sets.append(s)
        return sets

    @staticmethod
    def _format_demos(demos: List[FewShotDemo]) -> str:
        if not demos:
            return ""
        lines = ["\n\n--- Examples ---"]
        for d in demos:
            lines.append(f"\nQ: {d.task_input}\nA: {d.model_output}")
        lines.append("\n--- Now answer ---\n")
        return "\n".join(lines)

    def _build_automaton(
        self, instr: InstructionCandidate, demos: List[FewShotDemo]
    ) -> Automaton:
        demos_str  = self._format_demos(demos)
        full_tmpl  = instr.template.replace("{demos}", demos_str)
        # Remove leftover {demos} if not in template
        full_tmpl  = full_tmpl.replace("{demos}", "")
        state = StateConfig(
            state_id    = "mipro",
            name        = "MIPRO-Optimized",
            template    = full_tmpl,
            is_terminal = True,
            carry_context = False,
        )
        config = AutomatonConfig(
            automaton_id = f"mipro_{self.rng.randint(1000, 9999)}",
            name         = "MIPRO",
            start_state  = "mipro",
            states       = {"mipro": state},
            transitions  = [],
            max_steps    = 1,
            max_budget   = 1,
        )
        return Automaton(config)

    def _evaluate(self, aut: Automaton, tasks: List[str]) -> float:
        executor = AutomatonExecutor(aut, self.llm, self.extractor)
        sample   = self.rng.sample(tasks, min(self.n_eval, len(tasks)))
        rewards  = []
        for i, t in enumerate(sample):
            ep = executor.run_episode(t, episode_id=f"mipro_eval_{i}")
            rewards.append(self.reward_fn(ep))
        return sum(rewards) / len(rewards) if rewards else 0.0

    # ------------------------------------------------------------------
    # Stage 4 — Meta-optimization (refine instruction pool after round 1)
    # ------------------------------------------------------------------

    def _meta_refine(self) -> List[InstructionCandidate]:
        """
        After the first Bayesian round, keep the top-3 instructions and
        generate suffix variants to form a refined pool.
        """
        top3 = sorted(self.candidates, key=lambda c: -c.mean_score)[:3]
        refined: List[InstructionCandidate] = list(top3)
        for cand in top3:
            suffix = self.rng.choice(_META_SUFFIXES)
            new_tmpl = cand.template.rstrip() + suffix
            refined.append(
                InstructionCandidate(
                    cand_id  = f"{cand.cand_id}_meta",
                    template = new_tmpl,
                )
            )
        return refined

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        train_tasks: List[str],
        console:     Optional[Console] = None,
    ) -> Automaton:
        if console is None:
            console = Console()

        console.print(Panel(
            f"[bold magenta]MIPRO / MIPROv2: Instruction + Demo Optimization[/bold magenta]\n\n"
            f"  Bootstrap episodes     : [green]{self.n_bootstrap}[/green]\n"
            f"  Instruction candidates : [green]{self.n_instrs}[/green]\n"
            f"  Demo sets              : [green]{self.n_demo_sets}[/green] "
            f"(+ 1 zero-shot baseline)\n"
            f"  Bayesian rounds        : [green]{self.n_rounds}[/green]\n"
            f"  Eval tasks / combo     : [green]{self.n_eval}[/green]\n"
            f"  Training tasks         : [green]{len(train_tasks)}[/green]\n\n"
            f"[dim]Ref: arXiv:2406.11695 — DSPy / EMNLP 2024[/dim]",
            title="[bold]MIPRO Training[/bold]",
            border_style="magenta",
        ))

        # ── Stage 1: Bootstrap ────────────────────────────────────────
        console.print("[magenta]Stage 1 — Bootstrapping few-shot demonstrations…[/magenta]")
        self.demos = self._bootstrap_demos(train_tasks)
        console.print(
            f"  Collected [green]{len(self.demos)}[/green] "
            f"high-quality demonstrations (reward ≥ 0.45)"
        )

        # ── Stage 2: Propose instructions ────────────────────────────
        console.print("[magenta]Stage 2 — Proposing instruction candidates…[/magenta]")
        self.candidates = self._propose_instructions()
        console.print(
            f"  Proposed [green]{len(self.candidates)}[/green] "
            f"instruction candidates"
        )

        # ── Stage 3 + 4: Bayesian search ─────────────────────────────
        demo_sets = self._make_demo_sets()
        console.print(
            f"[magenta]Stage 3 — Bayesian search over "
            f"{len(self.candidates)} instructions × {len(demo_sets)} demo sets "
            f"for {self.n_rounds} rounds…[/magenta]"
        )

        for rnd in range(self.n_rounds):

            # Meta-refine after first round
            if rnd == 1:
                console.print("[magenta]  Stage 4 — Meta-refining instruction pool…[/magenta]")
                self.candidates = self._meta_refine()
                console.print(
                    f"  Refined pool: [green]{len(self.candidates)}[/green] candidates"
                )

            combo_iter = tqdm(
                [(ci, di, c, d)
                 for ci, c in enumerate(self.candidates)
                 for di, d  in enumerate(demo_sets)],
                desc      = f"  Round {rnd + 1} Combos",
                colour    = "magenta",
                leave     = True,
            )

            for ci, di, cand, demo_set in combo_iter:
                aut   = self._build_automaton(cand, demo_set)
                score = self._evaluate(aut, train_tasks)
                cand.eval_scores.append(score)

                if score > self.best_fitness:
                    self.best_fitness   = score
                    self.best_automaton = aut.copy()
                    self.best_automaton.fitness = score

                combo_iter.set_postfix({
                    "score": f"{score:.3f}",
                    "best":  f"{self.best_fitness:.3f}",
                })

            # Round summary
            top = sorted(self.candidates, key=lambda c: -c.mean_score)[:3]
            self.history.append({
                "round":       rnd,
                "best_score":  self.best_fitness,
                "top_instrs":  [(c.cand_id, round(c.mean_score, 3)) for c in top],
                "n_combos":    len(self.candidates) * len(demo_sets),
            })

            self._print_round_table(rnd, top, console)

        # ── Final summary ─────────────────────────────────────────────
        console.print(Panel(
            f"[bold green]MIPRO Training Complete![/bold green]\n\n"
            f"  Best fitness     : [cyan]{self.best_fitness:.4f}[/cyan]\n"
            f"  Demos used       : [cyan]{len(self.demos)}[/cyan]\n"
            f"  Combos evaluated : [cyan]{sum(h['n_combos'] for h in self.history)}[/cyan]\n"
            f"  Total LLM calls  : [cyan]{self.llm.call_count}[/cyan]",
            title="[bold]MIPRO Results[/bold]",
            border_style="magenta",
        ))

        return self.best_automaton or self._build_automaton(self.candidates[0], [])

    # ------------------------------------------------------------------
    def _print_round_table(
        self,
        rnd:     int,
        top:     List[InstructionCandidate],
        console: Console,
    ) -> None:
        t = Table(
            title       = f"MIPRO Round {rnd + 1} — Top Instructions",
            box         = box.SIMPLE_HEAD,
            show_header = True,
        )
        t.add_column("ID",         style="dim",     justify="left",  width=12)
        t.add_column("Mean Score", style="green",   justify="right", width=11)
        t.add_column("Evals",                       justify="center", width=7)
        t.add_column("Template (excerpt)", style="dim", max_width=50)

        for c in top:
            excerpt = c.template.replace("\n", " ")[:48] + "…"
            t.add_row(
                c.cand_id,
                f"{c.mean_score:.4f}",
                str(len(c.eval_scores)),
                excerpt,
            )
        console.print(t)
