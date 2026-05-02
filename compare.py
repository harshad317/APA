#!/usr/bin/env python3
"""
compare.py
──────────
Three-way benchmark comparison:

  ┌──────────────────────────────────────────────────────────────────────┐
  │  Method │  Approach                         │  Key paper             │
  ├──────────┼───────────────────────────────────┼────────────────────────┤
  │  APA    │  Stateful FSA + evolutionary       │  This work (NeurIPS)   │
  │  GEPA   │  Reflective prompt evolution       │  arXiv:2507.19457      │
  │  MIPRO  │  Instruction + demo optimisation   │  arXiv:2406.11695      │
  └──────────┴───────────────────────────────────┴────────────────────────┘

Benchmarks (same tasks for all three methods)
──────────────────────────────────────────────
  1.  In-distribution QA      (easy / medium / hard)
  2.  Distribution shift      (clean train → longer/noisier test)
  3.  Perturbation stability  (paraphrase / noise augmented inputs)

Metrics
───────
  • Average composite reward
  • Robustness delta  (in-dist reward − shifted reward)
  • Sample efficiency (LLM calls during training)
  • Inference-time branching (path entropy — APA only)
  • Wins per scenario

Rich tables + tqdm progress bars for all output.
"""
from __future__ import annotations

import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from rich import box
from tqdm import tqdm

from adaptive_prompt_automaton.core.automaton import (
    Automaton, AutomatonConfig, StateConfig, TransitionConfig,
)
from adaptive_prompt_automaton.core.features import FeatureExtractor
from adaptive_prompt_automaton.core.executor import AutomatonExecutor, Episode
from adaptive_prompt_automaton.search.evolution import EvolutionarySearch

# Auto-detect DSPy: use official dspy.GEPA when available,
# fall back to the hand-reimplementation for environments without dspy.
try:
    from adaptive_prompt_automaton.search.gepa_dspy import GEPADSPySearch as _GEPABackend
    _GEPA_BACKEND_LABEL = "GEPA (DSPy official)"
    _GEPA_USES_DSPY     = True
except ImportError:
    from adaptive_prompt_automaton.search.gepa import GEPASearch as _GEPABackend  # type: ignore[assignment]
    _GEPA_BACKEND_LABEL = "GEPA (hand-reimpl)"
    _GEPA_USES_DSPY     = False

# Auto-detect DSPy: use official MIPROv2 wrapper when available,
# fall back to the hand-reimplementation for environments without dspy.
try:
    from adaptive_prompt_automaton.search.mipro_dspy import MIPRODSPySearch as _MIPROBackend
    _MIPRO_BACKEND_LABEL = "MIPRO (DSPy MIPROv2)"
    _MIPRO_USES_DSPY     = True
except ImportError:
    from adaptive_prompt_automaton.search.mipro import MIPROSearch as _MIPROBackend  # type: ignore[assignment]
    _MIPRO_BACKEND_LABEL = "MIPRO (hand-reimpl)"
    _MIPRO_USES_DSPY     = False
from adaptive_prompt_automaton.eval.benchmarks import (
    make_qa_benchmark,
    make_distribution_shift_benchmark,
    make_perturbation_benchmark,
    composite_reward,
    Task,
)
from adaptive_prompt_automaton.utils.api import get_llm_api

console = Console()
SEED = 42
random.seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# Automaton factories (same for APA and GEPA seeds; MIPRO ignores topology)
# ══════════════════════════════════════════════════════════════════════════════

def build_apa_automaton() -> Automaton:
    """4-state APA used as seed for both APA (evolutionary) and GEPA."""
    states = {
        "start": StateConfig(
            state_id="start", name="Start",
            template=(
                "You are a precise and knowledgeable AI assistant.\n"
                "Answer the following question clearly and directly:\n\n"
                "{input}\n\nProvide a concise, accurate answer."
            ),
            role="user", max_tokens=256, is_terminal=False, carry_context=True,
        ),
        "decompose": StateConfig(
            state_id="decompose", name="Decompose",
            template=(
                "This is a complex, multi-part question. Break it down step by step.\n\n"
                "Question: {input}\n\nPrevious attempt: {context}\n\n"
                "Decompose and solve each sub-task, then synthesise the final answer."
            ),
            role="user", max_tokens=512, is_terminal=False, carry_context=True,
        ),
        "verify": StateConfig(
            state_id="verify", name="Verify",
            template=(
                "Verify the following answer and correct any errors.\n\n"
                "Original question: {input}\nPrevious answer: {context}\n\n"
                "Is this correct and complete? Provide the verified answer."
            ),
            role="user", max_tokens=256, is_terminal=False, carry_context=True,
        ),
        "terminal": StateConfig(
            state_id="terminal", name="Terminal",
            template=(
                "State the final, definitive answer to:\n\n{input}\n\n"
                "Based on the analysis: {context}\n\nAnswer concisely and confidently."
            ),
            role="user", max_tokens=128, is_terminal=True, carry_context=False,
        ),
    }
    transitions = [
        TransitionConfig(source_state="start",     target_state="decompose",
                         guard_type="threshold",   feature_name="is_long_input",
                         threshold=0.5, operator=">",      priority=3),
        TransitionConfig(source_state="start",     target_state="verify",
                         guard_type="threshold",   feature_name="uncertainty_score",
                         threshold=0.30, operator=">",     priority=2),
        TransitionConfig(source_state="start",     target_state="terminal",
                         guard_type="threshold",   feature_name="answer_confidence",
                         threshold=0.70, operator=">=",    priority=1),
        TransitionConfig(source_state="decompose", target_state="verify",
                         guard_type="threshold",   feature_name="uncertainty_score",
                         threshold=0.35, operator=">",     priority=2),
        TransitionConfig(source_state="decompose", target_state="terminal",
                         guard_type="always",      feature_name="input_length",
                         threshold=0.0, operator="always", priority=1),
        TransitionConfig(source_state="verify",    target_state="terminal",
                         guard_type="always",      feature_name="input_length",
                         threshold=0.0, operator="always", priority=1),
    ]
    return Automaton(AutomatonConfig(
        automaton_id="apa_seed", name="APA-seed",
        start_state="start", states=states,
        transitions=transitions, max_steps=6, max_budget=8,
    ))


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ══════════════════════════════════════════════════════════════════════════════

def eval_on_tasks(
    automaton: Automaton,
    llm:       object,
    extractor: FeatureExtractor,
    tasks:     List[Task],
    desc:      str   = "Evaluating",
    colour:    str   = "cyan",
) -> List[Episode]:
    executor = AutomatonExecutor(automaton, llm, extractor)
    episodes: List[Episode] = []
    for task in tqdm(tasks, desc=f"  {desc}", colour=colour, leave=True):
        ep        = executor.run_episode(task.input_text, episode_id=task.task_id)
        ep.reward = composite_reward(ep)
        episodes.append(ep)
    return episodes


def avg(episodes: List[Episode]) -> float:
    r = [composite_reward(e) for e in episodes]
    return sum(r) / len(r) if r else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Rich rendering helpers
# ══════════════════════════════════════════════════════════════════════════════

def render_scenario_table(
    scenario_name:  str,
    results:        Dict[str, List[Episode]],
    task_ids:       List[str],
) -> None:
    """Per-task reward table for one benchmark scenario."""
    methods = list(results.keys())
    t = Table(title=scenario_name, box=box.DOUBLE_EDGE, highlight=True)
    t.add_column("Task", style="bold", max_width=20)
    for m in methods:
        colour = {"APA": "cyan", "GEPA": "yellow", "MIPRO": "magenta"}.get(m, "white")
        t.add_column(m, justify="right", style=colour)
    t.add_column("Winner", justify="center")

    for i, tid in enumerate(task_ids):
        row_vals = {m: composite_reward(results[m][i]) for m in methods}
        winner   = max(row_vals, key=row_vals.__getitem__)
        w_style  = {"APA": "cyan", "GEPA": "yellow", "MIPRO": "magenta"}.get(winner, "white")
        t.add_row(
            tid[:19],
            *[f"{row_vals[m]:.3f}" for m in methods],
            f"[bold {w_style}]{winner}[/bold {w_style}]",
        )

    # Average row
    avgs = {m: avg(results[m]) for m in methods}
    best_avg = max(avgs, key=avgs.__getitem__)
    w_style  = {"APA": "cyan", "GEPA": "yellow", "MIPRO": "magenta"}.get(best_avg, "white")
    t.add_row(
        "[bold]AVG[/bold]",
        *[f"[bold]{avgs[m]:.3f}[/bold]" for m in methods],
        f"[bold {w_style}]{best_avg}[/bold {w_style}]",
    )
    console.print(t)
    console.print()


def render_summary_table(
    in_dist:   Dict[str, List[Episode]],
    shifted:   Dict[str, List[Episode]],
    perturbed: Dict[str, List[Episode]],
    train_calls: Dict[str, int],
    apa_automaton: Automaton,
) -> None:
    """Master summary: reward, robustness, efficiency, branching."""
    console.print(Rule("[bold]Master Comparison Summary[/bold]"))

    t = Table(
        title       = "APA vs GEPA vs MIPRO — All Scenarios",
        box         = box.DOUBLE_EDGE,
        highlight   = True,
        show_lines  = True,
    )

    col_styles = {"APA": "cyan", "GEPA": "yellow", "MIPRO": "magenta"}
    t.add_column("Metric", style="bold", min_width=30)
    for m, sty in col_styles.items():
        t.add_column(m, justify="right", style=sty, min_width=10)
    t.add_column("Best", justify="center", min_width=8)

    def add_metric(label: str, values: Dict[str, float], higher_is_better: bool = True):
        best = max(values, key=values.__getitem__) if higher_is_better \
               else min(values, key=values.__getitem__)
        w_sty = col_styles.get(best, "white")
        t.add_row(
            label,
            *[f"{values[m]:.4f}" for m in col_styles],
            f"[bold {w_sty}]{best}[/bold {w_sty}]",
        )

    def add_int_metric(label: str, values: Dict[str, int], higher_is_better: bool = False):
        best = min(values, key=values.__getitem__) if not higher_is_better \
               else max(values, key=values.__getitem__)
        w_sty = col_styles.get(best, "white")
        t.add_row(
            label,
            *[str(values[m]) for m in col_styles],
            f"[bold {w_sty}]{best}[/bold {w_sty}]",
        )

    # Reward metrics
    in_avgs   = {m: avg(in_dist[m])   for m in col_styles}
    sh_avgs   = {m: avg(shifted[m])   for m in col_styles}
    pe_avgs   = {m: avg(perturbed[m]) for m in col_styles}
    rob_delta = {m: in_avgs[m] - sh_avgs[m] for m in col_styles}   # smaller = more robust

    add_metric("In-Distribution Avg Reward",     in_avgs)
    add_metric("Distribution Shift Avg Reward",  sh_avgs)
    add_metric("Perturbation Stability Reward",  pe_avgs)

    t.add_row("", "", "", "", "")  # separator

    add_metric(
        "Robustness Δ (in-dist − shifted) ↓",
        rob_delta,
        higher_is_better=False,   # smaller drop = better
    )

    t.add_row("", "", "", "", "")  # separator

    add_int_metric(
        "Training LLM calls ↓",
        {m: train_calls[m] for m in col_styles},
        higher_is_better=False,
    )

    # Path entropy (APA only)
    apa_entropy = round(apa_automaton.state_visit_entropy(), 3)
    t.add_row(
        "Path Entropy (branching diversity)",
        f"[cyan]{apa_entropy}[/cyan]",
        "[dim]N/A[/dim]",
        "[dim]N/A[/dim]",
        "[cyan]APA[/cyan]",
    )
    t.add_row(
        "Runtime Branching (stateful FSA)",
        "[cyan bold]YES[/cyan bold]",
        "[dim]NO[/dim]",
        "[dim]NO[/dim]",
        "[cyan]APA[/cyan]",
    )
    t.add_row(
        "Few-Shot Demos",
        "[dim]NO[/dim]",
        "[dim]NO[/dim]",
        "[magenta bold]YES[/magenta bold]",
        "[magenta]MIPRO[/magenta]",
    )
    t.add_row(
        "LLM-guided reflection",
        "[dim]NO[/dim]",
        "[yellow bold]YES[/yellow bold]",
        "[dim]NO[/dim]",
        "[yellow]GEPA[/yellow]",
    )

    console.print(t)


def render_wins_table(
    in_dist:   Dict[str, List[Episode]],
    shifted:   Dict[str, List[Episode]],
    perturbed: Dict[str, List[Episode]],
) -> None:
    """Count wins (highest reward) per task per scenario."""
    console.print(Rule("[bold]Win Counts[/bold]"))
    methods = ["APA", "GEPA", "MIPRO"]
    wins:   Dict[str, int] = {m: 0 for m in methods}
    losses: Dict[str, int] = {m: 0 for m in methods}
    total   = 0

    for scenario_eps in [in_dist, shifted, perturbed]:
        n = min(len(scenario_eps[m]) for m in methods)
        for i in range(n):
            total += 1
            vals = {m: composite_reward(scenario_eps[m][i]) for m in methods}
            winner = max(vals, key=vals.__getitem__)
            worst  = min(vals, key=vals.__getitem__)
            wins[winner]   += 1
            losses[worst]  += 1

    t = Table(title="Wins / Losses across all scenarios", box=box.ROUNDED)
    t.add_column("Method",  style="bold")
    t.add_column("Wins",    justify="right", style="green")
    t.add_column("Losses",  justify="right", style="red")
    t.add_column("Win Rate",justify="right")

    for m in methods:
        sty = {"APA": "cyan", "GEPA": "yellow", "MIPRO": "magenta"}[m]
        t.add_row(
            f"[{sty}]{m}[/{sty}]",
            str(wins[m]),
            str(losses[m]),
            f"{100 * wins[m] / total:.1f}%",
        )
    console.print(t)
    console.print()


def render_efficiency_chart(
    train_calls: Dict[str, int],
    train_fitness: Dict[str, float],
) -> None:
    """ASCII bar chart of training LLM calls vs. peak fitness."""
    console.print(Rule("[bold]Sample Efficiency[/bold]"))

    t = Table(title="Training Cost vs. Peak Fitness", box=box.ROUNDED)
    t.add_column("Method",         style="bold")
    t.add_column("LLM Calls",      justify="right")
    t.add_column("Peak Fitness",   justify="right", style="green")
    t.add_column("Fitness / 100 calls", justify="right", style="cyan")

    for m, sty in {"APA": "cyan", "GEPA": "yellow", "MIPRO": "magenta"}.items():
        calls   = train_calls[m]
        fitness = train_fitness[m]
        eff     = fitness / (calls / 100) if calls > 0 else 0.0
        t.add_row(
            f"[{sty}]{m}[/{sty}]",
            str(calls),
            f"{fitness:.4f}",
            f"{eff:.4f}",
        )
    console.print(t)
    console.print()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    _gepa_tag  = (
        "[green]dspy.GEPA (official)[/green]" if _GEPA_USES_DSPY
        else "[dim]hand-reimpl[/dim]"
    )
    _mipro_tag = (
        "[green]dspy.MIPROv2 (official)[/green]" if _MIPRO_USES_DSPY
        else "[dim]hand-reimpl[/dim]"
    )
    console.print(Panel.fit(
        "[bold bright_white]APA  vs  GEPA  vs  MIPRO[/bold bright_white]\n\n"
        "[dim]Benchmark comparison on identical tasks, same MockLLM backend,\n"
        "same reward function, same random seed.[/dim]\n\n"
        "  [cyan bold]APA[/cyan bold]   — Adaptive Prompt Automaton (stateful FSA, evolutionary search)\n"
        f"  [yellow bold]GEPA[/yellow bold]  — Reflective Prompt Evolution (ICLR 2026, arXiv:2507.19457)\n"
        f"         [dim]backend:[/dim] {_gepa_tag}\n"
        f"  [magenta bold]MIPRO[/magenta bold] — Instruction + Demo Optimisation  (EMNLP 2024, arXiv:2406.11695)\n"
        f"         [dim]backend:[/dim] {_mipro_tag}",
        title="[bold]Method Comparison[/bold]",
        border_style="white",
    ))

    # ── Shared infrastructure ──────────────────────────────────────────────
    extractor = FeatureExtractor(long_input_threshold=120)

    # Separate LLM instances so call counts are tracked independently
    llm_apa   = get_llm_api("mock", uncertainty_rate=0.32, latency=0.01, seed=SEED)
    llm_gepa  = get_llm_api("mock", uncertainty_rate=0.32, latency=0.01, seed=SEED)
    llm_mipro = get_llm_api("mock", uncertainty_rate=0.32, latency=0.01, seed=SEED)
    # One more for inference (shared so we don't bias call counts)
    llm_eval  = get_llm_api("mock", uncertainty_rate=0.32, latency=0.01, seed=SEED)

    # ── Benchmarks ─────────────────────────────────────────────────────────
    console.print(Panel("[bold]Building Benchmark Suites[/bold]", border_style="dim"))
    qa_bench               = make_qa_benchmark()
    train_bench, test_bench = make_distribution_shift_benchmark()
    perturb_bench          = make_perturbation_benchmark()

    train_tasks  = qa_bench.inputs() + train_bench.inputs()
    eval_tasks   = qa_bench.sample(n=10, seed=SEED)
    shift_tasks  = test_bench.tasks
    perturb_tasks = perturb_bench.sample(n=9, seed=SEED)

    console.print(
        f"  Train  : [green]{len(train_tasks)}[/green] tasks\n"
        f"  Eval   : [green]{len(eval_tasks)}[/green]  (in-distribution)\n"
        f"  Shift  : [green]{len(shift_tasks)}[/green]  (distribution-shifted)\n"
        f"  Perturb: [green]{len(perturb_tasks)}[/green]  (paraphrase / noise)\n"
    )

    # ══════════════════════════════════════════════════════════════════════
    # TRAIN — APA (Evolutionary Search)
    # ══════════════════════════════════════════════════════════════════════
    console.print(Panel(
        "[cyan bold]Training APA — Evolutionary Search[/cyan bold]",
        border_style="cyan",
    ))
    apa_seed = build_apa_automaton()
    apa_search = EvolutionarySearch(
        initial_automaton = apa_seed,
        llm_api           = llm_apa,
        feature_extractor = extractor,
        reward_fn         = composite_reward,
        population_size   = 8,
        n_generations     = 10,
        mutation_rate     = 0.40,
        elite_frac        = 0.25,
        tournament_size   = 3,
        n_eval_tasks      = 5,
        seed              = SEED,
    )
    best_apa   = apa_search.run(train_tasks, console=console)
    apa_calls  = llm_apa.call_count
    apa_fitness = apa_search.best_fitness

    # ══════════════════════════════════════════════════════════════════════
    # TRAIN — GEPA (Reflective Prompt Evolution)
    # ══════════════════════════════════════════════════════════════════════
    console.print(Panel(
        f"[yellow bold]Training {_GEPA_BACKEND_LABEL}[/yellow bold]",
        border_style="yellow",
    ))

    if _GEPA_USES_DSPY:
        # Official dspy.GEPA path
        gepa_search  = _GEPABackend(
            auto             = "light",
            n_eval_tasks     = 5,
            seed             = SEED,
            uncertainty_rate = 0.32,
        )
        best_gepa    = gepa_search.run(train_tasks, console=console)
        gepa_calls   = gepa_search.call_count
        gepa_fitness = gepa_search.best_fitness
    else:
        # Hand-reimplementation fallback path
        gepa_seed   = build_apa_automaton()
        gepa_search = _GEPABackend(
            initial_automaton    = gepa_seed,
            llm_api              = llm_gepa,
            feature_extractor    = extractor,
            reward_fn            = composite_reward,
            n_iterations         = 10,
            n_trajectory_samples = 6,
            failure_threshold    = 0.40,
            n_eval_tasks         = 5,
            seed                 = SEED,
        )
        best_gepa    = gepa_search.run(train_tasks, console=console)
        gepa_calls   = llm_gepa.call_count
        gepa_fitness = gepa_search.best_fitness

    # ══════════════════════════════════════════════════════════════════════
    # TRAIN — MIPRO (Instruction + Demo Optimisation)
    # ══════════════════════════════════════════════════════════════════════
    console.print(Panel(
        f"[magenta bold]Training {_MIPRO_BACKEND_LABEL}[/magenta bold]",
        border_style="magenta",
    ))

    if _MIPRO_USES_DSPY:
        # DSPy MIPROv2 path — manages its own MockLLM internally
        mipro_search = _MIPROBackend(
            auto             = "light",
            n_eval_tasks     = 5,
            seed             = SEED,
            uncertainty_rate = 0.32,
        )
        best_mipro    = mipro_search.run(train_tasks, console=console)
        mipro_calls   = mipro_search.call_count
        mipro_fitness = mipro_search.best_fitness
    else:
        # Hand-reimplementation fallback path
        mipro_search = _MIPROBackend(
            llm_api                  = llm_mipro,
            feature_extractor        = extractor,
            reward_fn                = composite_reward,
            n_bootstrap_episodes     = 12,
            n_instruction_candidates = 6,
            n_demo_sets              = 3,
            max_demos_per_set        = 2,
            n_bayesian_rounds        = 3,
            n_eval_tasks             = 5,
            seed                     = SEED,
        )
        best_mipro    = mipro_search.run(train_tasks, console=console)
        mipro_calls   = llm_mipro.call_count
        mipro_fitness = mipro_search.best_fitness

    train_calls   = {"APA": apa_calls,  "GEPA": gepa_calls,  "MIPRO": mipro_calls}
    train_fitness = {"APA": apa_fitness,"GEPA": gepa_fitness,"MIPRO": mipro_fitness}

    # ══════════════════════════════════════════════════════════════════════
    # EVALUATE — In-distribution
    # ══════════════════════════════════════════════════════════════════════
    console.print(Panel("[bold]Evaluating on In-Distribution Tasks[/bold]", border_style="dim"))
    in_dist = {
        "APA":   eval_on_tasks(best_apa,   llm_eval, extractor, eval_tasks,
                               desc="APA (in-dist)",   colour="cyan"),
        "GEPA":  eval_on_tasks(best_gepa,  llm_eval, extractor, eval_tasks,
                               desc="GEPA (in-dist)",  colour="yellow"),
        "MIPRO": eval_on_tasks(best_mipro, llm_eval, extractor, eval_tasks,
                               desc="MIPRO (in-dist)", colour="magenta"),
    }
    render_scenario_table(
        "In-Distribution QA",
        in_dist,
        [t.task_id for t in eval_tasks],
    )

    # ══════════════════════════════════════════════════════════════════════
    # EVALUATE — Distribution shift
    # ══════════════════════════════════════════════════════════════════════
    console.print(Panel("[bold]Evaluating on Distribution-Shifted Tasks[/bold]", border_style="dim"))
    shifted = {
        "APA":   eval_on_tasks(best_apa,   llm_eval, extractor, shift_tasks,
                               desc="APA (shifted)",   colour="cyan"),
        "GEPA":  eval_on_tasks(best_gepa,  llm_eval, extractor, shift_tasks,
                               desc="GEPA (shifted)",  colour="yellow"),
        "MIPRO": eval_on_tasks(best_mipro, llm_eval, extractor, shift_tasks,
                               desc="MIPRO (shifted)", colour="magenta"),
    }
    render_scenario_table(
        "Distribution Shift Robustness",
        shifted,
        [t.task_id for t in shift_tasks],
    )

    # ══════════════════════════════════════════════════════════════════════
    # EVALUATE — Perturbation stability
    # ══════════════════════════════════════════════════════════════════════
    console.print(Panel("[bold]Evaluating on Perturbed Tasks[/bold]", border_style="dim"))
    perturbed = {
        "APA":   eval_on_tasks(best_apa,   llm_eval, extractor, perturb_tasks,
                               desc="APA (perturbed)",   colour="cyan"),
        "GEPA":  eval_on_tasks(best_gepa,  llm_eval, extractor, perturb_tasks,
                               desc="GEPA (perturbed)",  colour="yellow"),
        "MIPRO": eval_on_tasks(best_mipro, llm_eval, extractor, perturb_tasks,
                               desc="MIPRO (perturbed)", colour="magenta"),
    }
    render_scenario_table(
        "Perturbation Stability",
        perturbed,
        [t.task_id for t in perturb_tasks],
    )

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    render_summary_table(in_dist, shifted, perturbed, train_calls, best_apa)
    render_efficiency_chart(train_calls, train_fitness)
    render_wins_table(in_dist, shifted, perturbed)

    # ── Final verdict ──────────────────────────────────────────────────────
    all_avgs = {
        "APA":   (avg(in_dist["APA"])   + avg(shifted["APA"])   + avg(perturbed["APA"]))   / 3,
        "GEPA":  (avg(in_dist["GEPA"])  + avg(shifted["GEPA"])  + avg(perturbed["GEPA"]))  / 3,
        "MIPRO": (avg(in_dist["MIPRO"]) + avg(shifted["MIPRO"]) + avg(perturbed["MIPRO"])) / 3,
    }
    best_overall = max(all_avgs, key=all_avgs.__getitem__)
    sty = {"APA": "cyan", "GEPA": "yellow", "MIPRO": "magenta"}[best_overall]

    console.print(Panel.fit(
        f"[bold]Overall winner (mean across all three scenarios):[/bold]\n\n"
        f"  [bold {sty}]{best_overall}[/bold {sty}]  "
        f"avg reward = [bold {sty}]{all_avgs[best_overall]:.4f}[/bold {sty}]\n\n"
        f"  APA   : {all_avgs['APA']:.4f}  "
        f"[dim](stateful FSA, runtime branching)[/dim]\n"
        f"  GEPA  : {all_avgs['GEPA']:.4f}  "
        f"[dim](reflective evolution, Pareto frontier)[/dim]\n"
        f"  MIPRO : {all_avgs['MIPRO']:.4f}  "
        f"[dim](instruction + demo optimisation)[/dim]\n\n"
        f"[dim]Note: results use MockLLM; swap in a real API for production numbers.[/dim]",
        title="[bold]Verdict[/bold]",
        border_style=sty,
    ))


if __name__ == "__main__":
    main()
