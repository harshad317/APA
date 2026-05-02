#!/usr/bin/env python3
"""
ifbench_eval.py
───────────────
Evaluate APA, GEPA, and MIPRO (DSPy MIPROv2) on the IFBench
instruction-following benchmark with all six constraint parsers.

Usage
─────
    # Run all methods on all parsers (default)
    python ifbench_eval.py

    # Restrict to specific parser types
    python ifbench_eval.py --parsers keyword length

    # Restrict to specific methods
    python ifbench_eval.py --methods apa gepa

    # Skip training (load cached automata) — not yet implemented; placeholder
    python ifbench_eval.py --eval-only

    # Verbose: print each episode's constraint check
    python ifbench_eval.py --verbose

Benchmark structure
───────────────────
  48 IFTasks (8 per parser type × 6 types) split 24/24 train/test.
  Compliance score per task: 0–1 via deterministic parser.
  Pass threshold: ≥ 0.5 compliance.

Methods
───────
  APA   — 4-state automaton trained via evolutionary search
  GEPA  — reflective prompt evolution (Pareto frontier)
  MIPRO — DSPy MIPROv2 (falls back to hand-reimpl if dspy unavailable)
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich import box
from tqdm import tqdm

from adaptive_prompt_automaton.core.automaton import (
    Automaton, AutomatonConfig, StateConfig, TransitionConfig,
)
from adaptive_prompt_automaton.core.features import FeatureExtractor
from adaptive_prompt_automaton.core.executor import AutomatonExecutor, Episode
from adaptive_prompt_automaton.search.evolution import EvolutionarySearch
from adaptive_prompt_automaton.search.gepa import GEPASearch
from adaptive_prompt_automaton.eval.ifbench import (
    make_ifbench_benchmark,
    ifbench_reward,
    per_parser_accuracy,
    IFTask,
    PARSERS,
)
from adaptive_prompt_automaton.utils.api import get_llm_api

# DSPy backend auto-detection (same as compare.py)
try:
    from adaptive_prompt_automaton.search.mipro_dspy import MIPRODSPySearch as _MIPROBackend
    _MIPRO_LABEL    = "MIPRO (DSPy MIPROv2)"
    _MIPRO_USE_DSPY = True
except ImportError:
    from adaptive_prompt_automaton.search.mipro import MIPROSearch as _MIPROBackend  # type: ignore
    _MIPRO_LABEL    = "MIPRO (hand-reimpl)"
    _MIPRO_USE_DSPY = False

console = Console(width=160)
SEED    = 42
random.seed(SEED)

ALL_METHODS  = ["apa", "gepa", "mipro"]
ALL_PARSERS  = list(PARSERS.keys())   # keyword, length, format, startend, case, composite

METHOD_COLOURS = {"apa": "cyan", "gepa": "yellow", "mipro": "magenta"}
METHOD_LABELS  = {"apa": "APA", "gepa": "GEPA", "mipro": _MIPRO_LABEL}


# ══════════════════════════════════════════════════════════════════════════════
# APA topology (same 4-state FSA as compare.py)
# ══════════════════════════════════════════════════════════════════════════════

def build_apa_seed() -> Automaton:
    states = {
        "start": StateConfig(
            state_id="start", name="Start",
            template=(
                "You are a precise assistant that follows instructions exactly.\n"
                "Answer the following question, obeying ALL constraints stated:\n\n"
                "{input}\n\nProvide a concise, accurate, constraint-compliant answer."
            ),
            role="user", max_tokens=256, is_terminal=False, carry_context=True,
        ),
        "decompose": StateConfig(
            state_id="decompose", name="Decompose",
            template=(
                "This is a complex question. First identify the constraints, "
                "then answer step by step while obeying them.\n\n"
                "Question + constraints: {input}\n\nPrevious attempt: {context}\n\n"
                "Step 1: list constraints. Step 2: verify. Step 3: answer."
            ),
            role="user", max_tokens=512, is_terminal=False, carry_context=True,
        ),
        "verify": StateConfig(
            state_id="verify", name="Verify",
            template=(
                "Check that the answer below satisfies all stated constraints.\n\n"
                "Question: {input}\nDraft answer: {context}\n\n"
                "Verify each constraint, correct any violation, return final answer."
            ),
            role="user", max_tokens=256, is_terminal=False, carry_context=True,
        ),
        "terminal": StateConfig(
            state_id="terminal", name="Terminal",
            template=(
                "State the final constraint-compliant answer to:\n\n{input}\n\n"
                "Based on: {context}\n\nFinal answer (satisfying all constraints):"
            ),
            role="user", max_tokens=128, is_terminal=True, carry_context=False,
        ),
    }
    transitions = [
        TransitionConfig(source_state="start",     target_state="decompose",
                         guard_type="threshold",   feature_name="is_long_input",
                         threshold=0.5,  operator=">",      priority=3),
        TransitionConfig(source_state="start",     target_state="verify",
                         guard_type="threshold",   feature_name="uncertainty_score",
                         threshold=0.30, operator=">",      priority=2),
        TransitionConfig(source_state="start",     target_state="terminal",
                         guard_type="threshold",   feature_name="answer_confidence",
                         threshold=0.70, operator=">=",     priority=1),
        TransitionConfig(source_state="decompose", target_state="verify",
                         guard_type="threshold",   feature_name="uncertainty_score",
                         threshold=0.35, operator=">",      priority=2),
        TransitionConfig(source_state="decompose", target_state="terminal",
                         guard_type="always",      feature_name="input_length",
                         threshold=0.0,  operator="always", priority=1),
        TransitionConfig(source_state="verify",    target_state="terminal",
                         guard_type="always",      feature_name="input_length",
                         threshold=0.0,  operator="always", priority=1),
    ]
    return Automaton(AutomatonConfig(
        automaton_id="apa_if_seed", name="APA-IFBench-Seed",
        start_state="start", states=states,
        transitions=transitions, max_steps=6, max_budget=8,
    ))


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation helper
# ══════════════════════════════════════════════════════════════════════════════

def eval_on_iftasks(
    automaton: Automaton,
    llm:       object,
    extractor: FeatureExtractor,
    tasks:     List[IFTask],
    desc:      str  = "Evaluating",
    colour:    str  = "cyan",
    verbose:   bool = False,
) -> List[Episode]:
    executor = AutomatonExecutor(automaton, llm, extractor)
    episodes: List[Episode] = []

    for task in tqdm(tasks, desc=f"  {desc}", colour=colour, leave=True):
        ep    = executor.run_episode(task.input_text, episode_id=task.task_id)
        score = ifbench_reward(ep, task)
        ep.reward = score

        if verbose:
            status = "✓" if score >= 0.5 else "✗"
            console.print(
                f"    [{colour}]{status}[/{colour}] "
                f"[dim]{task.task_id}[/dim] "
                f"[{colour}]{task.constraint_type}[/{colour}] "
                f"score={score:.2f}  "
                f"[dim]{task.constraint_desc[:40]}[/dim]"
            )
        episodes.append(ep)
    return episodes


# ══════════════════════════════════════════════════════════════════════════════
# Rich rendering
# ══════════════════════════════════════════════════════════════════════════════

def render_per_parser_table(
    results:        Dict[str, List[Episode]],
    tasks:          List[IFTask],
    active_parsers: List[str],
    title:          str = "Per-Parser Compliance",
) -> None:
    """One row per parser type, one column per method."""
    methods = [m for m in ALL_METHODS if m in results]

    tbl = Table(title=title, box=box.DOUBLE_EDGE, highlight=True)
    tbl.add_column("Parser Type",  style="bold", min_width=14)
    tbl.add_column("N",            justify="right", style="dim", min_width=3)
    for mth in methods:
        mcol = METHOD_COLOURS[mth]
        tbl.add_column(f"{METHOD_LABELS[mth].split()[0]} Mean", justify="right",
                       style=mcol, min_width=8)
        tbl.add_column("Pass%", justify="right", style=mcol, min_width=7)
    tbl.add_column("Best", justify="center", min_width=6)

    # Aggregate per parser per method
    type_stats: Dict[str, Dict[str, Dict]] = {}
    for mth, eps in results.items():
        type_stats[mth] = per_parser_accuracy(eps, tasks)

    # Collect all constraint types present in the filtered task set
    all_types = sorted({task.constraint_type for task in tasks
                        if task.constraint_type in active_parsers})

    for ptype in all_types:
        row: List[str] = [ptype]
        n = type_stats[methods[0]].get(ptype, {}).get("n", 0)
        row.append(str(n))

        best_m = max(methods,
                     key=lambda mth, pt=ptype: type_stats[mth].get(pt, {}).get("mean_score", 0.0))
        for mth in methods:
            st = type_stats[mth].get(ptype, {"mean_score": 0.0, "pass_rate": 0.0})
            row.append(f"{st['mean_score']:.3f}")
            row.append(f"{st['pass_rate']*100:.1f}%")

        bcol = METHOD_COLOURS.get(best_m, "white")
        row.append(f"[bold {bcol}]{best_m.upper()}[/bold {bcol}]")
        tbl.add_row(*row)

    # Overall row
    overall_row: List[str] = ["[bold]OVERALL[/bold]", str(len(tasks))]
    best_m = max(methods,
                 key=lambda mth: sum(ep.reward for ep in results[mth]) / len(results[mth]))
    for mth in methods:
        omean  = sum(ep.reward for ep in results[mth]) / len(results[mth])
        oprate = sum(ep.reward >= 0.5 for ep in results[mth]) / len(results[mth])
        overall_row.append(f"[bold]{omean:.3f}[/bold]")
        overall_row.append(f"[bold]{oprate*100:.1f}%[/bold]")
    bcol = METHOD_COLOURS.get(best_m, "white")
    overall_row.append(f"[bold {bcol}]{best_m.upper()}[/bold {bcol}]")
    tbl.add_row(*overall_row)

    console.print(tbl)
    console.print()


def render_per_task_table(
    results: Dict[str, List[Episode]],
    tasks:   List[IFTask],
    title:   str = "Per-Task Compliance Scores",
) -> None:
    """Full per-task breakdown (optional verbose table)."""
    methods = [m for m in ALL_METHODS if m in results]
    t = Table(title=title, box=box.SIMPLE_HEAD, highlight=True)
    t.add_column("ID",         style="dim",   max_width=14)
    t.add_column("Type",       style="bold",  max_width=12)
    t.add_column("Constraint", style="dim",   max_width=35)
    for m in methods:
        t.add_column(m.upper(), justify="right", style=METHOD_COLOURS[m], min_width=7)
    t.add_column("Best", justify="center")

    for i, task in enumerate(tasks):
        scores = {m: results[m][i].reward for m in methods}
        best   = max(scores, key=scores.__getitem__)
        col    = METHOD_COLOURS.get(best, "white")
        t.add_row(
            task.task_id[:13],
            task.constraint_type,
            task.constraint_desc[:34],
            *[f"{scores[m]:.2f}" for m in methods],
            f"[bold {col}]{best.upper()}[/bold {col}]",
        )
    console.print(t)
    console.print()


def render_summary_panel(
    results:        Dict[str, List[Episode]],
    tasks:          List[IFTask],
    active_parsers: List[str],
) -> None:
    methods = [m for m in ALL_METHODS if m in results]
    lines   = []
    for m in methods:
        mean  = sum(ep.reward for ep in results[m]) / len(results[m])
        prate = sum(ep.reward >= 0.5 for ep in results[m]) / len(results[m])
        col   = METHOD_COLOURS[m]
        lines.append(
            f"  [{col} bold]{METHOD_LABELS[m].split()[0]:<6}[/{col} bold]"
            f"  mean={mean:.3f}  pass={prate*100:.1f}%"
        )

    best_m = max(methods,
                 key=lambda m: sum(ep.reward for ep in results[m]) / len(results[m]))
    col    = METHOD_COLOURS[best_m]
    console.print(Panel(
        f"[bold]IFBench results — {len(tasks)} tasks — "
        f"parsers: {', '.join(active_parsers)}[/bold]\n\n"
        + "\n".join(lines) + "\n\n"
        + f"[dim]Best overall:[/dim] [{col} bold]{METHOD_LABELS[best_m]}[/{col} bold]",
        title="[bold]IFBench Summary[/bold]",
        border_style=col,
    ))


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(args: argparse.Namespace) -> None:
    active_methods  = args.methods
    active_parsers  = args.parsers
    verbose         = args.verbose

    # ── Banner ─────────────────────────────────────────────────────────────
    _dspy_tag = "[green]DSPy MIPROv2[/green]" if _MIPRO_USE_DSPY else "[dim]hand-reimpl[/dim]"
    console.print(Panel.fit(
        "[bold bright_white]IFBench Evaluation[/bold bright_white]\n\n"
        "[dim]Instruction-Following benchmark — verifiable constraint parsers\n"
        "Same MockLLM, same seed, deterministic scoring.[/dim]\n\n"
        f"  [cyan bold]APA[/cyan bold]   — Adaptive Prompt Automaton (stateful FSA)\n"
        f"  [yellow bold]GEPA[/yellow bold]  — Reflective Prompt Evolution\n"
        f"  [magenta bold]MIPRO[/magenta bold] — {_dspy_tag}\n\n"
        f"  Methods  : [green]{', '.join(active_methods)}[/green]\n"
        f"  Parsers  : [green]{', '.join(active_parsers)}[/green]",
        title="[bold]IFBench — APA vs GEPA vs MIPRO[/bold]",
        border_style="white",
    ))

    # ── Build benchmark ─────────────────────────────────────────────────────
    console.print(Panel("[bold]Building IFBench[/bold]", border_style="dim"))
    train_suite, test_suite = make_ifbench_benchmark()

    # Filter to active parsers
    train_tasks: List[IFTask] = [
        t for t in train_suite.tasks if t.constraint_type in active_parsers
    ]
    test_tasks: List[IFTask] = [
        t for t in test_suite.tasks if t.constraint_type in active_parsers
    ]

    console.print(
        f"  Train: [green]{len(train_tasks)}[/green] tasks  "
        f"Test: [green]{len(test_tasks)}[/green] tasks  "
        f"Parsers: [green]{', '.join(active_parsers)}[/green]"
    )

    extractor = FeatureExtractor(long_input_threshold=120)
    llm_apa   = get_llm_api("mock", uncertainty_rate=0.32, latency=0.0, seed=SEED)
    llm_gepa  = get_llm_api("mock", uncertainty_rate=0.32, latency=0.0, seed=SEED)
    llm_mipro = get_llm_api("mock", uncertainty_rate=0.32, latency=0.0, seed=SEED)
    llm_eval  = get_llm_api("mock", uncertainty_rate=0.32, latency=0.0, seed=SEED)

    train_inputs = [t.input_text for t in train_tasks]

    trained: Dict[str, Automaton] = {}

    # ══════════════════════════════════════════════════════════════════════
    # TRAIN
    # ══════════════════════════════════════════════════════════════════════

    if "apa" in active_methods:
        console.print(Panel(
            "[cyan bold]Training APA — Evolutionary Search[/cyan bold]",
            border_style="cyan",
        ))
        search = EvolutionarySearch(
            initial_automaton = build_apa_seed(),
            llm_api           = llm_apa,
            feature_extractor = extractor,
            reward_fn         = lambda ep: ifbench_reward(
                ep, train_tasks[ep.episode_id % len(train_tasks)]
                if isinstance(ep.episode_id, int)
                else _find_task(ep.episode_id, train_tasks)
            ),
            population_size  = 6,
            n_generations    = 8,
            mutation_rate    = 0.40,
            elite_frac       = 0.25,
            tournament_size  = 3,
            n_eval_tasks     = min(5, len(train_tasks)),
            seed             = SEED,
        )
        trained["apa"] = search.run(train_inputs, console=console)

    if "gepa" in active_methods:
        console.print(Panel(
            "[yellow bold]Training GEPA — Reflective Prompt Evolution[/yellow bold]",
            border_style="yellow",
        ))
        search = GEPASearch(
            initial_automaton    = build_apa_seed(),
            llm_api              = llm_gepa,
            feature_extractor    = extractor,
            reward_fn            = lambda ep: ifbench_reward(
                ep, train_tasks[ep.episode_id % len(train_tasks)]
                if isinstance(ep.episode_id, int)
                else _find_task(ep.episode_id, train_tasks)
            ),
            n_iterations         = 8,
            n_trajectory_samples = 5,
            failure_threshold    = 0.40,
            n_eval_tasks         = min(5, len(train_tasks)),
            seed                 = SEED,
        )
        trained["gepa"] = search.run(train_inputs, console=console)

    if "mipro" in active_methods:
        console.print(Panel(
            f"[magenta bold]Training {_MIPRO_LABEL}[/magenta bold]",
            border_style="magenta",
        ))
        if _MIPRO_USE_DSPY:
            search = _MIPROBackend(
                auto             = "light",
                n_eval_tasks     = min(5, len(train_tasks)),
                seed             = SEED,
                uncertainty_rate = 0.32,
            )
            trained["mipro"] = search.run(train_inputs, console=console)
        else:
            search = _MIPROBackend(
                llm_api                  = llm_mipro,
                feature_extractor        = extractor,
                reward_fn                = lambda ep: ifbench_reward(
                    ep, train_tasks[ep.episode_id % len(train_tasks)]
                    if isinstance(ep.episode_id, int)
                    else _find_task(ep.episode_id, train_tasks)
                ),
                n_bootstrap_episodes     = 10,
                n_instruction_candidates = 5,
                n_demo_sets              = 3,
                max_demos_per_set        = 2,
                n_bayesian_rounds        = 3,
                n_eval_tasks             = min(5, len(train_tasks)),
                seed                     = SEED,
            )
            trained["mipro"] = search.run(train_inputs, console=console)

    # ══════════════════════════════════════════════════════════════════════
    # EVALUATE on test split
    # ══════════════════════════════════════════════════════════════════════
    console.print(Panel(
        "[bold]Evaluating on IFBench Test Split[/bold]", border_style="dim"
    ))
    results: Dict[str, List[Episode]] = {}

    for method, automaton in trained.items():
        col = METHOD_COLOURS[method]
        eps = eval_on_iftasks(
            automaton, llm_eval, extractor, test_tasks,
            desc    = f"{method.upper()} (IFBench test)",
            colour  = col,
            verbose = verbose,
        )
        results[method] = eps

    # ══════════════════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════════════════
    console.print(Rule("[bold]Per-Parser Compliance[/bold]"))
    render_per_parser_table(results, test_tasks, active_parsers,
                            title="IFBench — Compliance by Parser Type")

    if verbose or len(test_tasks) <= 24:
        console.print(Rule("[bold]Per-Task Breakdown[/bold]"))
        render_per_task_table(results, test_tasks,
                              title="IFBench — Per-Task Scores")

    render_summary_panel(results, test_tasks, active_parsers)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: look up an IFTask by episode_id string
# ──────────────────────────────────────────────────────────────────────────────

def _find_task(episode_id: str, tasks: List[IFTask]) -> IFTask:
    for t in tasks:
        if t.task_id == episode_id:
            return t
    return tasks[0]   # fallback


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="IFBench: evaluate APA / GEPA / MIPRO on instruction-following tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
────────
  # All methods, all parsers
  python ifbench_eval.py

  # Only keyword + length parsers
  python ifbench_eval.py --parsers keyword length

  # Only APA and MIPRO
  python ifbench_eval.py --methods apa mipro

  # APA only, all parsers, verbose
  python ifbench_eval.py --methods apa --verbose

  # Single parser type
  python ifbench_eval.py --parsers composite

Parser types available
──────────────────────
  keyword    must include / exclude specific words
  length     word/sentence count constraint
  format     bullet list, numbered list, JSON, code block, table
  startend   must start or end with a given phrase
  case       ALL CAPS / lowercase / Title Case
  composite  two constraints combined (AND logic)
        """,
    )
    p.add_argument(
        "--methods", nargs="+",
        choices=ALL_METHODS, default=ALL_METHODS,
        metavar="METHOD",
        help=f"Methods to evaluate (default: all). Choices: {ALL_METHODS}",
    )
    p.add_argument(
        "--parsers", nargs="+",
        choices=ALL_PARSERS, default=ALL_PARSERS,
        metavar="PARSER",
        help=f"Parser types to include (default: all). Choices: {ALL_PARSERS}",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print per-episode constraint check results during evaluation",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    SEED = args.seed
    random.seed(SEED)
    main(args)
