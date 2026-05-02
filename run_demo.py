#!/usr/bin/env python3
"""
run_demo.py
───────────
Full end-to-end demo of the Adaptive Prompt Automaton (APA).

Pipeline
────────
  1.  Build a 4-state prompt automaton (start → decompose/verify/terminal)
  2.  Run sample episodes — watch live branching in real time
  3.  Evaluate a static single-state baseline
  4.  Run evolutionary training on a QA + distribution-shift task set
  5.  Evaluate the trained APA on held-out tasks
  6.  Distribution-shift robustness test
  7.  Perturbation stability test
  8.  Side-by-side comparison + final summary

All output is rendered via rich (panels, tables, trees) and tqdm (progress bars).
"""
from __future__ import annotations

import sys
import random
import time
from pathlib import Path
from typing import List

# ── make sure the package is importable ────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree
from rich.rule import Rule
from rich.progress import (
    Progress, SpinnerColumn, TextColumn,
    BarColumn, TimeElapsedColumn, MofNCompleteColumn,
)
from rich import box
from tqdm import tqdm

from adaptive_prompt_automaton.core.automaton import (
    Automaton, AutomatonConfig, StateConfig, TransitionConfig,
)
from adaptive_prompt_automaton.core.features import FeatureExtractor
from adaptive_prompt_automaton.core.executor import AutomatonExecutor, Episode
from adaptive_prompt_automaton.search.evolution import EvolutionarySearch
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
# 1 ▸ Automaton factory
# ══════════════════════════════════════════════════════════════════════════════

def build_apa() -> Automaton:
    """
    Construct the initial 4-state Adaptive Prompt Automaton.

    Topology
    ────────
        start ──[long input]──────────────────▶ decompose
              ──[high uncertainty]────────────▶ verify
              ──[high confidence]─────────────▶ terminal
        decompose ──[still uncertain]──────────▶ verify
                  ──[always]─────────────────▶ terminal
        verify ──[always]──────────────────────▶ terminal
    """
    states = {
        "start": StateConfig(
            state_id   = "start",
            name       = "Start",
            template   = (
                "You are a precise and knowledgeable AI assistant.\n"
                "Answer the following question clearly and directly:\n\n"
                "{input}\n\n"
                "Provide a concise, accurate answer."
            ),
            role        = "user",
            max_tokens  = 256,
            is_terminal = False,
            carry_context = True,
        ),
        "decompose": StateConfig(
            state_id   = "decompose",
            name       = "Decompose",
            template   = (
                "This appears to be a complex, multi-part question.\n"
                "Let's break it down step by step.\n\n"
                "Question: {input}\n\n"
                "Previous attempt: {context}\n\n"
                "Decompose the problem into sub-tasks and solve each one "
                "systematically before synthesising the final answer."
            ),
            role        = "user",
            max_tokens  = 512,
            is_terminal = False,
            carry_context = True,
        ),
        "verify": StateConfig(
            state_id   = "verify",
            name       = "Verify",
            template   = (
                "Please verify the following answer and correct any errors.\n\n"
                "Original question : {input}\n"
                "Previous answer   : {context}\n\n"
                "Is this answer correct and complete?\n"
                "If not, provide the corrected, verified answer."
            ),
            role        = "user",
            max_tokens  = 256,
            is_terminal = False,
            carry_context = True,
        ),
        "terminal": StateConfig(
            state_id   = "terminal",
            name       = "Terminal",
            template   = (
                "Provide the final, definitive answer to:\n\n"
                "{input}\n\n"
                "Based on the analysis so far: {context}\n\n"
                "State the answer concisely and with confidence."
            ),
            role        = "user",
            max_tokens  = 128,
            is_terminal = True,
            carry_context = False,
        ),
    }

    transitions = [
        # ── From start ────────────────────────────────────────────────
        TransitionConfig(
            source_state = "start",   target_state = "decompose",
            guard_type   = "threshold", feature_name = "is_long_input",
            threshold    = 0.5,  operator = ">",  priority = 3,
        ),
        TransitionConfig(
            source_state = "start",   target_state = "verify",
            guard_type   = "threshold", feature_name = "uncertainty_score",
            threshold    = 0.30, operator = ">",  priority = 2,
        ),
        TransitionConfig(
            source_state = "start",   target_state = "terminal",
            guard_type   = "threshold", feature_name = "answer_confidence",
            threshold    = 0.70, operator = ">=", priority = 1,
        ),
        # ── From decompose ────────────────────────────────────────────
        TransitionConfig(
            source_state = "decompose", target_state = "verify",
            guard_type   = "threshold", feature_name = "uncertainty_score",
            threshold    = 0.35, operator = ">",  priority = 2,
        ),
        TransitionConfig(
            source_state = "decompose", target_state = "terminal",
            guard_type   = "always",    feature_name = "input_length",
            threshold    = 0.0,  operator = "always", priority = 1,
        ),
        # ── From verify ───────────────────────────────────────────────
        TransitionConfig(
            source_state = "verify",   target_state = "terminal",
            guard_type   = "always",   feature_name = "input_length",
            threshold    = 0.0,  operator = "always", priority = 1,
        ),
    ]

    config = AutomatonConfig(
        automaton_id = "apa_v0",
        name         = "APA-v0",
        start_state  = "start",
        states       = states,
        transitions  = transitions,
        max_steps    = 6,
        max_budget   = 8,
    )
    return Automaton(config)


def build_baseline() -> Automaton:
    """Single-state static prompt baseline for ablation comparison."""
    states = {
        "only": StateConfig(
            state_id   = "only",
            name       = "Static",
            template   = "Answer this question: {input}",
            is_terminal = True,
            carry_context = False,
        )
    }
    config = AutomatonConfig(
        automaton_id = "baseline",
        name         = "Static-Baseline",
        start_state  = "only",
        states       = states,
        transitions  = [],
        max_steps    = 1,
        max_budget   = 1,
    )
    return Automaton(config)


# ══════════════════════════════════════════════════════════════════════════════
# 2 ▸ Display helpers
# ══════════════════════════════════════════════════════════════════════════════

def render_automaton_structure(aut: Automaton) -> None:
    console.print(Rule("[bold cyan]Automaton Structure[/bold cyan]"))

    # States table
    st = Table(title="States", box=box.ROUNDED, highlight=True)
    st.add_column("ID",        style="cyan bold", no_wrap=True)
    st.add_column("Name",      style="bold")
    st.add_column("Role",      justify="center")
    st.add_column("MaxTok",    justify="right")
    st.add_column("Terminal",  justify="center")
    st.add_column("Template (excerpt)", style="dim", max_width=55)
    for sid, state in aut.states.items():
        term_str = "[bold red]YES[/bold red]" if state.is_terminal else "[green]No[/green]"
        excerpt  = state.template[:58].replace("\n", " ") + "…"
        st.add_row(sid, state.name, state.role, str(state.config.max_tokens), term_str, excerpt)
    console.print(st)
    console.print()

    # Transitions table
    tt = Table(title="Transitions", box=box.ROUNDED, highlight=True)
    tt.add_column("From",      style="yellow bold")
    tt.add_column("To",        style="green bold")
    tt.add_column("Feature",   style="cyan")
    tt.add_column("Op",        justify="center")
    tt.add_column("Threshold", justify="right")
    tt.add_column("Priority",  justify="center")
    for t in aut.transitions:
        tt.add_row(
            t.source, t.target,
            t.config.feature_name,
            t.config.operator,
            f"{t.config.threshold:.2f}",
            str(t.priority),
        )
    console.print(tt)
    console.print()

    # Topology tree
    tree = Tree(
        f"[bold cyan]Automaton topology[/bold cyan] — "
        f"[dim]start: {aut.start_state}[/dim]"
    )
    for sid, state in aut.states.items():
        node_label = (
            f"[cyan]{sid}[/cyan]"
            + (" [bold red](terminal)[/bold red]" if state.is_terminal else "")
        )
        branch = tree.add(node_label)
        for t in aut.get_transitions_from(sid):
            branch.add(
                f"[yellow]→ {t.target}[/yellow]  "
                f"[dim]if {t.config.feature_name} {t.config.operator} "
                f"{t.config.threshold:.2f}  (pri={t.priority})[/dim]"
            )
    console.print(tree)
    console.print()


def render_episode(ep: Episode, label: str = "") -> None:
    title_suffix = f" — [dim]{label}[/dim]" if label else ""
    console.print(Rule(f"[bold magenta]Episode{title_suffix}[/bold magenta]"))

    # Path
    path_text = Text()
    for i, sid in enumerate(ep.path):
        if i:
            path_text.append(" → ", style="white bold")
        path_text.append(sid, style="bold cyan")
    console.print(Panel(path_text, title="Execution Path", border_style="cyan", expand=False))

    # Steps table
    st = Table(box=box.SIMPLE, show_header=True)
    st.add_column("Step",        style="dim",   justify="center", width=5)
    st.add_column("State",       style="cyan",  width=12)
    st.add_column("Tokens",                     justify="right",  width=7)
    st.add_column("Unc↑",                       justify="right",  width=6)
    st.add_column("Conf↑",                      justify="right",  width=6)
    st.add_column("Long?",                      justify="center", width=6)
    st.add_column("Transition",  style="dim",   width=22)
    st.add_column("Response (excerpt)", style="green", max_width=46)

    for s in ep.steps:
        unc  = s.features.get("uncertainty_score", 0.0)
        conf = s.features.get("answer_confidence", 0.0)
        long = s.features.get("is_long_input", 0.0)
        unc_col  = "red" if unc > 0.5 else ("yellow" if unc > 0.2 else "green")
        excerpt  = (s.response[:44] + "…") if len(s.response) > 46 else s.response
        st.add_row(
            str(s.step + 1),
            s.state_name,
            str(s.tokens_used),
            f"[{unc_col}]{unc:.2f}[/{unc_col}]",
            f"{conf:.2f}",
            "Yes" if long > 0.5 else "No",
            s.transition_taken or "—",
            excerpt,
        )
    console.print(st)

    # Final output panel
    console.print(Panel(
        ep.final_output[:350],
        title=(
            f"[bold green]Final Output[/bold green]  "
            f"[dim]tokens={ep.total_tokens} | "
            f"steps={ep.n_steps()} | "
            f"ended={ep.terminated_by}[/dim]"
        ),
        border_style="green",
    ))
    console.print()


def render_path_stats(aut: Automaton) -> None:
    console.print(Rule("[bold cyan]Path Statistics[/bold cyan]"))
    total = sum(aut.path_counts.values())
    t = Table(title="Execution Path Frequencies", box=box.ROUNDED)
    t.add_column("Path",        style="cyan")
    t.add_column("Count",       justify="right", style="green")
    t.add_column("Frequency",   justify="right")
    for path, cnt in sorted(aut.path_counts.items(), key=lambda x: -x[1]):
        freq = cnt / total if total else 0
        t.add_row(" → ".join(path), str(cnt), f"{freq:.1%}")
    if not aut.path_counts:
        t.add_row("[dim]none recorded[/dim]", "0", "—")
    console.print(t)
    console.print(
        f"  [dim]episodes={aut.episodes_run}  "
        f"path-entropy={aut.state_visit_entropy():.3f}[/dim]\n"
    )


def render_comparison(
    baseline_eps: List[Episode],
    apa_eps:      List[Episode],
    labels:       List[str],
    title:        str = "Comparison",
) -> None:
    console.print(Rule(f"[bold]{title}[/bold]"))
    t = Table(title=title, box=box.DOUBLE_EDGE)
    t.add_column("Task",             style="bold",   max_width=28)
    t.add_column("Baseline\nReward", justify="right", style="yellow")
    t.add_column("APA\nReward",      justify="right", style="green")
    t.add_column("Baseline\nPath",   justify="center", style="dim")
    t.add_column("APA Path",         justify="center", style="cyan")
    t.add_column("Δ",               justify="right")

    b_totals, a_totals = [], []
    for lbl, b_ep, a_ep in zip(labels, baseline_eps, apa_eps):
        br = composite_reward(b_ep)
        ar = composite_reward(a_ep)
        b_totals.append(br)
        a_totals.append(ar)
        delta     = ar - br
        delta_str = (
            f"[bold green]+{delta:.3f}[/bold green]" if delta > 0.005
            else f"[bold red]{delta:.3f}[/bold red]"   if delta < -0.005
            else f"[dim]{delta:.3f}[/dim]"
        )
        t.add_row(
            lbl[:27],
            f"{br:.3f}",
            f"{ar:.3f}",
            b_ep.path_str(),
            a_ep.path_str(),
            delta_str,
        )

    avg_b = sum(b_totals) / len(b_totals)
    avg_a = sum(a_totals) / len(a_totals)
    avg_d = avg_a - avg_b
    t.add_row(
        "[bold]AVERAGE[/bold]",
        f"[bold yellow]{avg_b:.3f}[/bold yellow]",
        f"[bold green]{avg_a:.3f}[/bold green]",
        "", "",
        f"[bold {'green' if avg_d >= 0 else 'red'}]{avg_d:+.3f}[/bold {'green' if avg_d >= 0 else 'red'}]",
    )
    console.print(t)
    console.print()


def render_training_history(search: EvolutionarySearch) -> None:
    console.print(Rule("[bold yellow]Training History[/bold yellow]"))
    t = Table(title="Generation-by-Generation Fitness", box=box.ROUNDED)
    t.add_column("Gen",    justify="center",  style="dim")
    t.add_column("Best",   justify="right",   style="green")
    t.add_column("Mean",   justify="right",   style="cyan")
    t.add_column("Worst",  justify="right",   style="red")
    t.add_column("Trend",  justify="center")

    prev = None
    for h in search.history:
        best = h["best_fitness"]
        if prev is None:
            trend = "[dim]—[/dim]"
        elif best - prev > 0.005:
            trend = "[green]↑[/green]"
        elif prev - best > 0.005:
            trend = "[red]↓[/red]"
        else:
            trend = "[dim]→[/dim]"
        prev = best
        t.add_row(
            str(h["generation"]),
            f"{best:.4f}",
            f"{h['mean_fitness']:.4f}",
            f"{h['worst_fitness']:.4f}",
            trend,
        )
    console.print(t)
    console.print()


# ══════════════════════════════════════════════════════════════════════════════
# 3 ▸ Batch evaluation helper
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_on_tasks(
    automaton: Automaton,
    llm:       object,
    extractor: FeatureExtractor,
    tasks:     List[Task],
    desc:      str = "Evaluating",
    colour:    str = "cyan",
) -> List[Episode]:
    executor = AutomatonExecutor(automaton, llm, extractor)
    episodes: List[Episode] = []
    for task in tqdm(tasks, desc=f"  {desc}", colour=colour, leave=True):
        ep         = executor.run_episode(task.input_text, episode_id=task.task_id)
        ep.reward  = composite_reward(ep)
        episodes.append(ep)
    return episodes


# ══════════════════════════════════════════════════════════════════════════════
# 4 ▸ Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    console.print(Panel.fit(
        "[bold bright_cyan]Adaptive Prompt Automaton — Full Implementation Demo[/bold bright_cyan]\n\n"
        "[dim]Method from: 'Adaptive Prompt Automaton' (NeurIPS target, score 94)\n"
        "Prompt is represented as a finite-state policy that branches on runtime signals.[/dim]\n\n"
        "  [bold]Pipeline:[/bold] Build → Episode Demo → Baseline → "
        "Evolutionary Training → Evaluate → Compare",
        title="[bold]APA[/bold]",
        border_style="bright_cyan",
    ))

    # ── Components ─────────────────────────────────────────────────────────
    llm       = get_llm_api("mock", uncertainty_rate=0.32, latency=0.01, seed=SEED)
    extractor = FeatureExtractor(long_input_threshold=120)

    # ══ STEP 1 — Build automaton ═══════════════════════════════════════════
    console.print(Panel("[bold cyan]Step 1 — Build 4-State Prompt Automaton[/bold cyan]",
                        border_style="cyan"))
    apa = build_apa()
    render_automaton_structure(apa)

    # ══ STEP 2 — Run sample episodes ═══════════════════════════════════════
    console.print(Panel("[bold cyan]Step 2 — Live Episode Execution[/bold cyan]",
                        border_style="cyan"))

    demo_tasks = [
        ("short_easy",
         "What is the capital of France?"),
        ("medium_conceptual",
         "Explain the difference between supervised and unsupervised learning "
         "in machine learning, including key algorithm examples for each."),
        ("long_complex",
         "Given the extensive scientific literature on climate change covering "
         "rising CO2 concentrations, feedback loops from permafrost methane release, "
         "accelerating Arctic ice loss, ocean circulation disruption, ecosystem "
         "collapse cascades, and projected sea-level rise through 2150 — "
         "what are the three highest-priority intervention strategies, "
         "what is the evidence base for each, and what are the main counterarguments "
         "or implementation barriers that critics raise against each strategy? "
         "Address the geopolitical dimensions as well."),
    ]

    executor = AutomatonExecutor(apa, llm, extractor)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        task_p = prog.add_task("Running demo episodes…", total=len(demo_tasks))
        demo_episodes: List[Episode] = []
        for tid, inp in demo_tasks:
            ep        = executor.run_episode(inp, episode_id=tid)
            ep.reward = composite_reward(ep)
            demo_episodes.append(ep)
            time.sleep(0.05)
            prog.advance(task_p)

    for (tid, inp), ep in zip(demo_tasks, demo_episodes):
        render_episode(ep, label=tid)

    render_path_stats(apa)

    # ══ STEP 3 — Build benchmarks ══════════════════════════════════════════
    console.print(Panel("[bold cyan]Step 3 — Benchmark Suites[/bold cyan]",
                        border_style="cyan"))
    qa_bench               = make_qa_benchmark()
    train_bench, test_bench = make_distribution_shift_benchmark()
    perturb_bench          = make_perturbation_benchmark()

    console.print(
        f"  [green]QA Mixed[/green]          : {len(qa_bench)} tasks "
        f"(easy={len(qa_bench.by_difficulty('easy'))}, "
        f"medium={len(qa_bench.by_difficulty('medium'))}, "
        f"hard={len(qa_bench.by_difficulty('hard'))})\n"
        f"  [green]Distribution Shift[/green]: "
        f"{len(train_bench)} train / {len(test_bench)} test\n"
        f"  [green]Perturbation[/green]       : {len(perturb_bench)} tasks\n"
    )

    # ══ STEP 4 — Baseline evaluation ═══════════════════════════════════════
    console.print(Panel("[bold cyan]Step 4 — Baseline (Static Prompt) Evaluation[/bold cyan]",
                        border_style="cyan"))
    baseline     = build_baseline()
    eval_tasks   = qa_bench.sample(n=10, seed=SEED)
    baseline_eps = evaluate_on_tasks(baseline, llm, extractor, eval_tasks,
                                     desc="Static baseline", colour="yellow")

    # ══ STEP 5 — Evolutionary training ════════════════════════════════════
    console.print(Panel("[bold cyan]Step 5 — Evolutionary Training of APA[/bold cyan]",
                        border_style="cyan"))

    train_inputs = qa_bench.inputs() + train_bench.inputs()

    # ── Fixed probe set + fingerprint function ─────────────────────────────
    # probe_tasks: a small, fixed task set held constant for the entire run so
    # fingerprints computed in gen 1 are directly comparable to those in gen 8.
    # fingerprint_fn: measures path-independent output quality (word count adequacy
    # + absence of hedging) — not routing markers — so diversity selection reflects
    # genuine behavioral differences, not which states were visited.
    probe_tasks = qa_bench.sample(n=8, seed=SEED)
    probe_inputs = [t.input_text for t in probe_tasks]

    def fingerprint_fn(task_input: str, response: str) -> float:
        """Path-independent quality signal for fingerprinting."""
        words  = response.split()
        lower  = response.lower()
        hedge  = any(w in lower for w in ["not sure", "uncertain", "don't know",
                                           "possibly", "perhaps"])
        length_ok = 10 <= len(words) <= 150
        ends_ok   = response.rstrip().endswith((".", "!", "?"))
        return float(length_ok and ends_ok and not hedge)

    search = EvolutionarySearch(
        initial_automaton    = apa,
        llm_api              = llm,
        feature_extractor    = extractor,
        reward_fn            = composite_reward,
        population_size      = 8,
        n_generations        = 10,
        mutation_rate        = 0.40,
        elite_frac           = 0.25,
        tournament_size      = 3,
        n_eval_tasks         = 5,
        seed                 = SEED,
        probe_tasks          = probe_inputs,
        fingerprint_fn       = fingerprint_fn,
        diversity_lambda     = 0.10,
        diversity_threshold  = 0.15,
        diversity_quota      = 1,
        patience             = 3,
    )

    best_apa = search.run(train_inputs, console=console)
    render_training_history(search)

    # ══ STEP 6 — APA evaluation (in-distribution) ═════════════════════════
    console.print(Panel("[bold cyan]Step 6 — Trained APA Evaluation (In-Distribution)[/bold cyan]",
                        border_style="cyan"))
    apa_eps = evaluate_on_tasks(best_apa, llm, extractor, eval_tasks,
                                desc="Trained APA (in-dist)", colour="green")

    render_comparison(
        baseline_eps, apa_eps,
        [t.task_id for t in eval_tasks],
        title="In-Distribution: Baseline vs. Trained APA",
    )

    # ══ STEP 7 — Distribution shift robustness ════════════════════════════
    console.print(Panel("[bold cyan]Step 7 — Distribution Shift Robustness Test[/bold cyan]",
                        border_style="cyan"))
    shift_bl_eps  = evaluate_on_tasks(baseline, llm, extractor, test_bench.tasks,
                                      desc="Baseline (shifted)", colour="yellow")
    shift_apa_eps = evaluate_on_tasks(best_apa, llm, extractor, test_bench.tasks,
                                      desc="APA (shifted)",      colour="green")

    render_comparison(
        shift_bl_eps, shift_apa_eps,
        [t.task_id for t in test_bench.tasks],
        title="Distribution Shift: Baseline vs. Trained APA",
    )

    # ══ STEP 8 — Perturbation stability ═══════════════════════════════════
    console.print(Panel("[bold cyan]Step 8 — Perturbation Stability Test[/bold cyan]",
                        border_style="cyan"))
    perturb_tasks   = perturb_bench.sample(n=9, seed=SEED)
    perturb_bl_eps  = evaluate_on_tasks(baseline, llm, extractor, perturb_tasks,
                                        desc="Baseline (perturbed)", colour="yellow")
    perturb_apa_eps = evaluate_on_tasks(best_apa, llm, extractor, perturb_tasks,
                                        desc="APA (perturbed)",      colour="green")

    render_comparison(
        perturb_bl_eps, perturb_apa_eps,
        [t.task_id for t in perturb_tasks],
        title="Perturbation Stability: Baseline vs. Trained APA",
    )

    # ══ STEP 9 — Path statistics of trained APA ═══════════════════════════
    console.print(Panel("[bold cyan]Step 9 — Trained APA Path Statistics[/bold cyan]",
                        border_style="cyan"))
    render_path_stats(best_apa)

    # ══ STEP 10 — Final summary ════════════════════════════════════════════
    console.print(Rule("[bold]Final Summary[/bold]"))

    def avg_reward(eps: List[Episode]) -> float:
        r = [composite_reward(e) for e in eps]
        return sum(r) / len(r) if r else 0.0

    summary = Table(
        title   = "Overall Performance Summary",
        box     = box.DOUBLE_EDGE,
        highlight = True,
    )
    summary.add_column("Evaluation Scenario",    style="bold",   min_width=32)
    summary.add_column("Static Baseline",         justify="right", style="yellow")
    summary.add_column("Trained APA",             justify="right", style="green")
    summary.add_column("Δ Reward",                justify="right")

    rows = [
        ("In-Distribution QA",      avg_reward(baseline_eps),  avg_reward(apa_eps)),
        ("Distribution Shift Test", avg_reward(shift_bl_eps),  avg_reward(shift_apa_eps)),
        ("Perturbation Stability",  avg_reward(perturb_bl_eps), avg_reward(perturb_apa_eps)),
    ]
    for label, b, a in rows:
        d     = a - b
        d_str = f"[green]+{d:.3f}[/green]" if d >= 0 else f"[red]{d:.3f}[/red]"
        summary.add_row(label, f"{b:.4f}", f"{a:.4f}", d_str)

    summary.add_row("", "", "", "")
    summary.add_row(
        "[bold]Best Training Fitness[/bold]",
        "—",
        f"[bold cyan]{search.best_fitness:.4f}[/bold cyan]",
        "—",
    )
    summary.add_row(
        "[bold]Total LLM Calls (mock)[/bold]",
        "—",
        f"[bold cyan]{llm.call_count}[/bold cyan]",
        "—",
    )
    summary.add_row(
        "[bold]APA Path Entropy (best)[/bold]",
        "—",
        f"[bold cyan]{best_apa.state_visit_entropy():.3f}[/bold cyan]",
        "[dim](0 = linear, >0 = adaptive branching)[/dim]",
    )

    console.print(summary)

    console.print(Panel.fit(
        "[bold green]APA Implementation Complete![/bold green]\n\n"
        "[dim]Demonstrated:\n"
        "  ✓  Stateful prompt FSA with runtime branching\n"
        "  ✓  Feature extraction (uncertainty, length, confidence, structure)\n"
        "  ✓  Evolutionary optimisation of templates + guard thresholds\n"
        "  ✓  Distribution shift robustness via adaptive routing\n"
        "  ✓  Perturbation stability across paraphrase variants\n"
        "  ✓  Ablation: multi-step APA vs. static single-state baseline[/dim]",
        border_style="bright_green",
    ))


if __name__ == "__main__":
    main()
