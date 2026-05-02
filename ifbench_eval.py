#!/usr/bin/env python3
"""
ifbench_eval.py
───────────────
Evaluate APA, GEPA, and MIPRO on the official IFBench benchmark.

Each method works as it does in its respective paper:

  APA   — 4-state FSA trained via evolutionary search (composite_reward).
           Unchanged from the APA paper.

  GEPA  — Official dspy.GEPA with the IFBench two-stage rewriter pipeline:
             Stage 1 (fixed)  : "Respond to the query" → draft_answer
             Stage 2 (learned): dspy.GEPA evolves the rewrite instruction
           Trained on allenai/IF_multi_constraints_upto5 (IF-RLVR pool).
           Scored with the official allenai/IFBench constraint verifiers.

  MIPRO — Official dspy.MIPROv2 with the same two-stage rewriter pipeline.
           Trained on IF-RLVR pool with Bayesian instruction optimisation.
           Scored with the official allenai/IFBench constraint verifiers.

All three methods are evaluated on the official IFBench test set (300 prompts)
using prompt_loose accuracy — the primary metric from the IFBench paper.

Usage
─────
  # All methods (requires OPENAI_API_KEY for GEPA and MIPRO)
  python ifbench_eval.py --model gpt-4.1-mini

  # Mock mode (APA only — GEPA/MIPRO need a real LLM)
  python ifbench_eval.py --methods apa

  # Custom split sizes
  python ifbench_eval.py --model gpt-4.1-mini --train-size 200 --val-size 100

  # Skip GEPA/MIPRO training, only run APA
  python ifbench_eval.py --methods apa --model mock

Data
────
  Train/Val : allenai/IF_multi_constraints_upto5 on HuggingFace (default 300/100)
  Test      : vendor/ifbench/data/IFBench_test.jsonl (300 prompts, official)
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich import box
from tqdm import tqdm

# ── APA core ──────────────────────────────────────────────────────────────────
from adaptive_prompt_automaton.core.automaton import (
    Automaton, AutomatonConfig, StateConfig, TransitionConfig,
)
from adaptive_prompt_automaton.core.features import FeatureExtractor
from adaptive_prompt_automaton.core.executor import AutomatonExecutor, Episode
from adaptive_prompt_automaton.search.evolution import EvolutionarySearch
from adaptive_prompt_automaton.eval.benchmarks import composite_reward, Task
from adaptive_prompt_automaton.utils.api import get_llm_api

# ── Official IFBench ──────────────────────────────────────────────────────────
from adaptive_prompt_automaton.eval.ifbench_official import (
    IFBenchOfficialExample,
    IFBenchOfficialScorer,
    load_ifbench_test,
    load_ifbench_train_val,
)

# ── GEPA and MIPRO (official DSPy) ────────────────────────────────────────────
try:
    import dspy
    from adaptive_prompt_automaton.search.gepa_dspy import GEPADSPySearch
    from adaptive_prompt_automaton.search.mipro_dspy import MIPRODSPySearch
    _HAS_DSPY = True
except ImportError:
    _HAS_DSPY = False

console    = Console(width=160)
SEED       = 42
ALL_METHODS = ["apa", "gepa", "mipro"]


# ══════════════════════════════════════════════════════════════════════════════
# APA seed automaton (4-state FSA, unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def build_apa_seed() -> Automaton:
    """4-state APA FSA seed for evolutionary training."""
    states = {
        "start": StateConfig(
            state_id="start", name="Start",
            template=(
                "You are a precise assistant that follows instructions exactly.\n"
                "Answer the following, obeying ALL stated constraints:\n\n"
                "{input}\n\nProvide a concise, constraint-compliant answer."
            ),
            role="user", max_tokens=256, is_terminal=False, carry_context=True,
        ),
        "decompose": StateConfig(
            state_id="decompose", name="Decompose",
            template=(
                "This is a constrained task. Identify constraints, then answer.\n\n"
                "Task: {input}\nPrevious attempt: {context}\n\n"
                "Step 1: list all constraints. "
                "Step 2: verify your answer against each. "
                "Step 3: give the final answer."
            ),
            role="user", max_tokens=512, is_terminal=False, carry_context=True,
        ),
        "verify": StateConfig(
            state_id="verify", name="Verify",
            template=(
                "Check that the answer below satisfies all stated constraints.\n\n"
                "Task: {input}\nAnswer: {context}\n\n"
                "If any constraint is violated, rewrite the answer. "
                "Otherwise, output the answer unchanged."
            ),
            role="user", max_tokens=256, is_terminal=False, carry_context=True,
        ),
        "terminal": StateConfig(
            state_id="terminal", name="Terminal",
            template="{context}",
            role="assistant", max_tokens=256, is_terminal=True, carry_context=False,
        ),
    }
    transitions = [
        TransitionConfig(
            source_state="start", target_state="decompose",
            condition="complexity_high", priority=1,
        ),
        TransitionConfig(
            source_state="start", target_state="verify",
            condition="default", priority=2,
        ),
        TransitionConfig(
            source_state="decompose", target_state="verify",
            condition="default", priority=1,
        ),
        TransitionConfig(
            source_state="verify", target_state="terminal",
            condition="default", priority=1,
        ),
    ]
    cfg = AutomatonConfig(
        name="apa_ifbench_seed", start_state="start",
        states=states, transitions=transitions,
    )
    return Automaton(cfg)


# ══════════════════════════════════════════════════════════════════════════════
# APA training helpers
# ══════════════════════════════════════════════════════════════════════════════

def _apa_train_tasks_from_ifbench(
    examples: List[IFBenchOfficialExample],
) -> List[Task]:
    """Wrap IFBench examples as APA Task objects for evolutionary training."""
    return [
        Task(
            task_id    = ex.key,
            input_text = ex.prompt,
            expected   = "",
            category   = "ifbench",
            difficulty = "medium",
        )
        for ex in examples
    ]


def _apa_reward_for_ifbench(
    episode:  Episode,
    example:  IFBenchOfficialExample,
    scorer:   IFBenchOfficialScorer,
) -> float:
    """APA reward on IFBench: official prompt_loose score."""
    response = episode.final_output or ""
    return scorer.prompt_loose(example, response)


# ══════════════════════════════════════════════════════════════════════════════
# APA evaluation on official IFBench test set
# ══════════════════════════════════════════════════════════════════════════════

def eval_apa_on_ifbench(
    automaton: Automaton,
    llm:       object,
    examples:  List[IFBenchOfficialExample],
    scorer:    IFBenchOfficialScorer,
    desc:      str = "APA eval",
    verbose:   bool = False,
) -> Dict[str, float]:
    """
    Run APA automaton on official IFBench test examples.
    Returns dict with prompt_loose, instruction_loose, n.
    """
    extractor = FeatureExtractor()
    executor  = AutomatonExecutor(
        automaton         = automaton,
        llm_api           = llm,
        feature_extractor = extractor,
    )
    prompt_loose_scores   = []
    instruction_loose_scores = []

    bar = tqdm(examples, desc=f"  {desc}", colour="cyan", leave=False)
    for ex in bar:
        episode  = executor.run_episode(ex.prompt, verbose=verbose)
        response = episode.final_output or ""
        pl = scorer.prompt_loose(example=ex, response=response)
        il = scorer.instruction_loose(example=ex, response=response)
        prompt_loose_scores.append(pl)
        instruction_loose_scores.append(il)
        if verbose:
            console.print(
                f"  [dim]{ex.key}[/dim] pl={pl:.1f} il={il:.2f} "
                f"[dim]{ex.instruction_id_list}[/dim]"
            )

    n = len(examples)
    return {
        "prompt_loose":      sum(prompt_loose_scores) / n if n else 0.0,
        "instruction_loose": sum(instruction_loose_scores) / n if n else 0.0,
        "n":                 n,
    }


# ══════════════════════════════════════════════════════════════════════════════
# GEPA / MIPRO evaluation on official IFBench test set
# ══════════════════════════════════════════════════════════════════════════════

def eval_dspy_program_on_ifbench(
    program:  object,
    dspy_lm:  object,
    examples: List[IFBenchOfficialExample],
    scorer:   IFBenchOfficialScorer,
    desc:     str  = "DSPy eval",
    colour:   str  = "magenta",
    verbose:  bool = False,
) -> Dict[str, float]:
    """
    Run a compiled IFBenchRewriterProgram on official IFBench test examples.
    Uses 2-stage pipeline: stage-1 draft → stage-2 optimised rewrite.
    """
    from adaptive_prompt_automaton.search.gepa_dspy import generate_stage1_drafts

    console.print(f"  [dim]Stage 1 drafts for {len(examples)} examples …[/dim]")
    drafts = generate_stage1_drafts(examples, dspy_lm)

    prompt_loose_scores      = []
    instruction_loose_scores = []

    bar = tqdm(
        zip(examples, drafts),
        total   = len(examples),
        desc    = f"  {desc}",
        colour  = colour,
        leave   = False,
    )
    with dspy.context(lm=dspy_lm):
        for ex, draft in bar:
            try:
                pred     = program(
                    prompt          = ex.prompt,
                    draft_answer    = draft,
                    constraint_text = ex.get_constraint_text(),
                )
                response = str(getattr(pred, "response", "") or "")
            except Exception:
                response = ""
            pl = scorer.prompt_loose(example=ex, response=response)
            il = scorer.instruction_loose(example=ex, response=response)
            prompt_loose_scores.append(pl)
            instruction_loose_scores.append(il)
            if verbose:
                console.print(
                    f"  [dim]{ex.key}[/dim] pl={pl:.1f} il={il:.2f} "
                    f"[dim]{ex.instruction_id_list}[/dim]"
                )

    n = len(examples)
    return {
        "prompt_loose":      sum(prompt_loose_scores) / n if n else 0.0,
        "instruction_loose": sum(instruction_loose_scores) / n if n else 0.0,
        "n":                 n,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Results table
# ══════════════════════════════════════════════════════════════════════════════

def render_results_table(
    results: Dict[str, Dict[str, float]],
    title:   str = "IFBench Results — Official Test Set (prompt_loose)",
) -> None:
    tbl = Table(
        title      = title,
        box        = box.DOUBLE_EDGE,
        show_lines = True,
        header_style = "bold white",
    )
    tbl.add_column("Method",           style="bold",       width=26)
    tbl.add_column("Prompt Loose ↑",   style="cyan",       justify="center", width=16)
    tbl.add_column("Instr. Loose ↑",   style="green",      justify="center", width=16)
    tbl.add_column("N",                style="dim",        justify="center", width=6)

    method_styles = {"APA": "cyan", "GEPA": "yellow", "MIPRO": "magenta"}

    best_pl = max((v["prompt_loose"] for v in results.values()), default=0.0)

    for method, metrics in results.items():
        pl = metrics.get("prompt_loose", 0.0)
        il = metrics.get("instruction_loose", 0.0)
        n  = int(metrics.get("n", 0))
        style = method_styles.get(method, "white")

        pl_str = f"[bold green]{pl*100:.1f}%[/bold green]" if pl == best_pl else f"{pl*100:.1f}%"
        il_str = f"{il*100:.1f}%"

        tbl.add_row(
            f"[{style}]{method}[/{style}]",
            pl_str, il_str, str(n),
        )

    console.print(tbl)
    console.print()
    console.print(
        "[dim]prompt_loose = fraction of prompts where ALL constraints pass (loose mode)\n"
        "instr._loose  = mean fraction of individual constraints that pass[/dim]"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(args: argparse.Namespace) -> None:
    random.seed(SEED)
    model      = getattr(args, "model", "mock")
    api_key    = getattr(args, "api_key", None)
    train_size = getattr(args, "train_size", 300)
    val_size   = getattr(args, "val_size", 100)
    verbose    = getattr(args, "verbose", False)
    methods    = [m.lower() for m in args.methods]

    # ── Banner ─────────────────────────────────────────────────────────────
    _model_tag = (
        f"[bold green]{model}[/bold green]" if model != "mock"
        else "[dim]MockLLM[/dim]"
    )
    _gepa_tag  = "[green]dspy.GEPA (official)[/green]"   if _HAS_DSPY else "[red]DSPy not installed[/red]"
    _mipro_tag = "[green]dspy.MIPROv2 (official)[/green]" if _HAS_DSPY else "[red]DSPy not installed[/red]"

    console.print(Panel.fit(
        "[bold bright_white]IFBench Official Evaluation[/bold bright_white]\n\n"
        "[dim]Each method works exactly as in its respective paper.[/dim]\n\n"
        "  [cyan bold]APA[/cyan bold]    — 4-state FSA, evolutionary search (unchanged)\n"
        f"  [yellow bold]GEPA[/yellow bold]   — IFBench 2-stage rewriter  [{_gepa_tag}]\n"
        f"  [magenta bold]MIPRO[/magenta bold]  — IFBench 2-stage rewriter  [{_mipro_tag}]\n\n"
        f"  Model    : {_model_tag}\n"
        f"  Methods  : [green]{', '.join(methods)}[/green]\n"
        f"  Train    : [green]{train_size}[/green]  Val: [green]{val_size}[/green]  "
        f"Test: [green]300[/green] (official IFBench)",
        title="[bold]IFBench — APA vs GEPA vs MIPRO[/bold]",
        border_style="white",
    ))

    # ── Set up LLMs ────────────────────────────────────────────────────────
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if model != "mock" and not key:
        console.print("[red]Error:[/red] OPENAI_API_KEY is required for live models.")
        console.print("Set it with:  export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    if model != "mock":
        llm_apa  = get_llm_api("openai", model=model, api_key=key)
        dspy_lm  = dspy.LM(f"openai/{model}", api_key=key) if _HAS_DSPY else None
    else:
        llm_apa  = get_llm_api("mock", uncertainty_rate=0.32, latency=0.0, seed=SEED)
        dspy_lm  = None

    # ── Load official IFBench data ─────────────────────────────────────────
    scorer = IFBenchOfficialScorer()
    console.print(Panel("[bold]Loading IFBench Data[/bold]", border_style="dim"))

    console.print("  [dim]Loading official test set …[/dim]")
    test_examples = load_ifbench_test()
    console.print(f"  ✓ test set: [green]{len(test_examples)}[/green] prompts")

    train_examples: List[IFBenchOfficialExample] = []
    val_examples:   List[IFBenchOfficialExample] = []

    if ("gepa" in methods or "mipro" in methods) and model != "mock":
        console.print(f"  [dim]Loading IF-RLVR train/val from HuggingFace …[/dim]")
        try:
            train_examples, val_examples = load_ifbench_train_val(
                train_size = train_size,
                val_size   = val_size,
                seed       = SEED,
            )
            console.print(
                f"  ✓ train: [green]{len(train_examples)}[/green]  "
                f"val: [green]{len(val_examples)}[/green]"
            )
        except Exception as exc:
            console.print(f"  [yellow]⚠ HuggingFace load failed:[/yellow] {exc}")
            console.print("  [dim]GEPA and MIPRO will be skipped.[/dim]")
            methods = [m for m in methods if m == "apa"]
    elif ("gepa" in methods or "mipro" in methods) and model == "mock":
        console.print(
            "  [yellow]⚠ GEPA and MIPRO require a real LLM (--model gpt-4.1-mini etc.).[/yellow]\n"
            "  [dim]Running APA only in mock mode.[/dim]"
        )
        methods = [m for m in methods if m == "apa"]

    # ── Train each method ─────────────────────────────────────────────────
    trained: Dict[str, object] = {}   # method → automaton or dspy.Module

    # APA: evolutionary FSA (unchanged)
    if "apa" in methods:
        console.print(Panel(
            "[cyan bold]Training APA — Evolutionary FSA[/cyan bold]",
            border_style="cyan",
        ))
        # For APA training, use IFBench train examples (or subset) as tasks
        apa_tasks = _apa_train_tasks_from_ifbench(
            train_examples[:train_size] if train_examples else []
        )
        if not apa_tasks:
            # Mock mode: use a small synthetic task set for training
            apa_tasks = [
                Task(task_id=f"t{i}", input_text=f"Sample instruction-following task {i}.", expected="")
                for i in range(20)
            ]

        extractor  = FeatureExtractor()
        apa_search = EvolutionarySearch(
            initial_automaton = build_apa_seed(),
            llm_api           = llm_apa,
            feature_extractor = extractor,
            reward_fn         = composite_reward,
            n_generations     = 8,
            mutation_rate     = 0.40,
            elite_frac        = 0.25,
            tournament_size   = 3,
            n_eval_tasks      = min(5, len(apa_tasks)),
            seed              = SEED,
        )
        trained["apa"] = apa_search.run(
            [t.input_text for t in apa_tasks], console=console
        )
        console.print(
            f"  [green]✓[/green] APA trained  "
            f"fitness={apa_search.best_fitness:.4f}"
        )

    # GEPA: official dspy.GEPA with IFBench 2-stage pipeline
    if "gepa" in methods:
        console.print(Panel(
            "[yellow bold]Training GEPA — dspy.GEPA (official IFBench pipeline)[/yellow bold]",
            border_style="yellow",
        ))
        gepa_search = GEPADSPySearch(
            auto           = getattr(args, "gepa_auto", "light"),
            ifbench_scorer = scorer,
            train_examples = train_examples,
            val_examples   = val_examples,
            dspy_lm        = dspy_lm,
            seed           = SEED,
        )
        trained["gepa"] = gepa_search.run(console=console)

    # MIPRO: official dspy.MIPROv2 with IFBench 2-stage pipeline
    if "mipro" in methods:
        console.print(Panel(
            "[magenta bold]Training MIPRO — dspy.MIPROv2 (official IFBench pipeline)[/magenta bold]",
            border_style="magenta",
        ))
        mipro_search = MIPRODSPySearch(
            auto           = getattr(args, "mipro_auto", "light"),
            ifbench_scorer = scorer,
            train_examples = train_examples,
            val_examples   = val_examples,
            dspy_lm        = dspy_lm,
            seed           = SEED,
        )
        trained["mipro"] = mipro_search.run(console=console)

    # ── Evaluate on official test set ──────────────────────────────────────
    console.print(Rule("[bold]Official IFBench Test Evaluation (300 prompts)[/bold]"))
    results: Dict[str, Dict[str, float]] = {}

    if "apa" in trained:
        console.print(Panel("[cyan bold]Evaluating APA on official test set[/cyan bold]", border_style="cyan"))
        results["APA"] = eval_apa_on_ifbench(
            automaton = trained["apa"],
            llm       = llm_apa,
            examples  = test_examples,
            scorer    = scorer,
            desc      = "APA test",
            verbose   = verbose,
        )
        console.print(
            f"  APA  prompt_loose = [bold cyan]{results['APA']['prompt_loose']*100:.1f}%[/bold cyan]  "
            f"instr_loose = {results['APA']['instruction_loose']*100:.1f}%"
        )

    if "gepa" in trained:
        console.print(Panel("[yellow bold]Evaluating GEPA on official test set[/yellow bold]", border_style="yellow"))
        results["GEPA"] = eval_dspy_program_on_ifbench(
            program  = trained["gepa"],
            dspy_lm  = dspy_lm,
            examples = test_examples,
            scorer   = scorer,
            desc     = "GEPA test",
            colour   = "yellow",
            verbose  = verbose,
        )
        console.print(
            f"  GEPA prompt_loose = [bold yellow]{results['GEPA']['prompt_loose']*100:.1f}%[/bold yellow]  "
            f"instr_loose = {results['GEPA']['instruction_loose']*100:.1f}%"
        )

    if "mipro" in trained:
        console.print(Panel("[magenta bold]Evaluating MIPRO on official test set[/magenta bold]", border_style="magenta"))
        results["MIPRO"] = eval_dspy_program_on_ifbench(
            program  = trained["mipro"],
            dspy_lm  = dspy_lm,
            examples = test_examples,
            scorer   = scorer,
            desc     = "MIPRO test",
            colour   = "magenta",
            verbose  = verbose,
        )
        console.print(
            f"  MIPRO prompt_loose = [bold magenta]{results['MIPRO']['prompt_loose']*100:.1f}%[/bold magenta]  "
            f"instr_loose = {results['MIPRO']['instruction_loose']*100:.1f}%"
        )

    # ── Final table ────────────────────────────────────────────────────────
    console.print()
    render_results_table(results)

    # Print optimised instructions for GEPA/MIPRO
    if "gepa" in methods and "gepa" in trained:
        instr = gepa_search.get_optimised_instruction()
        console.print(Panel(
            f"[dim]{instr}[/dim]",
            title="[yellow]GEPA — Evolved Stage-2 Instruction[/yellow]",
            border_style="yellow",
        ))
    if "mipro" in methods and "mipro" in trained:
        instr = mipro_search.get_optimised_instruction()
        console.print(Panel(
            f"[dim]{instr}[/dim]",
            title="[magenta]MIPRO — Optimised Stage-2 Instruction[/magenta]",
            border_style="magenta",
        ))


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="IFBench: APA vs GEPA vs MIPRO on official IFBench benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
────────
  # APA only (mock, offline)
  python ifbench_eval.py --methods apa

  # All methods (live model, default 300 train / 100 val)
  python ifbench_eval.py --model gpt-4.1-mini

  # Custom split sizes
  python ifbench_eval.py --model gpt-4.1-mini --train-size 200 --val-size 100

  # Pass API key inline
  python ifbench_eval.py --model gpt-4.1-mini --api-key sk-...

  # Heavier search
  python ifbench_eval.py --model gpt-4.1-mini --gepa-auto medium --mipro-auto medium

  # Verbose (print per-example constraint results)
  python ifbench_eval.py --model gpt-4.1-mini --verbose

Metrics
───────
  Primary : prompt_loose  — 1.0 if ALL constraints pass (loose), else 0.0
  Secondary: instruction_loose — mean fraction of individual constraints passing
        """,
    )
    p.add_argument(
        "--methods", nargs="+",
        choices=ALL_METHODS, default=ALL_METHODS,
        metavar="METHOD",
        help=f"Methods to evaluate. Choices: {ALL_METHODS}",
    )
    p.add_argument(
        "--model", default="mock",
        help=(
            "LLM to use. 'mock' (default) for offline APA-only. "
            "Any OpenAI model name (e.g. 'gpt-4.1-mini') for live evaluation."
        ),
    )
    p.add_argument(
        "--api-key", dest="api_key", default=None,
        help="OpenAI API key (default: reads OPENAI_API_KEY env var)",
    )
    p.add_argument(
        "--train-size", dest="train_size", type=int, default=300,
        help="IF-RLVR train samples for GEPA/MIPRO (default: 300)",
    )
    p.add_argument(
        "--val-size", dest="val_size", type=int, default=100,
        help="IF-RLVR val samples for GEPA/MIPRO (default: 100)",
    )
    p.add_argument(
        "--gepa-auto", dest="gepa_auto", default="light",
        choices=["light", "medium", "heavy"],
        help="dspy.GEPA search intensity (default: light)",
    )
    p.add_argument(
        "--mipro-auto", dest="mipro_auto", default="light",
        choices=["light", "medium", "heavy"],
        help="dspy.MIPROv2 search intensity (default: light)",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print per-example constraint results during evaluation",
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
