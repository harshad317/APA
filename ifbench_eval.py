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

Methods run sequentially — each method fully completes (train → eval) before
the next one begins.  Within each eval pass, LLM calls are dispatched in
parallel via ThreadPoolExecutor (--workers N, default 4).

All three methods are evaluated on the official IFBench test set (300 prompts)
using prompt_loose accuracy — the primary metric from the IFBench paper.

Usage
─────
  # All methods (requires OPENAI_API_KEY for GEPA and MIPRO)
  python ifbench_eval.py --model gpt-4.1-mini

  # Mock mode (APA only — GEPA/MIPRO need a real LLM)
  python ifbench_eval.py --methods apa

  # Custom split sizes + parallelism
  python ifbench_eval.py --model gpt-4.1-mini --train-size 200 --val-size 100 --workers 8

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Callable, Dict, List, Optional

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
    from dspy.utils.callback import BaseCallback
    from adaptive_prompt_automaton.search.gepa_dspy import (
        GEPADSPySearch,
        generate_stage1_drafts,
    )
    from adaptive_prompt_automaton.search.mipro_dspy import MIPRODSPySearch
    _HAS_DSPY = True
except ImportError:
    _HAS_DSPY = False

console    = Console(width=160)
SEED       = 42
ALL_METHODS = ["apa", "gepa", "mipro"]


if _HAS_DSPY:
    class DSPyAPICallCounter(BaseCallback):
        """Thread-safe counter for DSPy LM invocations."""

        def __init__(self) -> None:
            self._lock = Lock()
            self.count = 0

        def on_lm_start(self, call_id: str, instance: object, inputs: dict) -> None:
            with self._lock:
                self.count += 1
else:
    class DSPyAPICallCounter:
        count = 0


def _call_count(llm: object) -> int:
    """Best-effort API-call counter for local LLM wrappers and DSPy counters."""
    return int(getattr(llm, "call_count", 0) or 0)


def _counter_count(counter: object) -> int:
    return int(getattr(counter, "count", 0) or 0)


# ══════════════════════════════════════════════════════════════════════════════
# Fallback probe tasks + proxy fingerprint
# Used when HuggingFace `datasets` is unavailable and train_examples is empty.
# These hardcoded prompts cover the major IFBench constraint categories
# (keyword inclusion, exact word-count, format/list, sentence structure,
#  enumeration, style, brevity, multi-attribute) so the fingerprint vector
# discriminates behaviourally distinct FSA variants even without real data.
# ══════════════════════════════════════════════════════════════════════════════

_FALLBACK_PROBE_TASKS: List[str] = [
    # keyword inclusion
    "Write a haiku about space exploration. It must contain the word 'cosmos'.",
    # exact word-count constraint
    "In exactly 50 words, describe the water cycle. Do not use the word 'rain'.",
    # enumeration + capitalisation
    "List exactly 3 benefits of regular exercise. Each item must start with a capital letter.",
    # sentence-count + punctuation
    "Write a two-sentence explanation of photosynthesis. End each sentence with a period.",
    # adjective count + format
    "Describe a sunset using exactly 5 adjectives. Present them as a numbered list.",
    # structure (setup/punchline)
    "Write a joke about programming. It must have a clearly labelled setup and punchline.",
    # brevity constraint + register
    "In fewer than 30 words, explain why the sky is blue. Use simple everyday language.",
    # multi-attribute product description
    "Write a product description for a pencil. It must mention its color, length, and primary use.",
]


def _make_proxy_fingerprint_fn() -> Callable[[str, str], float]:
    """
    Path-independent quality fingerprint used when no official IFBench scorer
    is available (e.g., ``datasets`` not installed).

    Returns 1.0 when the response is substantive (10–200 words), ends with
    terminal punctuation, and contains no hedging language; 0.0 otherwise.
    The signal is completely independent of the FSA routing path taken, so it
    cannot reward routing artefacts — only output quality.
    """
    _HEDGE_PHRASES = (
        "not sure", "uncertain", "possibly", "perhaps",
        "i don't know", "i'm not certain", "i cannot be sure",
        "it depends", "i'm unsure",
    )

    def fingerprint_fn(task_input: str, response: str) -> float:
        words   = response.split()
        lower   = response.lower()
        hedge   = any(ph in lower for ph in _HEDGE_PHRASES)
        length_ok = 10 <= len(words) <= 200
        ends_ok   = response.rstrip().endswith((".", "!", "?"))
        return float(length_ok and ends_ok and not hedge)

    return fingerprint_fn


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
        # Route through decompose when the initial answer is uncertain (confidence
        # < 0.80 means the model used hedging language).  This guard fires on LLM
        # *output* features, not on static input length, so routing varies across
        # episodes and PathEntropy stays > 0.
        #
        # Previous guard was `input_length > 0.12` (priority 2), which fired for
        # every IFBench prompt (all > 60 words) → every episode took the same
        # start→decompose→verify→terminal path → PathEntropy = 0.000 forever.
        TransitionConfig(
            source_state="start", target_state="decompose",
            guard_type="threshold", feature_name="answer_confidence",
            operator="<", threshold=0.80, priority=2,
        ),
        TransitionConfig(
            source_state="start", target_state="verify",
            guard_type="always", operator="always", priority=1,
        ),
        TransitionConfig(
            source_state="decompose", target_state="verify",
            guard_type="always", operator="always", priority=1,
        ),
        TransitionConfig(
            source_state="verify", target_state="terminal",
            guard_type="always", operator="always", priority=1,
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

def _build_stratified_probe_set(
    examples: List[IFBenchOfficialExample],
    n: int = 20,
    seed: int = 42,
) -> List[IFBenchOfficialExample]:
    """
    Select a fixed probe set that maximally covers IFBench constraint categories.

    Fix 2: stratify by the primary constraint category (first token of the first
    instruction_id, e.g. 'count', 'format', 'keyword') so the probe set covers
    the latent skill space rather than being a uniform random sample.  A uniform
    sample risks selecting tasks that are highly correlated in what they require,
    making fingerprints indistinguishable.

    Fix 4: the returned set is fixed for the entire training run — no refresh —
    so fingerprint vectors computed in generation 1 are directly comparable to
    those in generation 8.

    Parameters
    ----------
    examples : pool to sample from (train set preferred; no test-set leakage)
    n        : target probe set size (default matches --apa-eval-tasks)
    seed     : fixed seed for reproducibility

    Returns
    -------
    List of up to n IFBenchOfficialExample objects, stratified by constraint cat.
    """
    rng = random.Random(seed)

    # Group examples by primary constraint category
    groups: Dict[str, List[IFBenchOfficialExample]] = {}
    for ex in examples:
        if ex.instruction_id_list:
            raw_id = ex.instruction_id_list[0]
            cat    = raw_id.split(":")[0] if ":" in raw_id else raw_id
        else:
            cat = "uncategorised"
        groups.setdefault(cat, []).append(ex)

    cats     = sorted(groups.keys())
    per_cat  = max(1, n // max(len(cats), 1))

    selected: List[IFBenchOfficialExample] = []
    seen_keys: set = set()

    # Proportional sampling from each category
    for cat in cats:
        members = list(groups[cat])
        rng.shuffle(members)
        for ex in members:
            if ex.key not in seen_keys and len(selected) < n:
                selected.append(ex)
                seen_keys.add(ex.key)
                if len(selected) - sum(1 for e in selected if
                        (e.instruction_id_list[0].split(":")[0]
                         if e.instruction_id_list else "x") == cat) >= per_cat:
                    break

    # Fill remainder if categories ran short
    if len(selected) < n:
        remaining = [ex for ex in examples if ex.key not in seen_keys]
        rng.shuffle(remaining)
        for ex in remaining:
            if len(selected) >= n:
                break
            selected.append(ex)
            seen_keys.add(ex.key)

    return selected[:n]


def _make_fingerprint_fn(
    scorer:         "IFBenchOfficialScorer",
    probe_examples: List[IFBenchOfficialExample],
) -> Callable[[str, str], float]:
    """
    Create the fingerprint function passed to EvolutionarySearch.

    Fix 5: fingerprint values are derived from the same episode evaluation that
    computes fitness — zero additional API calls.  Each probe task contributes
    one float (instruction_loose ∈ [0, 1]) to the fingerprint vector.

    Using instruction_loose (continuous) rather than prompt_loose (binary) gives
    a more discriminating signal: two prompts that both fail a task may still
    have different constraint-level pass rates, yielding different fingerprints
    and landing in different clusters.

    Parameters
    ----------
    scorer         : IFBenchOfficialScorer instance
    probe_examples : the fixed probe set (same list passed to EvolutionarySearch)

    Returns
    -------
    fingerprint_fn(task_input, response) → float ∈ [0, 1]
    """
    prompt_to_example: Dict[str, IFBenchOfficialExample] = {
        ex.prompt: ex for ex in probe_examples
    }

    def fingerprint_fn(task_input: str, response: str) -> float:
        ex = prompt_to_example.get(task_input)
        if ex is None:
            return 0.0
        return scorer.instruction_loose(ex, response)

    return fingerprint_fn


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


def _make_ifbench_episode_reward(
    scorer: IFBenchOfficialScorer,
    examples: List[IFBenchOfficialExample],
) -> Callable[[Episode], float]:
    """
    Reward APA training with the same official IFBench verifier used at test time.

    prompt_loose is binary and sparse, so instruction_loose supplies partial credit
    when only some constraints pass. A small composite_reward tie-breaker keeps
    malformed/empty outputs below useful outputs when verifier scores tie.
    """
    by_prompt = {ex.prompt: ex for ex in examples}

    def reward(episode: Episode) -> float:
        ex = by_prompt.get(episode.task_input)
        if ex is None:
            return composite_reward(episode)

        response = episode.final_output or ""
        passed = scorer.per_instruction(ex, response)
        instr_score = sum(passed) / len(passed) if passed else 0.0
        prompt_score = 1.0 if passed and all(passed) else 0.0
        tie_break = max(0.0, composite_reward(episode)) * 0.05
        return min(1.0, 0.80 * prompt_score + 0.15 * instr_score + tie_break)

    return reward


# ══════════════════════════════════════════════════════════════════════════════
# APA evaluation on official IFBench test set  (parallel workers)
# ══════════════════════════════════════════════════════════════════════════════

def eval_apa_on_ifbench(
    automaton: Automaton,
    llm:       object,
    examples:  List[IFBenchOfficialExample],
    scorer:    IFBenchOfficialScorer,
    desc:      str  = "APA eval",
    workers:   int  = 4,
    verbose:   bool = False,
) -> Dict[str, float]:
    """
    Run APA automaton on official IFBench test examples with parallel workers.
    Returns dict with prompt_loose, instruction_loose, n.
    """
    extractor = FeatureExtractor()

    def _eval_one(ex: IFBenchOfficialExample):
        # Each thread gets its own executor instance (thread-safe: no shared
        # mutable state; automaton / llm / feature_extractor are read-only)
        _exec = AutomatonExecutor(
            automaton         = automaton,
            llm_api           = llm,
            feature_extractor = extractor,
        )
        episode  = _exec.run_episode(ex.prompt, verbose=False)
        response = episode.final_output or ""
        pl = scorer.prompt_loose(example=ex, response=response)
        il = scorer.instruction_loose(example=ex, response=response)
        return ex.key, pl, il

    prompt_loose_scores      = []
    instruction_loose_scores = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_eval_one, ex): ex for ex in examples}
        bar = tqdm(
            as_completed(futures),
            total  = len(examples),
            desc   = f"  {desc}",
            colour = "cyan",
            leave  = False,
        )
        for fut in bar:
            key, pl, il = fut.result()
            prompt_loose_scores.append(pl)
            instruction_loose_scores.append(il)
            if verbose:
                ex = futures[fut]
                console.print(
                    f"  [dim]{key}[/dim] pl={pl:.1f} il={il:.2f} "
                    f"[dim]{ex.instruction_id_list}[/dim]"
                )

    n = len(examples)
    return {
        "prompt_loose":      sum(prompt_loose_scores) / n if n else 0.0,
        "instruction_loose": sum(instruction_loose_scores) / n if n else 0.0,
        "n":                 n,
    }


# ══════════════════════════════════════════════════════════════════════════════
# GEPA / MIPRO evaluation on official IFBench test set  (parallel workers)
# ══════════════════════════════════════════════════════════════════════════════

def eval_dspy_program_on_ifbench(
    program:  object,
    dspy_lm:  object,
    examples: List[IFBenchOfficialExample],
    scorer:   IFBenchOfficialScorer,
    desc:     str  = "DSPy eval",
    colour:   str  = "magenta",
    workers:  int  = 4,
    verbose:  bool = False,
) -> Dict[str, float]:
    """
    Run a compiled IFBenchRewriterProgram on official IFBench test examples.
    Stage-1 drafts are generated in parallel; stage-2 inference is also
    parallelised with ThreadPoolExecutor(workers).
    """
    console.print(
        f"  [dim]Stage 1 drafts for {len(examples)} examples "
        f"(workers={workers}) …[/dim]"
    )
    drafts = generate_stage1_drafts(examples, dspy_lm, workers=workers)

    prompt_loose_scores      = []
    instruction_loose_scores = []

    # ensure DSPy global LM is set before threads inherit context
    dspy.configure(lm=dspy_lm)

    def _eval_one(ex: IFBenchOfficialExample, draft: str):
        with dspy.context(lm=dspy_lm):
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
        return ex.key, pl, il

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_eval_one, ex, draft): ex
            for ex, draft in zip(examples, drafts)
        }
        bar = tqdm(
            as_completed(futures),
            total  = len(examples),
            desc   = f"  {desc}",
            colour = colour,
            leave  = False,
        )
        for fut in bar:
            key, pl, il = fut.result()
            prompt_loose_scores.append(pl)
            instruction_loose_scores.append(il)
            if verbose:
                ex = futures[fut]
                console.print(
                    f"  [dim]{key}[/dim] pl={pl:.1f} il={il:.2f} "
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
        title        = title,
        box          = box.DOUBLE_EDGE,
        show_lines   = True,
        header_style = "bold white",
    )
    tbl.add_column("Method",          style="bold",  width=26)
    tbl.add_column("Prompt Loose ↑",  style="cyan",  justify="center", width=16)
    tbl.add_column("Instr. Loose ↑",  style="green", justify="center", width=16)
    tbl.add_column("Eval N",          style="dim",   justify="center", width=8)
    tbl.add_column("API Calls",       style="yellow", justify="right", width=12)

    method_styles = {"APA": "cyan", "GEPA": "yellow", "MIPRO": "magenta"}
    best_pl = max((v["prompt_loose"] for v in results.values()), default=0.0)

    for method, metrics in results.items():
        pl    = metrics.get("prompt_loose", 0.0)
        il    = metrics.get("instruction_loose", 0.0)
        n     = int(metrics.get("n", 0))
        calls = int(metrics.get("api_calls", 0))
        style = method_styles.get(method, "white")
        pl_str = (
            f"[bold green]{pl*100:.1f}%[/bold green]"
            if pl == best_pl else f"{pl*100:.1f}%"
        )
        tbl.add_row(
            f"[{style}]{method}[/{style}]",
            pl_str, f"{il*100:.1f}%", str(n), f"{calls:,}",
        )

    console.print(tbl)
    console.print()
    console.print(
        "[dim]prompt_loose = fraction of prompts where ALL constraints pass (loose mode)\n"
        "instr._loose  = mean fraction of individual constraints that pass\n"
        "eval_n        = number of official IFBench test prompts evaluated\n"
        "api_calls     = live LM calls made during that method's train + eval pipeline[/dim]"
    )


def _print_partial(method: str, metrics: Dict[str, float], style: str = "white") -> None:
    """Print a one-line result summary immediately after a method finishes."""
    pl = metrics["prompt_loose"] * 100
    il = metrics["instruction_loose"] * 100
    console.print(
        f"  [{style}]{method}[/{style}]  "
        f"prompt_loose = [bold {style}]{pl:.1f}%[/bold {style}]  "
        f"instr_loose = {il:.1f}%  "
        f"[dim](eval_n={int(metrics['n'])}, api_calls={int(metrics.get('api_calls', 0)):,})[/dim]"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Per-method train + eval  (called sequentially from main)
# ══════════════════════════════════════════════════════════════════════════════

def run_apa(
    args,
    llm_apa,
    train_examples: List[IFBenchOfficialExample],
    test_examples:  List[IFBenchOfficialExample],
    scorer:         IFBenchOfficialScorer,
) -> Dict[str, float]:
    """Train APA then evaluate on test set.  Returns metrics dict."""
    api_calls_before = _call_count(llm_apa)
    console.print(Panel(
        "[cyan bold]APA — Evolutionary FSA[/cyan bold]\n"
        "[dim]Training then evaluating on official test set.[/dim]",
        border_style="cyan",
    ))

    apa_tasks = _apa_train_tasks_from_ifbench(train_examples)
    if not apa_tasks:
        apa_tasks = [
            Task(task_id=f"t{i}", input_text=f"Sample instruction-following task {i}.", expected="")
            for i in range(20)
        ]

    extractor  = FeatureExtractor()
    reward_fn = (
        _make_ifbench_episode_reward(scorer, train_examples)
        if train_examples else composite_reward
    )
    reward_label = "official IFBench verifier" if train_examples else "composite proxy"
    console.print(f"  [dim]APA reward:[/dim] {reward_label}")

    # ── Build fixed stratified probe set (Fixes 2 & 4) ───────────────────
    # Probe set is selected once here and held constant for the entire run.
    # Size = apa_eval_tasks so API cost is identical to the previous random-
    # sample approach — fingerprinting is free (Fix 5).
    apa_eval_tasks_n = min(getattr(args, "apa_eval_tasks", 5), len(apa_tasks))
    probe_pool = train_examples if train_examples else []
    probe_examples: List[IFBenchOfficialExample] = []
    fp_fn = None
    if probe_pool and apa_eval_tasks_n > 0:
        probe_examples = _build_stratified_probe_set(
            probe_pool,
            n    = apa_eval_tasks_n,
            seed = SEED,
        )
        fp_fn = _make_fingerprint_fn(scorer, probe_examples)
        console.print(
            f"  [dim]Probe set: {len(probe_examples)} stratified tasks "
            f"(fixed for full run — Fixes 2/4/5)[/dim]"
        )

    probe_task_inputs = [ex.prompt for ex in probe_examples] if probe_examples else None

    # ── Fallback fingerprinting when no HuggingFace train data ───────────────
    # When datasets is not installed, probe_pool is empty → probe_examples is
    # empty → probe_task_inputs is None and fp_fn is None → EvolutionarySearch
    # would print "Fingerprinting: disabled" and skip all diversity mechanics.
    #
    # Instead: activate fingerprinting with 8 hardcoded IFBench-style tasks and
    # a path-independent proxy quality signal.  This is weaker than the official
    # IFBench verifier but still provides meaningful diversity pressure and keeps
    # the panel from misreporting "disabled".
    if probe_task_inputs is None:
        probe_task_inputs = _FALLBACK_PROBE_TASKS
        fp_fn             = _make_proxy_fingerprint_fn()
        console.print(
            f"  [yellow]⚠[/yellow]  No HuggingFace train data — "
            f"activating fallback fingerprinting "
            f"({len(_FALLBACK_PROBE_TASKS)} hardcoded proxy tasks, proxy quality signal).\n"
            f"  Install [cyan]datasets[/cyan] for official IFBench fingerprinting."
        )

    apa_search = EvolutionarySearch(
        initial_automaton    = build_apa_seed(),
        llm_api              = llm_apa,
        feature_extractor    = extractor,
        reward_fn            = reward_fn,
        n_generations        = 8,
        mutation_rate        = 0.40,
        elite_frac           = 0.25,
        tournament_size      = 3,
        n_eval_tasks         = apa_eval_tasks_n,
        seed                 = SEED,
        # Diversity regularisation (Fixes 1–5)
        probe_tasks          = probe_task_inputs,
        fingerprint_fn       = fp_fn,
        diversity_lambda     = 0.10,
        diversity_threshold  = 0.15,
        diversity_quota      = 1,
        # Parallelism — now active during training (was only in eval before)
        workers              = getattr(args, "workers", 1),
    )
    best_automaton = apa_search.run(
        [t.input_text for t in apa_tasks], console=console
    )
    console.print(
        f"  [green]✓[/green] APA trained  "
        f"fitness=[cyan]{apa_search.best_fitness:.4f}[/cyan]"
    )

    console.rule("[cyan]APA — Test Evaluation[/cyan]")
    metrics = eval_apa_on_ifbench(
        automaton = best_automaton,
        llm       = llm_apa,
        examples  = test_examples,
        scorer    = scorer,
        desc      = "APA test",
        workers   = getattr(args, "workers", 4),
        verbose   = getattr(args, "verbose", False),
    )
    metrics["api_calls"] = _call_count(llm_apa) - api_calls_before
    _print_partial("APA", metrics, "cyan")
    return metrics


def run_gepa(
    args,
    dspy_lm,
    train_examples: List[IFBenchOfficialExample],
    val_examples:   List[IFBenchOfficialExample],
    test_examples:  List[IFBenchOfficialExample],
    scorer:         IFBenchOfficialScorer,
) -> tuple:
    """Train GEPA then evaluate on test set.  Returns (metrics, gepa_search)."""
    call_counter = getattr(args, "dspy_call_counter", None)
    api_calls_before = _counter_count(call_counter)
    console.print(Panel(
        "[yellow bold]GEPA — dspy.GEPA (official IFBench pipeline)[/yellow bold]\n"
        "[dim]Training then evaluating on official test set.[/dim]",
        border_style="yellow",
    ))

    gepa_search = GEPADSPySearch(
        auto           = getattr(args, "gepa_auto", "light"),
        ifbench_scorer = scorer,
        train_examples = train_examples,
        val_examples   = val_examples,
        dspy_lm        = dspy_lm,
        seed           = SEED,
        workers        = getattr(args, "workers", 4),
    )
    trained_program = gepa_search.run(console=console)

    console.rule("[yellow]GEPA — Test Evaluation[/yellow]")
    metrics = eval_dspy_program_on_ifbench(
        program  = trained_program,
        dspy_lm  = dspy_lm,
        examples = test_examples,
        scorer   = scorer,
        desc     = "GEPA test",
        colour   = "yellow",
        workers  = getattr(args, "workers", 4),
        verbose  = getattr(args, "verbose", False),
    )
    metrics["api_calls"] = _counter_count(call_counter) - api_calls_before
    _print_partial("GEPA", metrics, "yellow")
    return metrics, gepa_search


def run_mipro(
    args,
    dspy_lm,
    train_examples: List[IFBenchOfficialExample],
    val_examples:   List[IFBenchOfficialExample],
    test_examples:  List[IFBenchOfficialExample],
    scorer:         IFBenchOfficialScorer,
) -> tuple:
    """Train MIPRO then evaluate on test set.  Returns (metrics, mipro_search)."""
    call_counter = getattr(args, "dspy_call_counter", None)
    api_calls_before = _counter_count(call_counter)
    console.print(Panel(
        "[magenta bold]MIPRO — dspy.MIPROv2 (official IFBench pipeline)[/magenta bold]\n"
        "[dim]Training then evaluating on official test set.[/dim]",
        border_style="magenta",
    ))

    mipro_search = MIPRODSPySearch(
        auto           = getattr(args, "mipro_auto", "light"),
        ifbench_scorer = scorer,
        train_examples = train_examples,
        val_examples   = val_examples,
        dspy_lm        = dspy_lm,
        seed           = SEED,
        workers        = getattr(args, "workers", 4),
    )
    trained_program = mipro_search.run(console=console)

    console.rule("[magenta]MIPRO — Test Evaluation[/magenta]")
    metrics = eval_dspy_program_on_ifbench(
        program  = trained_program,
        dspy_lm  = dspy_lm,
        examples = test_examples,
        scorer   = scorer,
        desc     = "MIPRO test",
        colour   = "magenta",
        workers  = getattr(args, "workers", 4),
        verbose  = getattr(args, "verbose", False),
    )
    metrics["api_calls"] = _counter_count(call_counter) - api_calls_before
    _print_partial("MIPRO", metrics, "magenta")
    return metrics, mipro_search


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(args: argparse.Namespace) -> None:
    random.seed(SEED)
    model      = getattr(args, "model", "mock")
    api_key    = getattr(args, "api_key", None)
    train_size = getattr(args, "train_size", 300)
    val_size   = getattr(args, "val_size", 100)
    workers    = getattr(args, "workers", 4)
    apa_eval_tasks = getattr(args, "apa_eval_tasks", 5)
    methods    = [m.lower() for m in args.methods]

    # ── Banner ─────────────────────────────────────────────────────────────
    _model_tag = (
        f"[bold green]{model}[/bold green]" if model != "mock"
        else "[dim]MockLLM[/dim]"
    )
    _gepa_tag  = "[green]dspy.GEPA (official)[/green]"    if _HAS_DSPY else "[red]DSPy not installed[/red]"
    _mipro_tag = "[green]dspy.MIPROv2 (official)[/green]" if _HAS_DSPY else "[red]DSPy not installed[/red]"

    console.print(Panel.fit(
        "[bold bright_white]IFBench Official Evaluation[/bold bright_white]\n\n"
        "[dim]Each method works exactly as in its respective paper.\n"
        "Methods run sequentially — each finishes train + eval before the next starts.[/dim]\n\n"
        "  [cyan bold]APA[/cyan bold]    — 4-state FSA, evolutionary search (unchanged)\n"
        f"  [yellow bold]GEPA[/yellow bold]   — IFBench 2-stage rewriter  [{_gepa_tag}]\n"
        f"  [magenta bold]MIPRO[/magenta bold]  — IFBench 2-stage rewriter  [{_mipro_tag}]\n\n"
        f"  Model    : {_model_tag}\n"
        f"  Methods  : [green]{', '.join(methods)}[/green]\n"
        f"  Workers  : [green]{workers}[/green] (parallel LLM calls per eval pass)\n"
        f"  APA eval tasks: [green]{apa_eval_tasks}[/green] per fitness evaluation\n"
        f"  DSPy cache: [green]disabled[/green] for GEPA/MIPRO\n"
        f"  Train    : [green]{train_size}[/green]  Val: [green]{val_size}[/green]  "
        f"Test: [green]300[/green] (official IFBench)",
        title="[bold]IFBench — APA vs GEPA vs MIPRO[/bold]",
        border_style="white",
    ))

    # ── Validate API key ───────────────────────────────────────────────────
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if model != "mock" and not key:
        console.print("[red]Error:[/red] OPENAI_API_KEY is required for live models.")
        console.print("Set it with:  export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    # ── Set up LLMs ────────────────────────────────────────────────────────
    if model != "mock":
        llm_apa = get_llm_api("openai", model=model, api_key=key)
        dspy_counter = DSPyAPICallCounter() if _HAS_DSPY else None
        dspy_lm = (
            dspy.LM(
                f"openai/{model}",
                api_key=key,
                cache=False,
                callbacks=[dspy_counter],
            )
            if _HAS_DSPY else None
        )
        setattr(args, "dspy_call_counter", dspy_counter)
    else:
        llm_apa = get_llm_api("mock", uncertainty_rate=0.32, latency=0.0, seed=SEED)
        dspy_lm = None
        setattr(args, "dspy_call_counter", None)

    # ── Load official IFBench data ─────────────────────────────────────────
    scorer = IFBenchOfficialScorer()
    console.print(Panel("[bold]Loading IFBench Data[/bold]", border_style="dim"))

    console.print("  [dim]Loading official test set …[/dim]")
    test_examples = load_ifbench_test()
    console.print(f"  ✓ test set: [green]{len(test_examples)}[/green] prompts")

    train_examples: List[IFBenchOfficialExample] = []
    val_examples:   List[IFBenchOfficialExample] = []

    need_hf = ("gepa" in methods or "mipro" in methods) and model != "mock"
    if need_hf:
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

    # ── Run each method sequentially: train then eval before moving on ─────
    results:        Dict[str, Dict[str, float]] = {}
    gepa_search_obj  = None
    mipro_search_obj = None

    console.rule("[bold white]Starting Evaluation — Sequential Method Pipeline[/bold white]")

    for method in methods:
        console.rule()

        if method == "apa":
            results["APA"] = run_apa(
                args, llm_apa, train_examples, test_examples, scorer
            )

        elif method == "gepa":
            results["GEPA"], gepa_search_obj = run_gepa(
                args, dspy_lm, train_examples, val_examples, test_examples, scorer
            )

        elif method == "mipro":
            results["MIPRO"], mipro_search_obj = run_mipro(
                args, dspy_lm, train_examples, val_examples, test_examples, scorer
            )

    # ── Final comparison table ─────────────────────────────────────────────
    console.print()
    console.rule("[bold white]Final Results[/bold white]")
    render_results_table(results)

    # ── Print optimised instructions (GEPA / MIPRO) ────────────────────────
    if gepa_search_obj is not None:
        instr = gepa_search_obj.get_optimised_instruction()
        console.print(Panel(
            f"[dim]{instr}[/dim]",
            title="[yellow]GEPA — Evolved Stage-2 Instruction[/yellow]",
            border_style="yellow",
        ))
    if mipro_search_obj is not None:
        instr = mipro_search_obj.get_optimised_instruction()
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

  # All methods, 8 parallel workers
  python ifbench_eval.py --model gpt-4.1-mini --workers 8

  # Custom split sizes
  python ifbench_eval.py --model gpt-4.1-mini --train-size 200 --val-size 100

  # Give APA a less noisy IFBench training signal
  python ifbench_eval.py --model gpt-4.1-mini --apa-eval-tasks 20

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

Sequential execution
────────────────────
  Each method runs to completion (train → eval) before the next begins.
  Within each eval pass, LLM calls are parallelised via --workers.
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
        "--workers", type=int, default=4,
        help="Parallel LLM workers for eval passes (default: 4)",
    )
    p.add_argument(
        "--apa-eval-tasks", dest="apa_eval_tasks", type=int, default=5,
        help=(
            "Training prompts sampled per APA fitness evaluation. Higher values "
            "give a less noisy IFBench reward but use more API calls (default: 5)."
        ),
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
