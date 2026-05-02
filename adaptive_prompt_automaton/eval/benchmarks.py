"""
eval/benchmarks.py
──────────────────
Benchmark task suites and reward functions for the Adaptive Prompt Automaton.

Suites
──────
  make_qa_benchmark()               — mixed QA (easy / medium / hard-long)
  make_distribution_shift_benchmark() — paired (clean train, shifted test) inputs
  make_perturbation_benchmark()     — paraphrase/noise augmented tasks

Reward
──────
  composite_reward(episode, ...)    — task_score + structure_bonus
                                       - token_penalty - uncertainty_penalty
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..core.executor import Episode


# ──────────────────────────────────────────────────────────────────────────────
# Task data-class
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Task:
    task_id:    str
    input_text: str
    expected:   str           = ""
    category:   str           = "qa"
    difficulty: str           = "medium"   # "easy" | "medium" | "hard"
    tags:       List[str]     = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# BenchmarkSuite
# ──────────────────────────────────────────────────────────────────────────────

class BenchmarkSuite:
    """Container for a collection of Task objects."""

    def __init__(self, name: str):
        self.name  = name
        self.tasks: List[Task] = []

    def add(self, task: Task) -> None:
        self.tasks.append(task)

    def sample(
        self,
        n:          int,
        difficulty: Optional[str] = None,
        seed:       Optional[int] = None,
    ) -> List[Task]:
        rng  = random.Random(seed)
        pool = [t for t in self.tasks if difficulty is None or t.difficulty == difficulty]
        return rng.sample(pool, min(n, len(pool)))

    def by_difficulty(self, difficulty: str) -> List[Task]:
        return [t for t in self.tasks if t.difficulty == difficulty]

    def inputs(self) -> List[str]:
        return [t.input_text for t in self.tasks]

    def __len__(self) -> int:
        return len(self.tasks)

    def __repr__(self) -> str:
        return f"BenchmarkSuite({self.name!r}, n={len(self.tasks)})"


# ──────────────────────────────────────────────────────────────────────────────
# Reward function
# ──────────────────────────────────────────────────────────────────────────────

def composite_reward(
    episode:             Episode,
    token_budget:        int   = 1500,
    uncertainty_penalty: float = 0.15,
    completeness_bonus:  float = 0.15,
    terminal_bonus:      float = 0.20,
    multi_step_bonus:    float = 0.05,
    expected:            str   = "",
    lexical_bonus:       float = 0.10,
) -> float:
    """
    Composite reward that measures output quality independently of the route
    taken through the automaton.

    Previous versions awarded bonuses for "step 1", "decompos", "verified", etc.
    — strings that MockLLM generates specifically when routed through decompose/
    verify states.  That caused the optimizer to learn to route, not to answer,
    producing circular self-reward.

    This version uses path-independent quality signals:

      + base score          (0.5 for ≥10 words — substantive response)
      + completeness_bonus  if output ends with a complete sentence and has ≥20 words
      + lexical_bonus       if expected is provided and output overlaps ≥ 1 key term
      + terminal_bonus      if episode reached a terminal state cleanly
      + multi_step_bonus    (small, 0.05) if adaptive routing fired at all
      - uncertainty_penalty if output hedges heavily
      - token_penalty       proportional to token over-use
    """
    out   = episode.final_output
    lower = out.lower()
    words = out.split()
    score = 0.0

    # Base reward for a substantive answer (at least 10 words)
    if len(words) >= 10:
        score += 0.5

    # Penalise excessive hedging — path-independent lexical signal
    hedge_words = ["not sure", "uncertain", "don't know", "possibly", "perhaps",
                   "not entirely certain", "i'm not confident"]
    n_hedges = sum(1 for w in hedge_words if w in lower)
    if n_hedges >= 2:
        score -= uncertainty_penalty
    elif n_hedges == 1:
        score -= uncertainty_penalty * 0.5

    # Completeness bonus: response has ≥20 words and ends with sentence-ending punctuation.
    # This rewards thorough, well-formed answers regardless of which state produced them.
    if len(words) >= 20 and out.rstrip().endswith((".", "!", "?")):
        score += completeness_bonus

    # Lexical overlap bonus: when an expected answer is provided, reward responses
    # that contain at least one key term from the expected answer.  This is a weak
    # factual-correctness proxy that does not depend on routing markers.
    if expected:
        exp_tokens = set(expected.lower().split())
        out_tokens = set(lower.split())
        if exp_tokens & out_tokens:
            score += lexical_bonus

    # Terminal-state bonus (rewards clean FSA termination, not specific route)
    if episode.terminated_by == "terminal_state":
        score += terminal_bonus

    # Small multi-step bonus — legitimately rewards adaptivity, but kept small (0.05)
    # so it cannot dominate the reward signal.
    if len(episode.path) > 1:
        score += multi_step_bonus

    # Token cost penalty
    over  = max(0, episode.total_tokens - token_budget)
    t_pen = min(over / token_budget, 0.25)
    score -= t_pen

    return max(-1.0, min(1.0, score))


# Alias for backward compatibility
simple_reward_fn = composite_reward


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark builders
# ──────────────────────────────────────────────────────────────────────────────

def make_qa_benchmark() -> BenchmarkSuite:
    """
    Mixed QA benchmark with three difficulty tiers.
    - easy  : short factual questions
    - medium: multi-sentence conceptual questions
    - hard  : long, multi-paragraph questions requiring decomposition
    """
    suite = BenchmarkSuite("QA-Mixed")

    easy = [
        ("What is the capital of France?",                 "Paris"),
        ("What is 2 + 2?",                                 "4"),
        ("What colour is the sky?",                        "blue"),
        ("How many days in a week?",                       "7"),
        ("What molecule is water?",                        "H2O"),
        ("Who wrote Romeo and Juliet?",                    "Shakespeare"),
        ("What planet is closest to the Sun?",             "Mercury"),
        ("What is the speed of light (approx.)?",          "3×10^8 m/s"),
    ]

    medium = [
        ("Explain the difference between supervised and unsupervised learning.",
         "supervised uses labelled data"),
        ("What is the time complexity of binary search?",          "O(log n)"),
        ("Describe how photosynthesis works.",                      "converts sunlight"),
        ("What are Newton's three laws of motion?",                 "inertia, F=ma, reaction"),
        ("How does a neural network learn via backpropagation?",    "gradient descent"),
        ("What is entropy in thermodynamics?",                      "measure of disorder"),
        ("Explain the CAP theorem in distributed systems.",         "consistency, availability, partition"),
        ("What is the difference between RNA and DNA?",             "single vs double strand"),
    ]

    _long_prefix = (
        "Given an extensive scientific document covering the following topics in depth: "
    )
    hard = [
        (
            _long_prefix +
            "rising global temperatures, accelerating ice-cap melt, sea-level rise projections "
            "through 2100, ecosystem disruption, species extinction rates, ocean acidification, "
            "permafrost methane release, and the socioeconomic consequences for coastal and "
            "low-lying nations — what are the three highest-priority mitigation strategies "
            "and what evidence supports prioritising each one over the alternatives?",
            "carbon capture, renewables, policy"
        ),
        (
            _long_prefix +
            "the history of machine learning from perceptrons in the 1950s through "
            "convolutional networks, recurrent networks, attention mechanisms, and the "
            "transformer architecture — explain the core theoretical insight behind "
            "the attention mechanism, why it outperforms recurrence for long sequences, "
            "and what its computational complexity bottleneck is.",
            "attention, O(n^2), parallelism"
        ),
        (
            "A train leaves station A at 9 am travelling at 60 mph toward station B, "
            "which is 240 miles away. Another train leaves station B at 10 am travelling "
            "toward station A at 80 mph. After they pass each other, each continues to the "
            "opposite terminal, waits 30 minutes, then returns. "
            "At what time do they meet on the return journey? "
            "Show your work clearly and state any assumptions. " * 2,
            "detailed calculation"
        ),
    ]

    for i, (inp, exp) in enumerate(easy):
        suite.add(Task(f"easy_{i}", inp, exp, "qa", "easy"))
    for i, (inp, exp) in enumerate(medium):
        suite.add(Task(f"med_{i}", inp, exp, "qa", "medium"))
    for i, (inp, exp) in enumerate(hard):
        suite.add(Task(f"hard_{i}", inp, exp, "qa", "hard"))

    return suite


def make_distribution_shift_benchmark() -> Tuple[BenchmarkSuite, BenchmarkSuite]:
    """
    Paired benchmark for distribution-shift robustness.
    - Train split: short, clean questions
    - Test split : same questions rewritten to be longer, noisier, and hedged
    """
    train = BenchmarkSuite("DShift-Train")
    test  = BenchmarkSuite("DShift-Test")

    pairs = [
        (
            "What is the capital of Germany?",
            "What is the capital of Germany? Note: consider the country's post-WWII division "
            "into East and West Germany, the role of Bonn as a provisional capital, "
            "and the reunification in 1990 — given this historical complexity, "
            "which city serves as the official capital today?",
        ),
        (
            "Calculate 15 × 7.",
            "Calculate 15 × 7. Note: multiplication of two-digit numbers can involve "
            "carrying operations that are easy to get wrong. Please verify your answer "
            "by also computing it as (10×7) + (5×7) and confirming the results match.",
        ),
        (
            "Name the largest planet in the solar system.",
            "Name the largest planet in the solar system. Some sources use diameter, "
            "others use mass, and still others use volume as the criterion for 'largest'. "
            "Please clarify which metric you are using, state the planet, and note "
            "whether the answer changes across these three metrics.",
        ),
        (
            "Who wrote Hamlet?",
            "Who wrote Hamlet? Attribution of Renaissance-era dramatic works is contested "
            "by some scholars due to collaborative authorship practices of the Elizabethan "
            "theatre. Address the mainstream attribution and briefly note any significant "
            "scholarly disputes or alternative authorship theories.",
        ),
        (
            "What is the boiling point of water?",
            "What is the boiling point of water? Note that this value is pressure-dependent "
            "and changes significantly at altitude. Provide the standard value at sea level "
            "(1 atm), explain why it differs at high altitude, and state the approximate "
            "boiling point at 3,000 metres above sea level.",
        ),
    ]

    for i, (clean, shifted) in enumerate(pairs):
        train.add(Task(f"train_{i}", clean,   "", "shift", "easy"))
        test.add( Task(f"test_{i}",  shifted, "", "shift", "hard"))

    return train, test


def make_perturbation_benchmark() -> BenchmarkSuite:
    """
    Benchmark that tests stability under input perturbations (paraphrase / noise).
    """
    suite = BenchmarkSuite("Perturbation")

    base_questions = [
        "What is the boiling point of water at sea level?",
        "Explain the concept of overfitting in machine learning.",
        "What is the difference between a compiler and an interpreter?",
        "Describe the process of cellular respiration.",
        "What does the central limit theorem state?",
    ]

    # For each base, add original + two perturbed versions
    paraphrase_prefix = [
        "",                                                             # original
        "In your own words, ",                                         # mild paraphrase
        "Using precise technical language, and avoiding ambiguity, ",  # strong paraphrase
    ]

    for i, q in enumerate(base_questions):
        for j, prefix in enumerate(paraphrase_prefix):
            perturbed = prefix + q if prefix else q
            difficulty = ["easy", "medium", "hard"][j]
            suite.add(Task(f"perturb_{i}_{j}", perturbed, "", "perturbation", difficulty))

    return suite
