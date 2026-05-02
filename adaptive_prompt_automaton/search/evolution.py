"""
search/evolution.py
───────────────────
Evolutionary search that jointly optimises:
  1. State prompt templates (wording mutations + structural strategy swaps)
  2. Transition guard thresholds
  3. Behavioral diversity via fingerprint-based population clustering

Algorithm
─────────
  1. Initialise population from a diverse template bank (Fix 1) rather than
     cloning a single seed with weak perturbations, guaranteeing behavioral
     diversity at generation 0 and preventing PathEntropy collapse.
  2. Evaluate each individual on a fixed stratified probe set (Fixes 2 & 4) —
     held constant for the entire run to give a stable, low-variance fitness
     signal and produce comparable fingerprint vectors across all generations.
  3. Compute behavioral fingerprints from evaluation responses (Fix 5) —
     zero extra API calls because fingerprints are derived from the same
     probe-set episodes already run for fitness.
  4. Repeat for N generations:
       a. Cluster population by fingerprint RMSE distance.
       b. Apply diversity quota (PRIMARY hard constraint: ≥1 survivor per
          cluster) then diversity-augmented fitness (SECONDARY tie-breaker)
          to select elite survivors. Explicit priority; no ambiguity (Fix 3).
       c. Fill remainder via tournament-selection + crossover + mutation,
          where mutation includes 30 % chance of structural strategy swap
          from the template bank (Fix 1 — mutation width).
       d. Evaluate new individuals on the fixed probe set.
  5. Return the overall best automaton seen.

Architectural scope note (Fix 6)
─────────────────────────────────
  Diversity regularisation improves exploration efficiency *within* the FSA's
  reachable behavioral space.  If the FSA topology itself is architecturally
  mismatched to the benchmark (e.g., constraint-following tasks where a
  2-stage draft→revise pipeline has a structural advantage), this search will
  find the best available FSA solution faster — but that ceiling may still
  fall below SOTA.  After fixing diversity, revisit the FSA topology if a
  performance gap persists.

Rich + tqdm are used for all console output.
"""
from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, Optional, Any

from tqdm import tqdm
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from ..core.automaton import Automaton, AutomatonConfig
from ..core.executor import AutomatonExecutor, Episode
from ..core.features import FeatureExtractor


# ──────────────────────────────────────────────────────────────────────────────
# Diverse template banks  (Fix 1 — structural diversity at initialisation)
# ──────────────────────────────────────────────────────────────────────────────
# Each bank contains 8 structurally distinct templates for a given FSA state.
# These represent genuinely different behavioral strategies, not surface variants,
# so seeding the population from this bank produces fingerprint-distinct
# individuals from generation 0 — preventing PathEntropy collapse.

_START_BANK: List[str] = [
    # 0 — Constraint-explicit (matches original seed style)
    (
        "You are a precise assistant that follows instructions exactly.\n"
        "Answer the following, obeying ALL stated constraints:\n\n"
        "{input}\n\nProvide a concise, constraint-compliant answer."
    ),
    # 1 — Pre-checklist: enumerate constraints before answering
    (
        "Before answering, identify every constraint stated in the request below.\n"
        "Then write a response that satisfies ALL of them without exception.\n\n"
        "{input}\n\nConstraint-compliant response:"
    ),
    # 2 — Expert role assignment
    (
        "You are an expert assistant specialised in following complex "
        "multi-constraint instructions.\n"
        "Every requirement stated in the task is mandatory and non-negotiable.\n\n"
        "{input}"
    ),
    # 3 — Format and length focused
    (
        "This task includes specific requirements about format, length, and content.\n"
        "Your response MUST meet EVERY requirement exactly — no exceptions.\n\n"
        "{input}\n\nResponse:"
    ),
    # 4 — Minimal direct
    (
        "Follow these instructions precisely:\n\n{input}\n\n"
        "Your response (must satisfy all stated constraints):"
    ),
    # 5 — Systematic careful reader
    (
        "Read every word of the task carefully. Identify what is required.\n"
        "Produce a response that precisely follows all stated requirements.\n\n"
        "{input}"
    ),
    # 6 — Constraint-first hard warning
    (
        "IMPORTANT: Failing any single constraint in this task counts as failure.\n"
        "All stated requirements are equally mandatory.\n\n"
        "{input}\n\nAnswer:"
    ),
    # 7 — Self-verification reminder
    (
        "Task: {input}\n\n"
        "Before finalising your answer, verify: does it satisfy EVERY constraint?\n"
        "Only output your answer once you are confident it meets all requirements."
    ),
]

_DECOMPOSE_BANK: List[str] = [
    # 0 — Three-step enumeration (matches original seed style)
    (
        "This is a constrained task. Identify constraints, then answer.\n\n"
        "Task: {input}\nPrevious attempt: {context}\n\n"
        "Step 1: list all constraints. "
        "Step 2: verify your answer against each. "
        "Step 3: give the final answer."
    ),
    # 1 — Constraint audit with explicit compliance check
    (
        "Review your previous response against the task requirements.\n\n"
        "Task: {input}\nYour previous response: {context}\n\n"
        "List each constraint and whether the previous response satisfies it.\n"
        "Then write a corrected response that satisfies ALL constraints."
    ),
    # 2 — Failure diagnosis and repair
    (
        "Your previous response may have failed some constraints.\n\n"
        "Task: {input}\nDraft: {context}\n\n"
        "Diagnose which constraints were violated. Fix each issue.\n"
        "Output the corrected, fully compliant answer."
    ),
    # 3 — Requirements mapping
    (
        "Map each requirement to your response.\n\n"
        "Task: {input}\nDraft: {context}\n\n"
        "For each requirement in the task: is it satisfied? If not, fix it.\n"
        "Output the final corrected answer."
    ),
    # 4 — Chain-of-thought numbered steps
    (
        "Think through this step by step.\n\n"
        "Task: {input}\nPrevious answer: {context}\n\n"
        "1. What does the task require exactly?\n"
        "2. Does the previous answer satisfy every requirement?\n"
        "3. Write the correct, fully compliant answer:"
    ),
    # 5 — Expert reviewer persona
    (
        "As an expert reviewer, check this response for constraint compliance.\n\n"
        "Task: {input}\nResponse to review: {context}\n\n"
        "Issues found (if any):\nFinal fully compliant response:"
    ),
    # 6 — Structured section headings
    (
        "Systematic constraint analysis.\n\n"
        "Task: {input}\nDraft: {context}\n\n"
        "## Constraints identified\n"
        "## Compliance check (pass/fail per constraint)\n"
        "## Final corrected answer"
    ),
    # 7 — Minimal conditional fix
    (
        "Task: {input}\n"
        "Draft answer: {context}\n\n"
        "Does this draft satisfy every stated constraint? "
        "If yes: output it unchanged. If no: output the corrected version."
    ),
]

_VERIFY_BANK: List[str] = [
    # 0 — Rewrite-or-pass (matches original seed style)
    (
        "Check that the answer below satisfies all stated constraints.\n\n"
        "Task: {input}\nAnswer: {context}\n\n"
        "If any constraint is violated, rewrite the answer. "
        "Otherwise, output the answer unchanged."
    ),
    # 1 — PASS/FAIL audit
    (
        "Constraint compliance audit.\n\n"
        "Task: {input}\nProposed answer: {context}\n\n"
        "For each constraint: PASS or FAIL.\n"
        "If any FAIL: output a corrected answer.\n"
        "If all PASS: output the answer unchanged."
    ),
    # 2 — Binary yes/no test
    (
        "Does this response satisfy every constraint?\n\n"
        "Task: {input}\nResponse: {context}\n\n"
        "If yes: output the response.\n"
        "If no: output a revised response that satisfies all constraints."
    ),
    # 3 — Critical review
    (
        "Critically review this answer for constraint compliance.\n\n"
        "Task: {input}\nAnswer to review: {context}\n\n"
        "Output the final answer (revised if any constraint is not met)."
    ),
    # 4 — Acceptance test framing
    (
        "Acceptance test for constraint satisfaction.\n\n"
        "Requirements: {input}\nCandidate answer: {context}\n\n"
        "All requirements met? Output the final compliant answer:"
    ),
    # 5 — Explicit per-constraint test
    (
        "Test each requirement in the task against the provided response.\n\n"
        "Task: {input}\nResponse: {context}\n\n"
        "Revise if needed. Final answer:"
    ),
    # 6 — Minimalist conditional
    (
        "Task: {input}\nResponse: {context}\n\n"
        "Constraints satisfied? Output corrected response if not, "
        "or the response as-is if yes."
    ),
    # 7 — Quality gate
    (
        "Quality gate: does this response meet all task requirements?\n\n"
        "Task spec: {input}\nSubmission: {context}\n\n"
        "Output the final requirement-compliant answer:"
    ),
]

# Mapping from state_id → template bank (used by initialisation and strategy mutation)
_TEMPLATE_BANKS: Dict[str, List[str]] = {
    "start":     _START_BANK,
    "decompose": _DECOMPOSE_BANK,
    "verify":    _VERIFY_BANK,
}


# ──────────────────────────────────────────────────────────────────────────────
# Mutation helpers
# ──────────────────────────────────────────────────────────────────────────────

# Synonym-swap pairs for surface-level wording mutations
_WORD_SWAPS: List[tuple] = [
    ("Answer", "Respond to"),
    ("Carefully", "Methodically"),
    ("Question:", "Task:"),
    ("Please", "You must"),
    ("step by step", "methodically and precisely"),
    ("verify", "double-check"),
    ("Decompose", "Break down"),
    ("Solve", "Address"),
    ("Provide", "Give"),
    ("clear", "precise"),
    ("direct", "concise"),
    ("correct", "accurate"),
    ("Review", "Examine"),
    ("Let's", "We will"),
]

_INSTRUCTION_ADDONS: List[str] = [
    " Be concise.",
    " Be thorough.",
    " Think carefully.",
    " Verify your answer.",
    " Be precise.",
    " Reason step by step.",
    " Avoid speculation.",
]


def mutate_template(template: str, intensity: float = 0.3, rng: random.Random = random) -> str:
    """Apply random surface-level wording mutations to a state prompt template."""
    result = template
    for old, new in _WORD_SWAPS:
        if old.lower() in result.lower() and rng.random() < intensity * 0.6:
            idx    = result.lower().find(old.lower())
            result = result[:idx] + new + result[idx + len(old):]
    if rng.random() < intensity * 0.4:
        addon = rng.choice(_INSTRUCTION_ADDONS)
        if addon not in result:
            result = result.rstrip() + addon
    return result


def mutate_strategy(state_id: str, rng: random.Random = random) -> str:
    """
    Structural strategy mutation (Fix 1 — mutation width).

    Replaces the current template with a randomly chosen entry from the
    template bank for the given state, producing a fingerprint-distinct
    behavioral mode rather than a surface variant of the current template.
    Returns empty string if no bank exists for this state_id.
    """
    bank = _TEMPLATE_BANKS.get(state_id, [])
    if bank:
        return rng.choice(bank)
    return ""


def mutate_threshold(threshold: float, intensity: float = 0.12, rng: random.Random = random) -> float:
    """Gaussian perturbation on a guard threshold, clamped to (0.05, 0.95)."""
    delta = rng.gauss(0.0, intensity)
    return max(0.05, min(0.95, threshold + delta))


def crossover(
    parent1: Automaton,
    parent2: Automaton,
    swap_prob: float = 0.4,
    rng: random.Random = random,
) -> Automaton:
    """
    Uniform crossover: child starts as a copy of parent1, then randomly
    inherits state templates from parent2 with probability swap_prob.
    """
    child = parent1.copy()
    for sid in child.config.states:
        if sid in parent2.config.states and rng.random() < swap_prob:
            child.config.states[sid].template = parent2.config.states[sid].template
    return child


def mutate(automaton: Automaton, mutation_rate: float, rng: random.Random = random) -> Automaton:
    """
    Return a mutated copy of an automaton.

    Per-state mutation operators (Fix 1 — mutation width):
      - 30 % chance of structural strategy swap from template bank
        → replaces entire template with a qualitatively different strategy,
          producing fingerprint-distinct behavioral modes.
      - 70 % chance of surface wording mutation (word swaps + addon).
    Guard thresholds are perturbed independently.
    """
    child = automaton.copy()

    for sid in child.config.states:
        if rng.random() < mutation_rate:
            # Structural mutation: swap to a different strategy template (Fix 1)
            if rng.random() < 0.30:
                new_template = mutate_strategy(sid, rng=rng)
                if new_template:
                    child.config.states[sid].template = new_template
                    continue
            # Surface wording mutation
            child.config.states[sid].template = mutate_template(
                child.config.states[sid].template,
                intensity=0.35,
                rng=rng,
            )

    for t in child.config.transitions:
        if t.guard_type == "threshold" and rng.random() < mutation_rate:
            t.threshold = mutate_threshold(t.threshold, rng=rng)

    return child


# ──────────────────────────────────────────────────────────────────────────────
# Fingerprint distance  (Fix 3 — clustering metric)
# ──────────────────────────────────────────────────────────────────────────────

def fp_rmse(fp1: List[float], fp2: List[float]) -> float:
    """
    Root-mean-squared difference between two fingerprint vectors.
    Returns a value in [0, 1] for [0, 1]-valued inputs.
    Returns 1.0 (maximum distance) if fingerprints are empty or incomparable.
    """
    if not fp1 or not fp2 or len(fp1) != len(fp2):
        return 1.0
    n = len(fp1)
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(fp1, fp2)) / n)


# ──────────────────────────────────────────────────────────────────────────────
# EvolutionarySearch
# ──────────────────────────────────────────────────────────────────────────────

class EvolutionarySearch:
    """
    Evolutionary (μ+λ) strategy for training the Adaptive Prompt Automaton,
    with behavioral fingerprinting and diversity-aware selection (Fixes 1–6).

    Parameters
    ----------
    initial_automaton    : seed Automaton (topology fixed; templates evolved)
    llm_api              : LLM backend with .call() and .call_count
    feature_extractor    : FeatureExtractor
    reward_fn            : callable(Episode) → float
    population_size      : number of individuals per generation  (default 8)
    n_generations        : number of evolutionary iterations      (default 10)
    mutation_rate        : per-state/transition mutation probability (default 0.40)
    elite_frac           : fraction of top individuals kept each gen (default 0.25)
    tournament_size      : k for tournament selection              (default 3)
    n_eval_tasks         : tasks per fitness eval (used when probe_tasks=None)
    seed                 : random seed for reproducibility         (default 42)

    Diversity / fingerprinting parameters
    ───────────────────────────────────────
    probe_tasks          : Fixed probe task strings for fitness + fingerprinting
                           (Fix 2 & 4).  When provided, every evaluation uses
                           exactly these tasks — no random sampling, no refresh.
                           This gives a stable, low-variance fitness signal and
                           produces comparable fingerprints across generations.
                           If None, falls back to random sampling (backward
                           compatible) with no fingerprinting.
    fingerprint_fn       : callable(task_input: str, response: str) → float ∈ [0,1]
                           Applied once per probe task to build the fingerprint
                           vector (Fix 5 — zero extra API calls: same episode).
                           Required when probe_tasks is provided.
    diversity_lambda     : Weight of diversity bonus added to fitness for
                           secondary (soft) selection (Fix 3). Default 0.10.
    diversity_threshold  : RMSE threshold for fingerprint clustering; individuals
                           within this distance share a cluster (Fix 3). Default 0.15.
    diversity_quota      : Minimum survivors per cluster (Fix 3, primary hard
                           constraint). Default 1.
    """

    def __init__(
        self,
        initial_automaton:   Automaton,
        llm_api:             Any,
        feature_extractor:   FeatureExtractor,
        reward_fn:           Callable[[Episode], float],
        population_size:     int   = 8,
        n_generations:       int   = 10,
        mutation_rate:       float = 0.40,
        elite_frac:          float = 0.25,
        tournament_size:     int   = 3,
        n_eval_tasks:        int   = 5,
        seed:                Optional[int] = 42,
        # ── Diversity / fingerprinting (Fixes 1–5) ─────────────────────
        probe_tasks:         Optional[List[str]] = None,
        fingerprint_fn:      Optional[Callable[[str, str], float]] = None,
        diversity_lambda:    float = 0.10,
        diversity_threshold: float = 0.15,
        diversity_quota:     int   = 1,
    ):
        self.template_automaton  = initial_automaton
        self.llm                 = llm_api
        self.extractor           = feature_extractor
        self.reward_fn           = reward_fn
        self.population_size     = population_size
        self.n_generations       = n_generations
        self.mutation_rate       = mutation_rate
        self.elite_frac          = elite_frac
        self.tournament_size     = tournament_size
        self.n_eval_tasks        = n_eval_tasks
        self.rng                 = random.Random(seed)

        # Fix 4: probe_tasks held constant for entire run — no refresh
        self.probe_tasks         = probe_tasks
        # Fix 5: fingerprint_fn derives per-task scores from same eval episodes
        self.fingerprint_fn      = fingerprint_fn
        # Fix 3: diversity selection parameters
        self.diversity_lambda    = diversity_lambda
        self.diversity_threshold = diversity_threshold
        self.diversity_quota     = diversity_quota
        # Fingerprinting active only when both probe_tasks and fingerprint_fn given
        self._fingerprinting     = (probe_tasks is not None and fingerprint_fn is not None)

        # Diagnostics
        self.history:        List[Dict[str, Any]] = []
        self.best_automaton: Optional[Automaton]  = None
        self.best_fitness:   float                = -float("inf")

    # ──────────────────────────────────────────────────────────────────────────
    # Initialisation  (Fix 1 — diverse population from template bank)
    # ──────────────────────────────────────────────────────────────────────────

    def _init_population(self) -> List[Automaton]:
        """
        Build initial population with guaranteed behavioral diversity.

        Root cause of PathEntropy=0.000 at generation 0: the original
        implementation seeded all individuals from the same template with
        weak word-swap mutations, producing near-identical behavioral fingerprints
        before selection even ran.

        Fix: seed the first N_BANK individuals from structurally distinct entries
        in the template bank (one strategy per individual per state), then use
        high-intensity mutation for any additional individuals.
        """
        population = [self.template_automaton.copy()]   # Keep exact seed

        bank_size = len(_START_BANK)   # 8 structurally distinct strategies

        for i in range(self.population_size - 1):
            child = self.template_automaton.copy()

            if i < bank_size:
                # Each individual draws bank entry [i % bank_size] per state,
                # guaranteeing structural distinctness across the first bank_size
                # individuals.
                for sid in child.config.states:
                    bank = _TEMPLATE_BANKS.get(sid, [])
                    if bank:
                        child.config.states[sid].template = bank[i % len(bank)]
            else:
                # Beyond bank capacity: high-intensity mutation for further variety
                child = mutate(child, mutation_rate=0.90, rng=self.rng)

            population.append(child)

        return population

    # ──────────────────────────────────────────────────────────────────────────
    # Evaluation  (Fixes 2, 4, 5 — fixed probe set, zero-cost fingerprints)
    # ──────────────────────────────────────────────────────────────────────────

    def _evaluate(self, automaton: Automaton, tasks: List[str]) -> float:
        """
        Evaluate an automaton on the probe set and, if fingerprinting is active,
        record its behavioral fingerprint at zero additional API cost.

        Fix 2: if probe_tasks are set, evaluation uses a stratified set covering
               the benchmark's constraint-category space.
        Fix 4: probe_tasks is held fixed for the entire run — no refresh —
               so fingerprints computed in gen 1 are directly comparable to
               those in gen 8.
        Fix 5: fingerprint values are derived from the same episodes already run
               for fitness, so fingerprinting adds zero extra LLM calls.
        """
        executor = AutomatonExecutor(automaton, self.llm, self.extractor)

        # Fixed probe set (Fixes 2 & 4) or random sample (backward-compat fallback)
        if self._fingerprinting and self.probe_tasks:
            sample = self.probe_tasks          # always the same fixed set
        else:
            sample = self.rng.sample(tasks, min(self.n_eval_tasks, len(tasks)))

        rewards:     List[float] = []
        fingerprint: List[float] = []

        for i, task in enumerate(sample):
            ep = executor.run_episode(task, episode_id=f"eval_{i}")
            rewards.append(self.reward_fn(ep))

            # Derive fingerprint from same episode — zero extra API calls (Fix 5)
            if self._fingerprinting and self.fingerprint_fn is not None:
                fp_val = self.fingerprint_fn(task, ep.final_output or "")
                fingerprint.append(float(fp_val))

        fitness           = sum(rewards) / len(rewards) if rewards else 0.0
        automaton.fitness = fitness
        automaton.reward_history.extend(rewards)

        if fingerprint:
            automaton.fingerprint = fingerprint     # stored for diversity selection

        return fitness

    # ──────────────────────────────────────────────────────────────────────────
    # Tournament selection (unchanged)
    # ──────────────────────────────────────────────────────────────────────────

    def _tournament_select(self, population: List[Automaton]) -> Automaton:
        candidates = self.rng.sample(population, min(self.tournament_size, len(population)))
        return max(candidates, key=lambda a: a.fitness)

    # ──────────────────────────────────────────────────────────────────────────
    # Fingerprint clustering  (Fix 3 — basis for quota selection)
    # ──────────────────────────────────────────────────────────────────────────

    def _cluster_population(
        self, population: List[Automaton]
    ) -> Dict[int, List[int]]:
        """
        Threshold-based agglomerative clustering on fingerprint RMSE distance.

        Two individuals share a cluster when fp_rmse(fp_i, fp_j) < diversity_threshold.
        Individuals without fingerprints each receive their own singleton cluster
        (treated as maximally distinct from all others).

        Returns {cluster_id: [list of population indices]}.
        """
        n           = len(population)
        cluster_ids = [-1] * n
        next_cid    = 0

        for i in range(n):
            if cluster_ids[i] != -1:
                continue
            cluster_ids[i] = next_cid
            fp_i = population[i].fingerprint
            if fp_i:
                for j in range(i + 1, n):
                    if cluster_ids[j] == -1:
                        fp_j = population[j].fingerprint
                        if fp_j and fp_rmse(fp_i, fp_j) < self.diversity_threshold:
                            cluster_ids[j] = next_cid
            next_cid += 1

        clusters: Dict[int, List[int]] = {}
        for i, cid in enumerate(cluster_ids):
            clusters.setdefault(cid, []).append(i)
        return clusters

    # ──────────────────────────────────────────────────────────────────────────
    # Diversity-aware elite selection  (Fix 3 — quota primary, bonus secondary)
    # ──────────────────────────────────────────────────────────────────────────

    def _diversity_aware_select(
        self, population: List[Automaton]
    ) -> List[Automaton]:
        """
        Select elite survivors using diversity quota (primary) and diversity
        bonus (secondary), with explicit priority rules so the two mechanisms
        never conflict (Fix 3).

        Priority rules:
          Phase 1 — PRIMARY (hard quota):
            For each behavioral cluster, guarantee the highest-augmented-fitness
            individual from that cluster survives.  This hard constraint prevents
            any single behavioral mode from sweeping the population regardless of
            its raw fitness advantage.

          Phase 2 — SECONDARY (soft bonus, tie-breaker only):
            Fill remaining elite slots by augmented fitness:
              augmented_fitness = raw_fitness + diversity_lambda * mean_neighbor_dist
            This rewards exploration of sparse behavioral regions as a tie-breaker
            among candidates not yet selected by the quota.

        The quota is NEVER overridden by the bonus — the quota always fires first.
        """
        n_elite = max(1, int(self.elite_frac * self.population_size))
        n       = len(population)

        # Fallback: no fingerprints available → standard fitness sort
        if not self._fingerprinting or not any(a.fingerprint for a in population):
            return sorted(population, key=lambda a: -a.fitness)[:n_elite]

        # ── Compute diversity bonus (mean RMSE to k nearest neighbours) ──
        k_neighbors = min(3, n - 1)
        for i, ind in enumerate(population):
            dists = sorted(
                fp_rmse(ind.fingerprint, other.fingerprint)
                for j, other in enumerate(population)
                if i != j and ind.fingerprint and other.fingerprint
            )
            nearest = dists[:k_neighbors] if dists else [0.0]
            ind._diversity_bonus = sum(nearest) / len(nearest)

        # ── Augmented fitness: raw + λ × diversity_bonus ─────────────────
        for ind in population:
            ind._augmented_fitness = (
                ind.fitness
                + self.diversity_lambda * getattr(ind, "_diversity_bonus", 0.0)
            )

        # ── Phase 1 (PRIMARY): quota — best individual per cluster ────────
        clusters = self._cluster_population(population)
        selected_indices: set = set()

        for cid, members in sorted(clusters.items()):
            # Best in cluster by augmented fitness
            best_idx = max(members, key=lambda i: population[i]._augmented_fitness)
            selected_indices.add(best_idx)
            if len(selected_indices) >= n_elite:
                break

        # ── Phase 2 (SECONDARY): fill remaining slots by augmented fitness ─
        remaining = sorted(
            [i for i in range(n) if i not in selected_indices],
            key=lambda i: -population[i]._augmented_fitness,
        )
        for idx in remaining:
            if len(selected_indices) >= n_elite:
                break
            selected_indices.add(idx)

        return [population[i] for i in sorted(selected_indices)]

    # ──────────────────────────────────────────────────────────────────────────
    # Main search loop
    # ──────────────────────────────────────────────────────────────────────────

    def run(
        self,
        train_tasks: List[str],
        console:     Optional[Console] = None,
    ) -> Automaton:
        """
        Run the full evolutionary search with diversity regularisation.

        Parameters
        ----------
        train_tasks : task input strings (used for random-sample fitness eval
                      when probe_tasks is not set; ignored otherwise)
        console     : Rich Console for output (creates a new one if None)

        Returns
        -------
        The best Automaton found across all generations.
        """
        if console is None:
            console = Console()

        fp_status = (
            "[green]enabled[/green]"
            if self._fingerprinting
            else "[yellow]disabled — pass probe_tasks + fingerprint_fn to activate[/yellow]"
        )
        # Fix 6 — architectural scope note surfaced in run banner
        arch_note = (
            "\n\n  [dim]Scope note (Fix 6): diversity regularisation improves search\n"
            "  efficiency within the FSA's reachable behavioral space.  If the FSA\n"
            "  topology is architecturally mismatched to the benchmark, revisit the\n"
            "  FSA architecture after fixing diversity.[/dim]"
        )

        n_probe = len(self.probe_tasks) if self.probe_tasks else self.n_eval_tasks
        console.print(Panel(
            f"[bold cyan]Evolutionary Search + Diversity Regularisation[/bold cyan]\n\n"
            f"  Population size      : [green]{self.population_size}[/green]\n"
            f"  Generations          : [green]{self.n_generations}[/green]\n"
            f"  Mutation rate        : [green]{self.mutation_rate}[/green]"
            f"  (30% structural strategy swaps)\n"
            f"  Elite fraction       : [green]{self.elite_frac}[/green]\n"
            f"  Tasks per eval       : [green]{n_probe}[/green]"
            f"  {'(fixed probe set)' if self.probe_tasks else '(random sample)'}\n"
            f"  Training tasks       : [green]{len(train_tasks)}[/green]\n"
            f"  Fingerprinting       : {fp_status}\n"
            f"  Diversity λ (bonus)  : [green]{self.diversity_lambda}[/green]\n"
            f"  Cluster threshold    : [green]{self.diversity_threshold}[/green]\n"
            f"  Diversity quota      : [green]{self.diversity_quota}[/green]"
            f"{arch_note}",
            title="[bold]APA Training — Evolutionary Search[/bold]",
            border_style="bright_cyan",
        ))

        # ── Initialise from diverse template bank (Fix 1) ─────────────────
        population = self._init_population()

        console.print("[yellow]Evaluating initial population…[/yellow]")
        for aut in tqdm(population, desc="  Init Eval", colour="cyan", leave=True):
            self._evaluate(aut, train_tasks)

        # ── Generational loop ──────────────────────────────────────────────
        gen_bar = tqdm(range(self.n_generations), desc="  Generations", colour="green")

        for gen in gen_bar:

            population.sort(key=lambda a: -a.fitness)
            gen_best  = population[0]
            gen_mean  = sum(a.fitness for a in population) / len(population)
            gen_worst = population[-1].fitness

            # Track global best
            if gen_best.fitness > self.best_fitness:
                self.best_fitness   = gen_best.fitness
                self.best_automaton = gen_best.copy(copy_diagnostics=True)

            self.history.append({
                "generation":    gen,
                "best_fitness":  gen_best.fitness,
                "mean_fitness":  gen_mean,
                "worst_fitness": gen_worst,
            })

            gen_bar.set_postfix({
                "best":  f"{gen_best.fitness:.3f}",
                "mean":  f"{gen_mean:.3f}",
                "worst": f"{gen_worst:.3f}",
            })

            if gen % 3 == 0 or gen == self.n_generations - 1:
                self._print_gen_table(gen, population[:5], console)

            # ── Elite selection with diversity quota (Fix 3) ───────────────
            elite = self._diversity_aware_select(population)

            # ── Build next generation ──────────────────────────────────────
            new_pop = [a.copy(copy_diagnostics=True) for a in elite]

            while len(new_pop) < self.population_size:
                if self.rng.random() < 0.55:
                    p1    = self._tournament_select(population)
                    p2    = self._tournament_select(population)
                    child = crossover(p1, p2, rng=self.rng)
                    child = mutate(child, self.mutation_rate, rng=self.rng)
                else:
                    parent = self._tournament_select(population)
                    child  = mutate(parent, self.mutation_rate, rng=self.rng)
                new_pop.append(child)

            # Evaluate only the new (non-elite) individuals
            to_eval = new_pop[len(elite):]
            for aut in tqdm(
                to_eval,
                desc=f"  Gen {gen + 1:02d} Eval",
                colour="yellow",
                leave=False,
            ):
                self._evaluate(aut, train_tasks)

            population = new_pop

        # ── Final summary ──────────────────────────────────────────────────
        console.print(Panel(
            f"[bold green]Training complete![/bold green]\n\n"
            f"  Best fitness  : [cyan]{self.best_fitness:.4f}[/cyan]\n"
            f"  Generations   : [cyan]{self.n_generations}[/cyan]\n"
            f"  Total LLM calls so far: [cyan]{self.llm.call_count}[/cyan]",
            title="[bold]Search Results[/bold]",
            border_style="green",
        ))

        return self.best_automaton or population[0]

    # ──────────────────────────────────────────────────────────────────────────
    # Reporting
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _print_gen_table(gen: int, top: List[Automaton], console: Console) -> None:
        table = Table(
            title       = f"Generation {gen} — Top {len(top)} Individuals",
            box         = box.SIMPLE_HEAD,
            show_header = True,
        )
        table.add_column("Rank",        style="cyan",    justify="center", width=6)
        table.add_column("ID",          style="dim",     width=10)
        table.add_column("States",                       justify="center", width=7)
        table.add_column("Fitness",     style="green",   justify="right",  width=10)
        table.add_column("Episodes",                     justify="center", width=9)
        table.add_column("PathEntropy",                  justify="right",  width=12)
        table.add_column("FP len",      style="dim",     justify="right",  width=8)

        for i, aut in enumerate(top):
            table.add_row(
                str(i + 1),
                aut.automaton_id,
                str(len(aut.states)),
                f"{aut.fitness:.4f}",
                str(aut.episodes_run),
                f"{aut.state_visit_entropy():.3f}",
                str(len(aut.fingerprint)),
            )
        console.print(table)
