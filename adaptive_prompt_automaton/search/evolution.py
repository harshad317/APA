"""
search/evolution.py
───────────────────
Evolutionary search that jointly optimises:
  1. State prompt templates (wording mutations)
  2. Transition guard thresholds
  3. (Optional) light topology mutations

Algorithm
─────────
  1. Initialise population by mutating the seed automaton.
  2. Evaluate each individual on a sample of training tasks.
  3. Repeat for N generations:
       a. Sort population by fitness.
       b. Keep top-k elite individuals unchanged.
       c. Fill remainder via tournament-selection + crossover + mutation.
       d. Evaluate new individuals.
  4. Return the overall best automaton seen.

Rich + tqdm are used for all console output.
"""
from __future__ import annotations

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
# Mutation helpers
# ──────────────────────────────────────────────────────────────────────────────

# Synonym-swap pairs for template wording mutations
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
    """Apply random wording mutations to a state prompt template."""
    result = template

    # Word-swap mutations
    for old, new in _WORD_SWAPS:
        if old.lower() in result.lower() and rng.random() < intensity * 0.6:
            # Case-insensitive replace first occurrence
            idx = result.lower().find(old.lower())
            result = result[:idx] + new + result[idx + len(old):]

    # Occasionally append an instruction tag
    if rng.random() < intensity * 0.4:
        addon = rng.choice(_INSTRUCTION_ADDONS)
        if addon not in result:
            result = result.rstrip() + addon

    return result


def mutate_threshold(threshold: float, intensity: float = 0.12, rng: random.Random = random) -> float:
    """Gaussian perturbation on a guard threshold, clamped to (0.05, 0.95)."""
    delta = rng.gauss(0.0, intensity)
    return max(0.05, min(0.95, threshold + delta))


def crossover(parent1: Automaton, parent2: Automaton, swap_prob: float = 0.4, rng: random.Random = random) -> Automaton:
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
    """Return a mutated copy of an automaton."""
    child = automaton.copy()

    # Mutate state templates
    for sid in child.config.states:
        if rng.random() < mutation_rate:
            child.config.states[sid].template = mutate_template(
                child.config.states[sid].template,
                intensity=0.35,
                rng=rng,
            )

    # Mutate guard thresholds
    for t in child.config.transitions:
        if t.guard_type == "threshold" and rng.random() < mutation_rate:
            t.threshold = mutate_threshold(t.threshold, rng=rng)

    return child


# ──────────────────────────────────────────────────────────────────────────────
# EvolutionarySearch
# ──────────────────────────────────────────────────────────────────────────────

class EvolutionarySearch:
    """
    Evolutionary (μ+λ) strategy for training the Adaptive Prompt Automaton.

    Parameters
    ----------
    initial_automaton  : seed Automaton (defines topology; will be mutated)
    llm_api            : LLM backend
    feature_extractor  : FeatureExtractor
    reward_fn          : callable(Episode) → float
    population_size    : number of individuals per generation
    n_generations      : number of evolutionary iterations
    mutation_rate      : probability of mutating each state/transition
    elite_frac         : fraction of top individuals carried forward unchanged
    tournament_size    : k for tournament selection
    n_eval_tasks       : tasks sampled per fitness evaluation
    seed               : random seed for reproducibility
    """

    def __init__(
        self,
        initial_automaton:  Automaton,
        llm_api:            Any,
        feature_extractor:  FeatureExtractor,
        reward_fn:          Callable[[Episode], float],
        population_size:    int   = 8,
        n_generations:      int   = 10,
        mutation_rate:      float = 0.40,
        elite_frac:         float = 0.25,
        tournament_size:    int   = 3,
        n_eval_tasks:       int   = 5,
        seed:               Optional[int] = 42,
    ):
        self.template_automaton = initial_automaton
        self.llm                = llm_api
        self.extractor          = feature_extractor
        self.reward_fn          = reward_fn
        self.population_size    = population_size
        self.n_generations      = n_generations
        self.mutation_rate      = mutation_rate
        self.elite_frac         = elite_frac
        self.tournament_size    = tournament_size
        self.n_eval_tasks       = n_eval_tasks
        self.rng                = random.Random(seed)

        # Diagnostics
        self.history:        List[Dict[str, Any]] = []
        self.best_automaton: Optional[Automaton]  = None
        self.best_fitness:   float                = -float("inf")

    # ------------------------------------------------------------------
    def _init_population(self) -> List[Automaton]:
        population = [self.template_automaton.copy()]
        for _ in range(self.population_size - 1):
            child = mutate(self.template_automaton, mutation_rate=0.6, rng=self.rng)
            population.append(child)
        return population

    def _evaluate(self, automaton: Automaton, tasks: List[str]) -> float:
        executor = AutomatonExecutor(automaton, self.llm, self.extractor)
        sample   = self.rng.sample(tasks, min(self.n_eval_tasks, len(tasks)))
        rewards  = []
        for i, task in enumerate(sample):
            ep      = executor.run_episode(task, episode_id=f"eval_{i}")
            rewards.append(self.reward_fn(ep))
        fitness            = sum(rewards) / len(rewards) if rewards else 0.0
        automaton.fitness  = fitness
        automaton.reward_history.extend(rewards)
        return fitness

    def _tournament_select(self, population: List[Automaton]) -> Automaton:
        candidates = self.rng.sample(population, min(self.tournament_size, len(population)))
        return max(candidates, key=lambda a: a.fitness)

    # ------------------------------------------------------------------
    def run(
        self,
        train_tasks: List[str],
        console:     Optional[Console] = None,
    ) -> Automaton:
        """
        Run the full evolutionary search.

        Parameters
        ----------
        train_tasks : list of task input strings used for fitness evaluation
        console     : Rich Console for output (creates a new one if None)

        Returns
        -------
        The best Automaton found across all generations.
        """
        if console is None:
            console = Console()

        console.print(Panel(
            f"[bold cyan]Evolutionary Search[/bold cyan]\n\n"
            f"  Population size : [green]{self.population_size}[/green]\n"
            f"  Generations     : [green]{self.n_generations}[/green]\n"
            f"  Mutation rate   : [green]{self.mutation_rate}[/green]\n"
            f"  Elite fraction  : [green]{self.elite_frac}[/green]\n"
            f"  Tasks per eval  : [green]{self.n_eval_tasks}[/green]\n"
            f"  Training tasks  : [green]{len(train_tasks)}[/green]",
            title="[bold]APA Training — Evolutionary Search[/bold]",
            border_style="bright_cyan",
        ))

        # ── Initialise ────────────────────────────────────────────────
        population = self._init_population()

        console.print("[yellow]Evaluating initial population…[/yellow]")
        for aut in tqdm(population, desc="  Init Eval", colour="cyan", leave=True):
            self._evaluate(aut, train_tasks)

        # ── Generational loop ─────────────────────────────────────────
        gen_bar = tqdm(range(self.n_generations), desc="  Generations", colour="green")

        for gen in gen_bar:

            population.sort(key=lambda a: -a.fitness)
            gen_best  = population[0]
            gen_mean  = sum(a.fitness for a in population) / len(population)
            gen_worst = population[-1].fitness

            # Track global best
            if gen_best.fitness > self.best_fitness:
                self.best_fitness   = gen_best.fitness
                self.best_automaton = gen_best.copy()

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

            # Print table every 3 generations
            if gen % 3 == 0 or gen == self.n_generations - 1:
                self._print_gen_table(gen, population[:5], console)

            # ── Build next generation ─────────────────────────────────
            n_elite      = max(1, int(self.elite_frac * self.population_size))
            new_pop      = [a.copy() for a in population[:n_elite]]

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
            to_eval = new_pop[n_elite:]
            for aut in tqdm(
                to_eval,
                desc=f"  Gen {gen + 1:02d} Eval",
                colour="yellow",
                leave=False,
            ):
                self._evaluate(aut, train_tasks)

            population = new_pop

        # ── Final summary ─────────────────────────────────────────────
        console.print(Panel(
            f"[bold green]Training complete![/bold green]\n\n"
            f"  Best fitness  : [cyan]{self.best_fitness:.4f}[/cyan]\n"
            f"  Generations   : [cyan]{self.n_generations}[/cyan]\n"
            f"  Total LLM calls so far: [cyan]{self.llm.call_count}[/cyan]",
            title="[bold]Search Results[/bold]",
            border_style="green",
        ))

        return self.best_automaton or population[0]

    # ------------------------------------------------------------------
    @staticmethod
    def _print_gen_table(gen: int, top: List[Automaton], console: Console) -> None:
        table = Table(
            title   = f"Generation {gen} — Top {len(top)} Individuals",
            box     = box.SIMPLE_HEAD,
            show_header = True,
        )
        table.add_column("Rank",     style="cyan",  justify="center", width=6)
        table.add_column("ID",       style="dim",   width=10)
        table.add_column("States",              justify="center", width=7)
        table.add_column("Fitness",  style="green", justify="right",  width=10)
        table.add_column("Episodes",            justify="center", width=9)
        table.add_column("PathEntropy",         justify="right",  width=12)

        for i, aut in enumerate(top):
            table.add_row(
                str(i + 1),
                aut.automaton_id,
                str(len(aut.states)),
                f"{aut.fitness:.4f}",
                str(aut.episodes_run),
                f"{aut.state_visit_entropy():.3f}",
            )
        console.print(table)
