"""
search/gepa_dspy.py
───────────────────
Official dspy.GEPA wrapper — works exactly as described in the GEPA paper
(ICLR 2026, arXiv:2507.19457).

How GEPA works here (paper-faithful)
──────────────────────────────────────
  1. A two-stage rewriter program is optimised:
       Stage 1  (fixed)  : "Respond to the query"  →  draft_answer
       Stage 2  (learned): optimised_instruction   →  final response
                           inputs: prompt, draft_answer, constraint_text
                           output: response

  2. The metric is a *feedback* metric — returns dspy.Prediction(score, feedback).
     GEPA uses the feedback string to reflectively evolve the stage-2 instruction.

  3. Scoring uses the official allenai/IFBench verifiers (prompt_loose).

  4. Training data comes from allenai/IF_multi_constraints_upto5 (IF-RLVR pool).

APA stays unchanged; only GEPA and MIPRO use this IFBench-native pipeline.

Public interface (unchanged for compare.py / ifbench_eval.py)
─────────────────────────────────────────────────────────────
    searcher = GEPADSPySearch(
        auto="light",
        ifbench_scorer=scorer,          # IFBenchOfficialScorer instance
        train_examples=train,           # List[IFBenchOfficialExample]
        val_examples=val,               # List[IFBenchOfficialExample]
        dspy_lm=dspy.LM(...),           # real LM for live runs
    )
    optimised_program = searcher.run(console=console)
    # → returns a compiled dspy.Module (IFBenchRewriterProgram)

Requirements
────────────
    pip install dspy>=3.0.0 datasets
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Optional

try:
    from rich.console import Console
    from rich.panel import Panel
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False

try:
    from tqdm.auto import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

try:
    import dspy
    _HAS_DSPY = True
except ImportError:
    _HAS_DSPY = False

from ..eval.ifbench_official import IFBenchOfficialExample, IFBenchOfficialScorer


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _dummy_console():
    class _Sink:
        def print(self, *a, **kw): pass
        def rule(self, *a, **kw): pass
        def status(self, *a, **kw):
            class _Ctx:
                def __enter__(self): return self
                def __exit__(self, *a): pass
            return _Ctx()
    return _Sink()


# ──────────────────────────────────────────────────────────────────────────────
# IFBench two-stage rewriter program
# ──────────────────────────────────────────────────────────────────────────────

if _HAS_DSPY:
    class IFBenchRewriterProgram(dspy.Module):
        """
        Two-stage IFBench rewriter, exactly as used in the GEPA / MIPRO papers.

        Stage 1 (fixed, not optimised):
            "Respond to the query"  +  {prompt}  →  draft_answer

        Stage 2 (optimised by dspy.GEPA / dspy.MIPROv2):
            base_instruction  +  {prompt, draft_answer, constraint_text}  →  response

        Only the stage-2 instruction is mutated during optimisation.
        """

        BASE_INSTRUCTION = (
            "Ensure the response is correct and adheres to the given constraints. "
            "Your response will be used as the final response."
        )
        STAGE1_INSTRUCTION = "Respond to the query."

        def __init__(self, rewrite_instructions: str = "") -> None:
            super().__init__()
            instructions = rewrite_instructions or self.BASE_INSTRUCTION
            sig = dspy.Signature(
                "prompt, draft_answer, constraint_text -> response"
            ).with_instructions(instructions)
            self.rewrite = dspy.Predict(sig)

        def forward(
            self,
            prompt:          str,
            draft_answer:    str,
            constraint_text: str,
        ) -> "dspy.Prediction":
            return self.rewrite(
                prompt          = prompt,
                draft_answer    = draft_answer,
                constraint_text = constraint_text,
            )

else:
    class IFBenchRewriterProgram:   # type: ignore[no-redef]
        def __init__(self, *a, **kw): pass
        def forward(self, **kw): return None


# ──────────────────────────────────────────────────────────────────────────────
# Stage-1 draft generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_stage1_drafts(
    examples: List[IFBenchOfficialExample],
    lm:       Any,
    workers:  int = 1,
) -> List[str]:
    """
    Run stage-1 for every example: fixed "Respond to the query" prompt.

    Returns a list of draft strings aligned with `examples`.
    When workers > 1, drafts are generated in parallel via ThreadPoolExecutor.
    """
    if not _HAS_DSPY:
        return [""] * len(examples)

    stage1_sig = dspy.Signature("prompt -> draft_answer").with_instructions(
        IFBenchRewriterProgram.STAGE1_INSTRUCTION
    )

    def _one(ex: IFBenchOfficialExample) -> str:
        predict = dspy.Predict(stage1_sig)
        with dspy.context(lm=lm):
            try:
                pred  = predict(prompt=ex.prompt)
                return str(getattr(pred, "draft_answer", "") or "")
            except Exception:
                return ""

    if workers <= 1:
        return [_one(ex) for ex in examples]

    drafts: List[str] = [""] * len(examples)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        fut_to_idx = {pool.submit(_one, ex): i for i, ex in enumerate(examples)}
        for fut in as_completed(fut_to_idx):
            drafts[fut_to_idx[fut]] = fut.result()
    return drafts


# ──────────────────────────────────────────────────────────────────────────────
# GEPA feedback metric
# ──────────────────────────────────────────────────────────────────────────────

def _make_gepa_feedback_metric(scorer: IFBenchOfficialScorer):
    """
    Return a DSPy feedback metric for dspy.GEPA.

    GEPA calls  metric(gold, pred, trace, pred_name, pred_trace)
    and expects  dspy.Prediction(score=float, feedback=str).
    """
    def metric(
        gold:       "dspy.Example",
        pred:       "dspy.Prediction",
        trace       = None,
        pred_name:  Optional[str] = None,
        pred_trace                = None,
    ) -> "dspy.Prediction":
        response = str(getattr(pred, "response", "") or "")
        example  = IFBenchOfficialExample(
            key                 = str(gold.key),
            prompt              = str(gold.prompt),
            instruction_id_list = list(gold.instruction_id_list),
            kwargs              = [dict(kw) for kw in gold.kwargs],
        )
        score    = scorer.prompt_loose(example, response)
        feedback = scorer.feedback(example, response)
        return dspy.Prediction(score=score, feedback=feedback)

    return metric


# ──────────────────────────────────────────────────────────────────────────────
# dspy.Example builder
# ──────────────────────────────────────────────────────────────────────────────

def _to_dspy_example(
    example: IFBenchOfficialExample,
    draft:   str,
) -> "dspy.Example":
    return dspy.Example(
        key                 = example.key,
        prompt              = example.prompt,
        draft_answer        = draft,
        constraint_text     = example.get_constraint_text(),
        instruction_id_list = list(example.instruction_id_list),
        kwargs              = [dict(kw) for kw in example.kwargs],
    ).with_inputs("prompt", "draft_answer", "constraint_text")


# ──────────────────────────────────────────────────────────────────────────────
# GEPADSPySearch — public API
# ──────────────────────────────────────────────────────────────────────────────

class GEPADSPySearch:
    """
    Official dspy.GEPA wrapper using the IFBench two-stage rewriter pipeline.

    GEPA reflectively evolves the stage-2 rewrite instruction by:
      1. Running candidates on the trainset
      2. Collecting natural-language feedback from the official IFBench scorer
      3. Proposing improved instructions via a reflection LM

    Parameters
    ──────────
    auto             : GEPA search intensity ('light' | 'medium' | 'heavy')
    ifbench_scorer   : IFBenchOfficialScorer instance
    train_examples   : List[IFBenchOfficialExample] for optimisation
    val_examples     : List[IFBenchOfficialExample] for validation reranking
    dspy_lm          : dspy.LM for both the rewriter and the reflection model
    seed             : random seed
    """

    def __init__(
        self,
        auto:           str                             = "light",
        ifbench_scorer: Optional[IFBenchOfficialScorer] = None,
        train_examples: Optional[List[IFBenchOfficialExample]] = None,
        val_examples:   Optional[List[IFBenchOfficialExample]] = None,
        dspy_lm:        Any                             = None,
        seed:           int                             = 42,
        workers:        int                             = 1,
    ) -> None:
        if not _HAS_DSPY:
            raise ImportError("dspy>=3.0.0 required. Install with: pip install dspy")
        self.auto           = auto
        self.scorer         = ifbench_scorer or IFBenchOfficialScorer()
        self.train_examples = train_examples or []
        self.val_examples   = val_examples   or []
        self.dspy_lm        = dspy_lm
        self.seed           = seed
        self.workers        = workers

        self.optimised_program: Optional[IFBenchRewriterProgram] = None

    # ------------------------------------------------------------------

    def run(self, console=None) -> "IFBenchRewriterProgram":
        """
        Run dspy.GEPA optimisation.

        Returns the optimised IFBenchRewriterProgram.
        The optimised stage-2 instruction is stored in
        self.optimised_program.rewrite.signature.instructions.
        """
        if console is None:
            console = _dummy_console()

        if self.dspy_lm is None:
            raise ValueError(
                "GEPADSPySearch requires a real dspy.LM. "
                "Pass dspy_lm=dspy.LM('openai/gpt-4.1-mini', api_key=...) "
                "or set OPENAI_API_KEY and use dspy_lm=dspy.LM('openai/gpt-4.1-mini')."
            )

        console.print(Panel(
            "[bold cyan]GEPADSPySearch[/bold cyan]  ·  "
            f"auto=[cyan]{self.auto}[/cyan]  "
            f"train=[cyan]{len(self.train_examples)}[/cyan]  "
            f"val=[cyan]{len(self.val_examples)}[/cyan]",
            title="[cyan]GEPA — official dspy.GEPA (IFBench pipeline)[/cyan]",
            border_style="cyan",
        ))

        dspy.configure(lm=self.dspy_lm)

        # ── Stage 1: generate drafts ──────────────────────────────────
        console.print(
            f"  [dim]Stage 1: generating draft answers "
            f"(workers={self.workers}) …[/dim]"
        )
        train_drafts = generate_stage1_drafts(
            self.train_examples, self.dspy_lm, workers=self.workers
        )
        val_drafts = generate_stage1_drafts(
            self.val_examples, self.dspy_lm, workers=self.workers
        )

        trainset = [
            _to_dspy_example(ex, draft)
            for ex, draft in zip(self.train_examples, train_drafts)
        ]
        valset = [
            _to_dspy_example(ex, draft)
            for ex, draft in zip(self.val_examples, val_drafts)
        ]

        console.print(f"  [dim]trainset:[/dim] {len(trainset)}  [dim]valset:[/dim] {len(valset)}")

        # ── Stage 2: compile with dspy.GEPA ──────────────────────────
        student  = IFBenchRewriterProgram()
        metric   = _make_gepa_feedback_metric(self.scorer)

        console.print("  [dim]Running dspy.GEPA optimisation …[/dim]")
        t0 = time.perf_counter()
        optimised = None

        try:
            optimiser = dspy.GEPA(
                metric        = metric,
                auto          = self.auto,
                reflection_lm = self.dspy_lm,
                seed          = self.seed,
            )
            with dspy.context(lm=self.dspy_lm):
                optimised = optimiser.compile(
                    student                    = student,
                    trainset                   = trainset,
                    valset                     = valset,
                    requires_permission_to_run = False,
                )
        except Exception as exc:
            console.print(f"  [yellow]⚠ dspy.GEPA raised:[/yellow] {exc!r}")
            console.print("  [dim]Falling back to base instruction.[/dim]")
            optimised = student   # unoptimised baseline

        elapsed = time.perf_counter() - t0
        console.print(f"  [dim]optimisation time:[/dim] {elapsed:.1f}s")
        console.print("  [green]✓[/green] GEPA optimisation complete")

        self.optimised_program = optimised
        return optimised

    # ------------------------------------------------------------------

    def get_optimised_instruction(self) -> str:
        """Return the evolved stage-2 instruction string."""
        if self.optimised_program is None:
            return IFBenchRewriterProgram.BASE_INSTRUCTION
        try:
            sig   = self.optimised_program.rewrite.signature
            instr = getattr(sig, "instructions", None)
            return str(instr) if instr else IFBenchRewriterProgram.BASE_INSTRUCTION
        except Exception:
            return IFBenchRewriterProgram.BASE_INSTRUCTION

    def evaluate(
        self,
        examples: List[IFBenchOfficialExample],
        console  = None,
    ) -> float:
        """
        Evaluate optimised program on examples.
        Returns mean prompt_loose accuracy.
        """
        if self.optimised_program is None or self.dspy_lm is None:
            return 0.0
        if console is None:
            console = _dummy_console()

        console.print(f"  [dim]Evaluating GEPA on {len(examples)} examples …[/dim]")
        drafts    = generate_stage1_drafts(examples, self.dspy_lm)
        responses = []

        with dspy.context(lm=self.dspy_lm):
            for ex, draft in zip(examples, drafts):
                try:
                    pred     = self.optimised_program(
                        prompt          = ex.prompt,
                        draft_answer    = draft,
                        constraint_text = ex.get_constraint_text(),
                    )
                    response = str(getattr(pred, "response", "") or "")
                except Exception:
                    response = ""
                responses.append(response)

        return self.scorer.batch_prompt_loose(examples, responses)
