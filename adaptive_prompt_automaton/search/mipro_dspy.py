"""
search/mipro_dspy.py
────────────────────
Official dspy.MIPROv2 wrapper — works exactly as described in the MIPRO paper
(EMNLP 2024, arXiv:2406.11695).

How MIPROv2 works here (paper-faithful)
────────────────────────────────────────
  1. The same two-stage IFBench rewriter program as GEPA is optimised:
       Stage 1  (fixed)  : "Respond to the query"  →  draft_answer
       Stage 2  (learned): optimised_instruction   →  final response
                           inputs: prompt, draft_answer, constraint_text
                           output: response

  2. The metric is a scalar metric → float (unlike GEPA's feedback metric).
     MIPROv2 uses Bayesian optimisation over candidate instructions.

  3. Scoring uses the official allenai/IFBench verifiers (prompt_loose).

  4. Training data comes from allenai/IF_multi_constraints_upto5 (IF-RLVR pool).

APA stays unchanged; only GEPA and MIPRO use this IFBench-native pipeline.

Public interface
─────────────────
    searcher = MIPRODSPySearch(
        auto="light",
        ifbench_scorer=scorer,
        train_examples=train,
        val_examples=val,
        dspy_lm=dspy.LM(...),
    )
    optimised_program = searcher.run(console=console)

Backward-compatible re-exports
────────────────────────────────
  MockDSPyLM, APADSPyProgram, _DSPyLLMAdapter, _make_single_state_automaton,
  _dummy_console, _tqdm_wrap
  — still importable so compare.py and the old ifbench_eval path keep working.

Requirements
────────────
    pip install dspy>=3.0.0 datasets
"""
from __future__ import annotations

import os
import re
import time
import random
from pathlib import Path
from typing import Any, Callable, List, Optional

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
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
    from dspy import BaseLM
    _HAS_DSPY = True
except ImportError:
    _HAS_DSPY = False
    BaseLM = object

from ..core.automaton import Automaton, AutomatonConfig, StateConfig
from ..core.executor import AutomatonExecutor, Episode
from ..core.features import FeatureExtractor as _FeatureExtractor
from ..eval.benchmarks import Task, composite_reward
from ..eval.ifbench_official import IFBenchOfficialExample, IFBenchOfficialScorer
from ..utils.api import MockLLM

# Re-use the shared IFBench program + draft generator from gepa_dspy
from .gepa_dspy import (
    IFBenchRewriterProgram,
    generate_stage1_drafts,
    _to_dspy_example,
    _dummy_console,
)


# ──────────────────────────────────────────────────────────────────────────────
# Backward-compatible helpers (used by compare.py APA path)
# ──────────────────────────────────────────────────────────────────────────────

def _tqdm_wrap(iterable, **kwargs):
    if _HAS_TQDM:
        return _tqdm(iterable, **kwargs)
    return iterable


def _make_single_state_automaton(instruction: str) -> Automaton:
    """Build a minimal 1-state terminal Automaton with the given instruction."""
    cfg = AutomatonConfig(
        name        = "mipro_dspy_automaton",
        start_state = "s0",
        max_budget  = 1,
        states      = {
            "s0": StateConfig(
                state_id    = "s0",
                name        = "terminal",
                template    = instruction + "\n\nInput: {input}\nAnswer:",
                is_terminal = True,
            )
        },
        transitions = [],
    )
    return Automaton(cfg)


class _Message:
    __slots__ = ("content", "role", "tool_calls", "reasoning_content")
    def __init__(self, content: str):
        self.content           = content
        self.role              = "assistant"
        self.tool_calls        = None
        self.reasoning_content = None

class _Choice:
    __slots__ = ("message", "finish_reason", "index", "logprobs")
    def __init__(self, content: str):
        self.message       = _Message(content)
        self.finish_reason = "stop"
        self.index         = 0
        self.logprobs      = None

class _Response:
    __slots__ = ("choices", "model", "usage", "_hidden_params")
    def __init__(self, content: str, model: str, n_tokens: int):
        self.choices        = [_Choice(content)]
        self.model          = model
        self.usage          = {"prompt_tokens": 10, "completion_tokens": n_tokens, "total_tokens": n_tokens + 10}
        self._hidden_params = {"response_cost": None}


class MockDSPyLM(BaseLM if _HAS_DSPY else object):  # type: ignore[misc]
    """DSPy 3.x BaseLM backed by APA's MockLLM — kept for backward compat."""

    _FIELD_PATTERN = re.compile(r'\[\[ ## (\w+) ## \]\]')

    def __init__(self, uncertainty_rate=0.15, latency=0.0, seed=42, model_name="mock-llm-v1"):
        if not _HAS_DSPY:
            raise ImportError("dspy is required")
        super().__init__(model=model_name, model_type="chat", temperature=0.7, max_tokens=512, cache=False)
        self._mock = MockLLM(uncertainty_rate=uncertainty_rate, latency=latency, seed=seed)

    def _detect_output_fields(self, messages: List[dict]) -> List[str]:
        for m in reversed(messages):
            if m.get("role") == "user":
                content = m.get("content", "")
                for line in content.splitlines():
                    if "respond with" in line.lower() or "starting with" in line.lower():
                        fields = self._FIELD_PATTERN.findall(line)
                        fields = [f for f in fields if f != "completed"]
                        if fields:
                            return fields
                fields = self._FIELD_PATTERN.findall(content)
                fields = [f for f in fields if f != "completed"]
                if fields:
                    return list(dict.fromkeys(fields))
        return ["answer"]

    def forward(self, prompt=None, messages=None, **kwargs) -> _Response:
        if messages:
            text = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        else:
            text = prompt or ""
        raw_response, n_tokens = self._mock.call(text)
        fields = self._detect_output_fields(messages or [])
        _PRIMARY = {"answer", "prediction", "output", "response", "result"}
        parts = []
        for field in fields:
            if field in _PRIMARY or len(fields) == 1:
                parts.append(f"[[ ## {field} ## ]]\n{raw_response}")
            else:
                stub = "The response demonstrates step-by-step reasoning with clear structure and a verified final answer."
                parts.append(f"[[ ## {field} ## ]]\n{stub}")
        return _Response(content="\n\n".join(parts), model=self.model, n_tokens=n_tokens)


class _DSPyLLMAdapter:
    """Wraps a dspy.LM as an APA-compatible .call() interface."""
    def __init__(self, dspy_lm):
        self._lm        = dspy_lm
        self.call_count = 0
    def call(self, prompt: str, role: str = "user", max_tokens: int = 256):
        self.call_count += 1
        try:
            outputs = self._lm(messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens)
            text    = outputs[0] if outputs else ""
            if isinstance(text, dict):
                text = text.get("text", "")
        except Exception:
            text = ""
        return text, len(str(text).split()) + 10


if _HAS_DSPY:
    class QASignature(dspy.Signature):
        """Answer the question accurately and concisely."""
        question: str = dspy.InputField(desc="The question or task to answer")
        answer:   str = dspy.OutputField(desc="A clear, accurate answer")

    class APADSPyProgram(dspy.Module):
        """Minimal DSPy program backed by QASignature — kept for backward compat."""
        def __init__(self):
            super().__init__()
            self.predict = dspy.Predict(QASignature)
        def forward(self, question: str) -> dspy.Prediction:
            return self.predict(question=question)
else:
    class QASignature: pass       # type: ignore[no-redef]
    class APADSPyProgram: pass    # type: ignore[no-redef]


# ──────────────────────────────────────────────────────────────────────────────
# MIPROv2 score metric (IFBench-native)
# ──────────────────────────────────────────────────────────────────────────────

def _make_mipro_score_metric(scorer: IFBenchOfficialScorer) -> Callable:
    """
    Return a scalar metric function for dspy.MIPROv2.
    MIPROv2 calls  metric(gold, pred, trace=None) → float.
    """
    def metric(
        gold:  "dspy.Example",
        pred:  "dspy.Prediction",
        trace  = None,
    ) -> float:
        response = str(getattr(pred, "response", "") or "")
        example  = IFBenchOfficialExample(
            key                 = str(gold.key),
            prompt              = str(gold.prompt),
            instruction_id_list = list(gold.instruction_id_list),
            kwargs              = [dict(kw) for kw in gold.kwargs],
        )
        return scorer.prompt_loose(example, response)

    return metric


# ──────────────────────────────────────────────────────────────────────────────
# MIPRODSPySearch — public API
# ──────────────────────────────────────────────────────────────────────────────

class MIPRODSPySearch:
    """
    Official dspy.MIPROv2 wrapper using the IFBench two-stage rewriter pipeline.

    MIPROv2 uses Bayesian optimisation to find the best stage-2 rewrite
    instruction by scoring candidate prompts on the training set.

    Parameters
    ──────────
    auto             : MIPROv2 search intensity ('light' | 'medium' | 'heavy')
    ifbench_scorer   : IFBenchOfficialScorer instance
    train_examples   : List[IFBenchOfficialExample] for optimisation
    val_examples     : List[IFBenchOfficialExample] for validation
    dspy_lm          : dspy.LM for both the rewriter and the proposer
    seed             : random seed
    """

    def __init__(
        self,
        auto:           str                              = "light",
        ifbench_scorer: Optional[IFBenchOfficialScorer]  = None,
        train_examples: Optional[List[IFBenchOfficialExample]] = None,
        val_examples:   Optional[List[IFBenchOfficialExample]] = None,
        dspy_lm:        Any                              = None,
        seed:           int                              = 42,
        workers:        int                              = 1,
        # Kept for backward compat with compare.py (ignored when IFBench path active)
        n_eval_tasks:     int   = 5,
        uncertainty_rate: float = 0.15,
        eval_llm:         Any   = None,
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

        # backward-compat properties
        self.n_eval_tasks     = n_eval_tasks
        self.uncertainty_rate = uncertainty_rate

        self.optimised_program: Optional[IFBenchRewriterProgram] = None
        self.best_fitness: float = 0.0

        # keep a mock llm for call_count tracking (compare.py uses this)
        if dspy_lm is not None:
            self._mock_llm = eval_llm or _DSPyLLMAdapter(dspy_lm)
        else:
            self._mock_llm = MockLLM(uncertainty_rate=uncertainty_rate, seed=seed)

    @property
    def call_count(self) -> int:
        return getattr(self._mock_llm, "call_count", 0)

    # ------------------------------------------------------------------

    def run(self, train_tasks=None, console=None) -> Any:
        """
        Run dspy.MIPROv2 optimisation.

        When ifbench_scorer and train_examples are set (IFBench mode):
            Optimises IFBenchRewriterProgram on the official data.
            Returns the compiled dspy.Module.

        Legacy mode (train_tasks provided, no ifbench_scorer):
            Falls back to APADSPyProgram + composite_reward for backward compat.
            Returns an APA Automaton (for compare.py).
        """
        if console is None:
            console = _dummy_console()

        # ── IFBench-native path ──────────────────────────────────────
        if self.train_examples and self.dspy_lm is not None:
            return self._run_ifbench(console)

        # ── Legacy / backward-compat APA path ────────────────────────
        return self._run_legacy(train_tasks or [], console)

    # ------------------------------------------------------------------

    def _run_ifbench(self, console) -> "IFBenchRewriterProgram":
        """IFBench-native MIPROv2: 2-stage rewriter + official scorer."""
        console.print(Panel(
            "[bold magenta]MIPRODSPySearch[/bold magenta]  ·  "
            f"auto=[magenta]{self.auto}[/magenta]  "
            f"train=[magenta]{len(self.train_examples)}[/magenta]  "
            f"val=[magenta]{len(self.val_examples)}[/magenta]",
            title="[magenta]MIPRO — official dspy.MIPROv2 (IFBench pipeline)[/magenta]",
            border_style="magenta",
        ))

        dspy.configure(lm=self.dspy_lm)

        # Stage 1 drafts
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

        trainset = [_to_dspy_example(ex, d) for ex, d in zip(self.train_examples, train_drafts)]
        valset   = [_to_dspy_example(ex, d) for ex, d in zip(self.val_examples,   val_drafts)]

        console.print(f"  [dim]trainset:[/dim] {len(trainset)}  [dim]valset:[/dim] {len(valset)}")

        student = IFBenchRewriterProgram()
        metric  = _make_mipro_score_metric(self.scorer)

        console.print("  [dim]Running dspy.MIPROv2 optimisation …[/dim]")
        t0        = time.perf_counter()
        optimised = None

        try:
            optimiser = dspy.MIPROv2(
                metric       = metric,
                prompt_model = self.dspy_lm,
                task_model   = self.dspy_lm,
                auto         = self.auto,
                seed         = self.seed,
                verbose      = False,
            )
            # MIPROv2 spawns worker processes (multiprocessing) internally.
            # The dspy_lm carries DSPyAPICallCounter which holds threading.Lock
            # — not picklable.  Strip callbacks before compile and restore after
            # to avoid "cannot pickle '_thread.lock' object".
            _callbacks_backup = list(getattr(self.dspy_lm, "callbacks", None) or [])
            if hasattr(self.dspy_lm, "callbacks"):
                self.dspy_lm.callbacks = []
            try:
                with dspy.context(lm=self.dspy_lm):
                    optimised = optimiser.compile(
                        student                    = student,
                        trainset                   = trainset,
                        valset                     = valset,
                        requires_permission_to_run = False,
                    )
            finally:
                # Always restore callbacks so the counter keeps working after
                if hasattr(self.dspy_lm, "callbacks"):
                    self.dspy_lm.callbacks = _callbacks_backup
        except Exception as exc:
            console.print(f"  [yellow]⚠ dspy.MIPROv2 raised:[/yellow] {exc!r}")
            console.print("  [dim]Falling back to base instruction.[/dim]")
            optimised = student

        elapsed = time.perf_counter() - t0
        console.print(f"  [dim]optimisation time:[/dim] {elapsed:.1f}s")
        console.print("  [green]✓[/green] MIPROv2 optimisation complete")

        self.optimised_program = optimised
        return optimised

    def _run_legacy(self, train_tasks: list, console) -> Automaton:
        """
        Legacy APA-compatible path — used by compare.py.
        Optimises APADSPyProgram with QASignature + composite_reward.
        Returns an APA Automaton.
        """
        console.print(Panel(
            "[bold magenta]MIPRODSPySearch (legacy APA mode)[/bold magenta]  ·  "
            f"auto=[magenta]{self.auto}[/magenta]  "
            f"tasks=[magenta]{min(len(train_tasks), self.n_eval_tasks)}[/magenta]",
            title="[magenta]MIPRO (DSPy MIPROv2)[/magenta]",
            border_style="magenta",
        ))

        # Build mock DSPy LM if no real LM
        if self.dspy_lm is not None:
            _dspy_lm = self.dspy_lm
        else:
            _dspy_lm = MockDSPyLM(
                uncertainty_rate = self.uncertainty_rate,
                latency          = 0.0,
                seed             = self.seed,
            )

        dspy.configure(lm=_dspy_lm)

        # Normalise train_tasks
        normalised: List[Task] = []
        for i, item in enumerate(train_tasks):
            if isinstance(item, str):
                normalised.append(Task(task_id=f"t{i}", input_text=item, expected=""))
            else:
                normalised.append(item)
        train_tasks = normalised

        rng      = random.Random(self.seed)
        tasks    = rng.sample(train_tasks, min(self.n_eval_tasks, len(train_tasks)))
        trainset = [
            dspy.Example(question=t.input_text, answer=t.expected or "").with_inputs("question")
            for t in tasks
        ]

        student = APADSPyProgram()
        metric  = _make_legacy_metric(self._mock_llm)

        console.print("  [dim]Running MIPROv2 optimisation …[/dim]")
        t0        = time.perf_counter()
        optimised = None
        try:
            optimiser = dspy.MIPROv2(
                metric       = metric,
                prompt_model = _dspy_lm,
                task_model   = _dspy_lm,
                auto         = self.auto,
                seed         = self.seed,
                verbose      = False,
            )
            optimised = optimiser.compile(
                student                    = student,
                trainset                   = trainset,
                requires_permission_to_run = False,
            )
        except Exception as exc:
            console.print(f"  [yellow]⚠ MIPROv2 raised:[/yellow] {exc!r}")

        elapsed = time.perf_counter() - t0
        console.print(f"  [dim]optimisation time:[/dim] {elapsed:.1f}s")

        if optimised is not None:
            best_automaton = _wrap_apa_automaton(optimised)
            console.print("  [green]✓[/green] MIPROv2 complete — wrapped into 1-state Automaton")
        else:
            best_automaton = _make_single_state_automaton(
                "Break down the question into clear steps. "
                "Step 1: identify key facts. Step 2: reason carefully. Step 3: state your answer."
            )
            console.print("  [yellow]↩[/yellow] Using fallback instruction")

        # Quick fitness probe
        _fe       = _FeatureExtractor()
        executor  = AutomatonExecutor(automaton=best_automaton, llm_api=self._mock_llm, feature_extractor=_fe)
        scores    = [composite_reward(executor.run_episode(t.input_text, verbose=False)) for t in train_tasks[:3]]
        self.best_fitness = sum(scores) / len(scores) if scores else 0.0
        console.print(f"  [dim]estimated fitness:[/dim] {self.best_fitness:.4f}")
        return best_automaton

    # ------------------------------------------------------------------

    def get_optimised_instruction(self) -> str:
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
        """Evaluate optimised program on examples. Returns mean prompt_loose."""
        if self.optimised_program is None or self.dspy_lm is None:
            return 0.0
        if console is None:
            console = _dummy_console()

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


# ──────────────────────────────────────────────────────────────────────────────
# Legacy helpers (backward compat for compare.py APA path)
# ──────────────────────────────────────────────────────────────────────────────

def _make_legacy_metric(llm: MockLLM) -> Callable:
    _fe = _FeatureExtractor()
    def metric(example, prediction, trace=None) -> float:
        answer_text: str = getattr(prediction, "answer", "") or ""
        from ..core.executor import ExecutionStep
        fv   = _fe.extract(task_input=example.question, llm_output=answer_text, samples=[answer_text],
                           verifier_score=0.5, tool_success=True, step=1)
        step = ExecutionStep(step=1, state_id="s0", state_name="terminal",
                             prompt=example.question, response=answer_text, features=fv,
                             transition_taken=None, tokens_used=len(answer_text.split())+10)
        episode = Episode(episode_id="metric_eval", task_input=example.question, path=["s0"],
                          steps=[step], final_output=answer_text,
                          total_tokens=step.tokens_used, reward=0.0, success=True, terminated_by="terminal_state")
        return float(composite_reward(episode))
    return metric


def _wrap_apa_automaton(optimised_program: "APADSPyProgram") -> Automaton:
    try:
        sig         = optimised_program.predict.signature
        instr_obj   = getattr(sig, "instructions", None)
        doc         = getattr(sig, "__doc__", None) or ""
        instruction = str(instr_obj) if instr_obj else (doc.strip() or "Answer the question.")
    except Exception:
        instruction = "Answer the question step by step."
    return _make_single_state_automaton(f"{instruction}\n\nInput: {{input}}\nAnswer:")
