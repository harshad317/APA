"""
search/mipro_dspy.py
────────────────────
DSPy MIPROv2 wrapper for the Adaptive Prompt Automaton comparison suite.

This module replaces the hand-rolled MIPROSearch with the official
dspy.MIPROv2 optimiser while keeping exactly the same public interface:

    MIPRODSPySearch(...).run(train_tasks, console) -> Automaton

The resulting Automaton is a 1-state wrapper whose prompt template has been
optimised by MIPROv2 — allowing a fair apples-to-apples comparison with
APA's 4-state evolutionary automaton and GEPA's reflective evolution.

Strategic note (for the NeurIPS paper)
───────────────────────────────────────
APA core files (automaton / features / executor / evolution) remain
completely framework-free.  DSPy is only used here, for the MIPRO
*baseline*, so that the comparison uses the official implementation rather
than a hand-reimplementation.  This strengthens the experimental rigour of
the paper.

Requirements
────────────
    pip install dspy>=3.0.0
"""
from __future__ import annotations

import os
import re
import sys
import time
import types
import random
from typing import List, Optional, Callable

# ── Rich / tqdm (graceful fallback if not installed) ──────────────────────────
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

# ── DSPy ──────────────────────────────────────────────────────────────────────
try:
    import dspy
    from dspy import BaseLM
    _HAS_DSPY = True
except ImportError:
    _HAS_DSPY = False
    BaseLM = object           # dummy so class definition doesn't crash

# ── APA internals ─────────────────────────────────────────────────────────────
from ..core.automaton import (
    Automaton, AutomatonConfig,
    StateConfig, TransitionConfig,
)
from ..core.executor import AutomatonExecutor, Episode
from ..core.features import FeatureExtractor as _FeatureExtractor
from ..eval.benchmarks import Task, composite_reward
from ..utils.api import MockLLM


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _dummy_console() -> "Console":
    """Return a Console that discards output when rich is unavailable."""
    class _Sink:
        def print(self, *a, **kw): pass
        def rule(self, *a, **kw): pass
    return _Sink()  # type: ignore


def _tqdm_wrap(iterable, **kwargs):
    """tqdm if available, else plain iteration."""
    if _HAS_TQDM:
        return _tqdm(iterable, **kwargs)
    return iterable


# ──────────────────────────────────────────────────────────────────────────────
# MockDSPyLM  — DSPy 3.x-compatible wrapper around APA's MockLLM
# ──────────────────────────────────────────────────────────────────────────────

class _Message:
    """Minimal stand-in for openai.types.chat.ChatCompletionMessage."""
    __slots__ = ("content", "role", "tool_calls", "reasoning_content")

    def __init__(self, content: str):
        self.content          = content
        self.role             = "assistant"
        self.tool_calls       = None
        self.reasoning_content = None


class _Choice:
    """Minimal stand-in for a single completion choice."""
    __slots__ = ("message", "finish_reason", "index", "logprobs")

    def __init__(self, content: str):
        self.message      = _Message(content)
        self.finish_reason = "stop"
        self.index        = 0
        self.logprobs     = None


class _Response:
    """
    Minimal object whose shape satisfies dspy.BaseLM._process_lm_response.

    Constraints (learned from source inspection):
      • response.choices    — iterable of _Choice objects
      • response.model      — str
      • response.usage      — dict  (NOT an object; dict() is called on it)
      • response._hidden_params — dict with optional 'response_cost' key
    """
    __slots__ = ("choices", "model", "usage", "_hidden_params")

    def __init__(self, content: str, model: str, n_tokens: int):
        self.choices        = [_Choice(content)]
        self.model          = model
        self.usage          = {
            "prompt_tokens":     10,
            "completion_tokens": n_tokens,
            "total_tokens":      n_tokens + 10,
        }
        self._hidden_params = {"response_cost": None}


class MockDSPyLM(BaseLM if _HAS_DSPY else object):  # type: ignore[misc]
    """
    DSPy 3.x BaseLM subclass backed by APA's MockLLM.

    The critical detail: DSPy's ChatAdapter.parse() uses the regex
        r'\\[\\[ ## (\\w+) ## \\]\\]'
    to split the LM response into named fields.  Therefore forward() must
    wrap the raw MockLLM text in the exact field-marker format:

        [[ ## answer ## ]]
        <answer text here>

    Without this wrapper every call raises AdapterParseError.
    """

    def __init__(
        self,
        uncertainty_rate: float = 0.15,
        latency:          float = 0.0,
        seed:             int   = 42,
        model_name:       str   = "mock-llm-v1",
    ):
        if not _HAS_DSPY:
            raise ImportError("dspy is required for MockDSPyLM. Run: pip install dspy")

        # BaseLM requires a model string and disables caching for mocks
        super().__init__(
            model       = model_name,
            model_type  = "chat",
            temperature = 0.7,
            max_tokens  = 512,
            cache       = False,
        )
        self._mock = MockLLM(
            uncertainty_rate = uncertainty_rate,
            latency          = latency,
            seed             = seed,
        )

    # ------------------------------------------------------------------
    # BaseLM interface
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Field detection helper
    # ------------------------------------------------------------------

    _FIELD_PATTERN = re.compile(r'\[\[ ## (\w+) ## \]\]')

    def _detect_output_fields(self, messages: List[dict]) -> List[str]:
        """
        Parse DSPy-formatted messages to find the expected output field names.

        DSPy always includes a line like:
            "Respond with the corresponding output fields, starting with the
             field `[[ ## fieldname ## ]]`, ..."
        in the last user message.  We extract all field names from that line,
        excluding the synthetic `completed` sentinel.
        """
        # Search last user message (most likely to contain the Respond line)
        for m in reversed(messages):
            if m.get("role") == "user":
                content = m.get("content", "")
                # Only the "Respond with" instruction line has field markers
                for line in content.splitlines():
                    if "respond with" in line.lower() or "starting with" in line.lower():
                        fields = self._FIELD_PATTERN.findall(line)
                        fields = [f for f in fields if f != "completed"]
                        if fields:
                            return fields
                # Fallback: scan the whole content for field markers
                fields = self._FIELD_PATTERN.findall(content)
                fields = [f for f in fields if f != "completed"]
                # Only use if these look like output specs (not input examples)
                if fields:
                    return list(dict.fromkeys(fields))  # deduplicate, preserve order
        return ["answer"]   # safe default

    # ------------------------------------------------------------------
    # BaseLM interface
    # ------------------------------------------------------------------

    def forward(
        self,
        prompt:   Optional[str]             = None,
        messages: Optional[List[dict]]      = None,
        **kwargs,
    ) -> _Response:
        """
        Call MockLLM and return a _Response whose content is wrapped in
        DSPy's field-marker format so ChatAdapter.parse() succeeds.

        Dynamically detects which output fields the current signature expects
        (by parsing the "Respond with …" line in the last user message) so
        that internal DSPy proposer signatures (e.g. with `observations`,
        `proposed_instruction`, etc.) are handled correctly alongside the
        normal QASignature `answer` field.
        """
        # Extract the last user message as the task text
        if messages:
            text = next(
                (m["content"] for m in reversed(messages) if m.get("role") == "user"),
                "",
            )
        else:
            text = prompt or ""

        raw_response, n_tokens = self._mock.call(text)

        # Detect which fields this signature expects
        fields = self._detect_output_fields(messages or [])

        # Build a response that satisfies ALL expected fields.
        # The main "answer"-like field gets the real MockLLM response;
        # auxiliary fields (observations, reasoning, etc.) get a short stub.
        _PRIMARY = {"answer", "prediction", "output", "response", "result"}
        parts = []
        for field in fields:
            if field in _PRIMARY or len(fields) == 1:
                parts.append(f"[[ ## {field} ## ]]\n{raw_response}")
            else:
                # Plausible one-liner stub for internal DSPy proposer fields
                stub = (
                    "The response demonstrates step-by-step reasoning "
                    "with clear structure and a verified final answer."
                )
                parts.append(f"[[ ## {field} ## ]]\n{stub}")

        formatted = "\n\n".join(parts)

        return _Response(
            content  = formatted,
            model    = self.model,
            n_tokens = n_tokens,
        )


# ──────────────────────────────────────────────────────────────────────────────
# DSPy Signature & Module
# ──────────────────────────────────────────────────────────────────────────────

if _HAS_DSPY:
    class QASignature(dspy.Signature):
        """Answer the question accurately and concisely."""

        question: str = dspy.InputField(desc="The question or task to answer")
        answer:   str = dspy.OutputField(desc="A clear, accurate answer")

    class APADSPyProgram(dspy.Module):
        """
        Minimal DSPy program used as the student for MIPROv2.

        Wraps a single dspy.Predict(QASignature) call.  MIPROv2 will
        optimise the instruction prefix and optionally add few-shot demos.
        """

        def __init__(self):
            super().__init__()
            self.predict = dspy.Predict(QASignature)

        def forward(self, question: str) -> dspy.Prediction:
            return self.predict(question=question)

else:
    # Stubs so the rest of the module can be imported even without dspy
    class QASignature:  # type: ignore[no-redef]
        pass

    class APADSPyProgram:  # type: ignore[no-redef]
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Metric bridge: dspy trainset example → APA composite_reward
# ──────────────────────────────────────────────────────────────────────────────

def _make_metric(llm: MockLLM) -> Callable:
    """
    Return a DSPy metric function compatible with MIPROv2.

    MIPROv2 calls  metric(example, prediction, trace=None) → float | bool.
    We bridge this to APA's composite_reward by building a minimal Episode.
    """
    _fe = _FeatureExtractor()
    executor = AutomatonExecutor(
        automaton         = _make_single_state_automaton("Answer the question: {input}"),
        llm_api           = llm,
        feature_extractor = _fe,
    )

    def metric(
        example:    "dspy.Example",
        prediction: "dspy.Prediction",
        trace       = None,
    ) -> float:
        answer_text: str = getattr(prediction, "answer", "") or ""
        # Build a minimal Episode for reward computation
        from ..core.executor import Episode, ExecutionStep
        fe  = _FeatureExtractor()
        fv  = fe.extract(
            task_input      = example.question,
            llm_output      = answer_text,
            samples         = [answer_text],
            verifier_score  = 0.5,
            tool_success    = True,
            step            = 1,
        )
        step = ExecutionStep(
            step            = 1,
            state_id        = "s0",
            state_name      = "terminal",
            prompt          = example.question,
            response        = answer_text,
            features        = fv,
            transition_taken = None,
            tokens_used     = len(answer_text.split()) + 10,
        )
        episode = Episode(
            episode_id      = "metric_eval",
            task_input      = example.question,
            path            = ["s0"],
            steps           = [step],
            final_output    = answer_text,
            total_tokens    = step.tokens_used,
            reward          = 0.0,
            success         = True,
            terminated_by   = "terminal_state",
        )
        return float(composite_reward(episode))

    return metric


# ──────────────────────────────────────────────────────────────────────────────
# Automaton builder helpers
# ──────────────────────────────────────────────────────────────────────────────

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


def _wrap_automaton(optimised_program: "APADSPyProgram") -> Automaton:
    """
    Convert a MIPROv2-optimised APADSPyProgram into an APA Automaton.

    We extract the instruction string that MIPROv2 found and embed it
    into a 1-state Automaton so compare.py can evaluate it identically
    to the other baselines.
    """
    try:
        # MIPROv2 stores the best instruction in predict.signature.__doc__
        # or in the extended_signature instructions attribute.
        sig       = optimised_program.predict.signature
        doc       = getattr(sig, "__doc__", None) or ""
        # Also try the instructions field if present
        instr_obj = getattr(sig, "instructions", None)
        instruction = str(instr_obj) if instr_obj else (doc.strip() or "Answer the question.")
    except Exception:
        instruction = "Answer the question step by step."

    template = (
        f"{instruction}\n\n"
        "Input: {input}\n"
        "Answer:"
    )
    return _make_single_state_automaton(template)


# ──────────────────────────────────────────────────────────────────────────────
# _DSPyLLMAdapter — wraps a real dspy.LM as an APA-compatible .call() LLM
# ──────────────────────────────────────────────────────────────────────────────

class _DSPyLLMAdapter:
    """
    Thin adapter: wraps any dspy.LM so it exposes the APA .call() interface.

    Used as eval_llm when MIPRODSPySearch is given a real dspy.LM but no
    explicit APA-compatible eval_llm.  The dspy.LM is called via its
    __call__ method with a plain user message.
    """

    def __init__(self, dspy_lm):
        self._lm        = dspy_lm
        self.call_count = 0

    def call(self, prompt: str, role: str = "user", max_tokens: int = 256):
        self.call_count += 1
        try:
            outputs = self._lm(
                messages   = [{"role": "user", "content": prompt}],
                max_tokens = max_tokens,
            )
            text = outputs[0] if outputs else ""
            if isinstance(text, dict):
                text = text.get("text", "")
        except Exception:
            text = ""
        return text, len(str(text).split()) + 10


# ──────────────────────────────────────────────────────────────────────────────
# MIPRODSPySearch — public API (mirrors MIPROSearch.run interface)
# ──────────────────────────────────────────────────────────────────────────────

class MIPRODSPySearch:
    """
    DSPy MIPROv2 wrapper with the same public interface as MIPROSearch.

        best_automaton = MIPRODSPySearch(...).run(train_tasks, console)

    Parameters
    ──────────
    auto : str
        MIPROv2 search intensity: 'light' | 'medium' | 'heavy'.
    n_eval_tasks : int
        Number of training tasks to use as the MIPROv2 trainset.
    seed : int
        Random seed for reproducibility.
    uncertainty_rate : float
        MockLLM uncertainty injection rate (0–1). Only used when dspy_lm is None.
    dspy_lm : dspy.BaseLM or None
        Optional pre-built DSPy LM to use instead of MockDSPyLM.
        Pass a real dspy.LM("openai/gpt-4.1-mini") here for live evaluation.
        When provided, eval_llm should also be set.
    eval_llm : object or None
        APA-compatible LLM (has .call(prompt) → (str, int)) used for the
        fitness probe and episode evaluation.  If None and dspy_lm is set,
        a lightweight adapter wraps dspy_lm for APA compatibility.
    """

    def __init__(
        self,
        auto:             str   = "light",
        n_eval_tasks:     int   = 5,
        seed:             int   = 42,
        uncertainty_rate: float = 0.15,
        dspy_lm           = None,   # real dspy.LM for live runs
        eval_llm          = None,   # APA-compatible LLM for fitness probe
    ):
        if not _HAS_DSPY:
            raise ImportError(
                "dspy>=3.0.0 is required for MIPRODSPySearch.\n"
                "Install with:  pip install dspy"
            )
        self.auto             = auto
        self.n_eval_tasks     = n_eval_tasks
        self.seed             = seed
        self.uncertainty_rate = uncertainty_rate

        if dspy_lm is not None:
            # Real LM path — use the provided dspy.LM directly
            self._dspy_lm  = dspy_lm
            self._mock_llm = eval_llm or _DSPyLLMAdapter(dspy_lm)
        else:
            # Mock path (default)
            self._mock_llm = MockLLM(
                uncertainty_rate = uncertainty_rate,
                seed             = seed,
            )
            self._dspy_lm = MockDSPyLM(
                uncertainty_rate = uncertainty_rate,
                latency          = 0.0,
                seed             = seed,
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def call_count(self) -> int:
        """Total MockLLM calls made during training (for sample-efficiency comparison)."""
        return self._mock_llm.call_count

    def run(
        self,
        train_tasks,          # List[str] or List[Task]
        console     = None,
    ) -> Automaton:
        """
        Run MIPROv2 optimisation on train_tasks and return the best Automaton.

        Parameters
        ──────────
        train_tasks : list of str or list of Task
            Training examples.  Accepts plain strings (task input text) or
            Task objects.  At most n_eval_tasks are used.
        console : rich.console.Console or None
            Optional rich console for progress output.

        Returns
        ───────
        Automaton
            A 1-state Automaton whose instruction has been optimised by MIPROv2.
        """
        # Normalise: accept both List[str] and List[Task]
        normalised: List[Task] = []
        for i, item in enumerate(train_tasks):
            if isinstance(item, str):
                normalised.append(Task(
                    task_id    = f"t{i}",
                    input_text = item,
                    expected   = "",
                ))
            else:
                normalised.append(item)
        train_tasks = normalised
        if console is None:
            console = _dummy_console()

        console.print(
            Panel(
                "[bold magenta]MIPRODSPySearch[/bold magenta]  ·  "
                f"auto=[cyan]{self.auto}[/cyan]  "
                f"tasks=[cyan]{min(len(train_tasks), self.n_eval_tasks)}[/cyan]",
                title="[magenta]MIPRO (DSPy MIPROv2)[/magenta]",
                border_style="magenta",
            )
        )

        # ── 1. Configure DSPy to use our mock LM ──────────────────────
        dspy.configure(lm=self._dspy_lm)

        # ── 2. Build trainset as dspy.Example objects ──────────────────
        rng   = random.Random(self.seed)
        tasks = rng.sample(train_tasks, min(self.n_eval_tasks, len(train_tasks)))

        trainset = [
            dspy.Example(
                question = t.input_text,
                answer   = t.expected or "",
            ).with_inputs("question")
            for t in tasks
        ]

        console.print(
            f"  [dim]trainset size:[/dim] {len(trainset)} examples"
        )

        # ── 3. Build student program ───────────────────────────────────
        student = APADSPyProgram()

        # ── 4. Build metric ────────────────────────────────────────────
        metric = _make_metric(self._mock_llm)

        # ── 5. Run MIPROv2 ────────────────────────────────────────────
        console.print("  [dim]Running MIPROv2 optimisation …[/dim]")
        t0 = time.perf_counter()

        try:
            optimiser = dspy.MIPROv2(
                metric       = metric,
                prompt_model = self._dspy_lm,
                task_model   = self._dspy_lm,
                auto         = self.auto,
                seed         = self.seed,
                verbose      = False,
            )

            optimised = optimiser.compile(
                student                  = student,
                trainset                 = trainset,
                requires_permission_to_run = False,
            )

        except Exception as exc:
            # Surface a clear error message and fall back to a reasonable default
            console.print(
                f"  [yellow]⚠ MIPROv2 raised:[/yellow] {exc!r}\n"
                "  [dim]Falling back to rule-based best instruction.[/dim]"
            )
            optimised = None

        elapsed = time.perf_counter() - t0
        console.print(f"  [dim]optimisation time:[/dim] {elapsed:.1f}s")

        # ── 6. Wrap optimised program → APA Automaton ─────────────────
        if optimised is not None:
            best_automaton = _wrap_automaton(optimised)
            console.print(
                "  [green]✓[/green] MIPROv2 complete — wrapped into 1-state Automaton"
            )
        else:
            # Fallback: use the instruction that tends to score highest on
            # MockLLM (structured decomposition triggers the structure bonus)
            fallback_instruction = (
                "Break down the question into clear steps. "
                "Step 1: identify key facts. "
                "Step 2: reason carefully. "
                "Step 3: state your answer."
            )
            best_automaton = _make_single_state_automaton(fallback_instruction)
            console.print(
                "  [yellow]↩[/yellow] Using fallback structured-decomposition instruction"
            )

        # ── 7. Quick fitness probe (re-use trainset for consistency) ──
        _fe = _FeatureExtractor()
        executor = AutomatonExecutor(
            automaton         = best_automaton,
            llm_api           = self._mock_llm,
            feature_extractor = _fe,
        )
        fitness_scores = []
        for task in train_tasks[:min(3, len(train_tasks))]:
            ep = executor.run_episode(task.input_text, verbose=False)
            fitness_scores.append(composite_reward(ep))
        self.best_fitness: float = (
            sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0.0
        )
        console.print(
            f"  [dim]estimated fitness:[/dim] {self.best_fitness:.4f}"
        )

        return best_automaton

    # ------------------------------------------------------------------
    # Convenience: quick eval so compare.py can call the same helpers
    # ------------------------------------------------------------------

    def evaluate(
        self,
        automaton:   Automaton,
        eval_tasks:  List[Task],
        console      = None,
    ) -> List[Episode]:
        """Evaluate automaton on eval_tasks; return list of Episodes."""
        if console is None:
            console = _dummy_console()

        _fe = _FeatureExtractor()
        executor = AutomatonExecutor(
            automaton         = automaton,
            llm_api           = self._mock_llm,
            feature_extractor = _fe,
        )
        episodes: List[Episode] = []

        bar = _tqdm_wrap(
            eval_tasks,
            desc   = "  MIPRO-DSPy eval",
            colour = "magenta",
            leave  = False,
        )
        for task in bar:
            ep = executor.run_episode(task.input_text, verbose=False)
            ep.reward = composite_reward(ep)
            episodes.append(ep)

        return episodes
