"""
eval/ifbench_official.py
────────────────────────
Official IFBench data loader and scorer.

Data sources
────────────
  Train / Val : allenai/IF_multi_constraints_upto5 on HuggingFace
                (IF-RLVR training pool, default 300 train + 100 val)
  Test        : vendor/ifbench/data/IFBench_test.jsonl
                (official AllenAI NeurIPS-2025 held-out set, 300 prompts)

Scorer
──────
  Wraps allenai/IFBench evaluation_lib verifiers.
  Returns prompt_loose (1.0 if ALL constraints pass, 0.0 otherwise) —
  the primary metric reported in the IFBench paper.

  None-valued kwargs are filtered before passing to the verifiers because
  test_instruction_following_loose does not filter them (unlike the strict
  variant), which otherwise causes TypeError inside build_description().

Usage
─────
  from adaptive_prompt_automaton.eval.ifbench_official import (
      IFBenchOfficialExample,
      IFBenchOfficialScorer,
      load_ifbench_train_val,
      load_ifbench_test,
  )

  scorer = IFBenchOfficialScorer()
  train, val = load_ifbench_train_val(train_size=300, val_size=100)
  test        = load_ifbench_test()

  score = scorer.prompt_loose(test[0], response="my response")  # 0.0 or 1.0
"""
from __future__ import annotations

import json
import sys
import random
import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Vendor path ───────────────────────────────────────────────────────────────
_VENDOR_DIR = Path(__file__).parent.parent.parent / "vendor" / "ifbench"


def _ensure_ifbench_on_path() -> None:
    """Add allenai/IFBench vendor directory to sys.path."""
    vendor = str(_VENDOR_DIR)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)


# ──────────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class IFBenchOfficialExample:
    """
    A single IFBench example.

    Fields mirror InputExample from evaluation_lib.py:
      key                  — unique identifier (str)
      prompt               — user prompt / query (includes constraint wording)
      instruction_id_list  — list of constraint type IDs, e.g. ["count:keywords_multiple"]
      kwargs               — per-constraint parameters (None values will be filtered by scorer)
      constraint_text      — human-readable constraint summary (auto-generated if None)
    """
    key:                 str
    prompt:              str
    instruction_id_list: List[str]
    kwargs:              List[Dict[str, Any]]
    constraint_text:     Optional[str] = field(default=None)

    def get_constraint_text(self) -> str:
        """Return constraint_text if set, otherwise generate from instruction IDs."""
        if self.constraint_text:
            return self.constraint_text
        _ensure_ifbench_on_path()
        try:
            import instructions_registry as _ir  # type: ignore[import]

            parts = []
            for inst_id, kw in zip(self.instruction_id_list, self.kwargs):
                checker_cls = _ir.INSTRUCTION_DICT.get(inst_id)
                if checker_cls is None:
                    raise KeyError(inst_id)
                clean_kw = {k: v for k, v in kw.items() if v is not None}
                checker = checker_cls(inst_id)
                parts.append(str(checker.build_description(**clean_kw)))
            if parts:
                return " | ".join(parts)
        except Exception:
            pass
        parts = []
        for inst_id, kw in zip(self.instruction_id_list, self.kwargs):
            clean_kw = {k: v for k, v in kw.items() if v is not None}
            parts.append(f"{inst_id}: {json.dumps(clean_kw)}")
        return " | ".join(parts) if parts else "(no constraints)"

    def to_apa_task_input(self) -> str:
        """
        Return the task input string for APA — the raw prompt augmented with
        a structured constraint checklist.

        GEPA provides constraint_text as an explicit third field to its
        stage-2 rewriter.  APA receives a single task_input string, so we
        embed the structured constraint metadata directly in the prompt text.
        This gives the APA rewrite state the same explicit constraint
        inventory that GEPA uses, improving satisfaction of format, count,
        keyword, and other precisely-specified constraints.

        Format:
          <raw prompt>

          ---
          Required constraints (ALL must be satisfied):
          • <constraint_id>: {<params>}
          ...

        The "---" separator makes the constraint block visually distinct from
        the freeform query so the model can address both independently.
        """
        if not self.instruction_id_list:
            return self.prompt

        constraint_text = self.get_constraint_text()
        raw_lines = [
            part.strip()
            for chunk in constraint_text.split("\t")
            for part in chunk.split(" | ")
            if part.strip()
        ]
        if raw_lines:
            lines = [f"  • {line}" for line in raw_lines]
        else:
            lines = []
            for inst_id, kw in zip(self.instruction_id_list, self.kwargs):
                clean_kw = {k: v for k, v in kw.items() if v is not None}
                lines.append(f"  • {inst_id}: {json.dumps(clean_kw)}")

        constraint_block = (
            "\n\n---\n"
            "Required constraints (ALL must be satisfied):\n"
            + "\n".join(lines)
        )
        return self.prompt + constraint_block


# ──────────────────────────────────────────────────────────────────────────────
# Scorer
# ──────────────────────────────────────────────────────────────────────────────

class IFBenchOfficialScorer:
    """
    Wraps allenai/IFBench evaluation_lib for inline constraint scoring.

    Provides:
      prompt_loose(example, response)       → 1.0 or 0.0
      prompt_strict(example, response)      → 1.0 or 0.0
      instruction_loose(example, response)  → fraction of constraints passed
      feedback(example, response)           → natural-language failure description
    """

    def __init__(self) -> None:
        _ensure_ifbench_on_path()
        try:
            import evaluation_lib as _el        # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "Could not import evaluation_lib from vendor/ifbench/. "
                "Make sure vendor/ifbench/ contains the allenai/IFBench files."
            ) from exc
        self._el = _el
        self._warm_nltk_resources()

    def _warm_nltk_resources(self) -> None:
        """Preload NLTK resources used by IFBench before threaded scoring."""
        try:
            import instructions_util as _iu  # type: ignore[import]

            _iu.count_stopwords("warm up")
            _iu.split_into_sentences("Warm up.")
            _iu._get_sentence_tokenizer()
        except Exception:
            # Scoring will surface the original verifier error if resources are
            # genuinely unavailable; this only avoids lazy-load races.
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clean_kwargs(self, kwargs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter None values — required for test_instruction_following_loose."""
        return [{k: v for k, v in kw.items() if v is not None} for kw in kwargs]

    def _make_input(self, example: IFBenchOfficialExample) -> Any:
        return self._el.InputExample(
            key                 = example.key,
            instruction_id_list = list(example.instruction_id_list),
            prompt              = example.prompt,
            kwargs              = self._clean_kwargs(example.kwargs),
        )

    def supports(self, example: IFBenchOfficialExample) -> bool:
        """Return True if the vendored IFBench scorer can verify this example."""
        try:
            import instructions_registry as _ir  # type: ignore[import]
            return bool(example.instruction_id_list) and all(
                inst_id in _ir.INSTRUCTION_DICT
                for inst_id in example.instruction_id_list
            )
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Public scoring methods
    # ------------------------------------------------------------------

    def prompt_loose(self, example: IFBenchOfficialExample, response: str) -> float:
        """1.0 if response satisfies ALL constraints (loose), else 0.0."""
        if not self.supports(example):
            return 0.0
        inp = self._make_input(example)
        out = self._el.test_instruction_following_loose(inp, {example.prompt: response})
        return 1.0 if out.follow_all_instructions else 0.0

    def prompt_strict(self, example: IFBenchOfficialExample, response: str) -> float:
        """1.0 if response satisfies ALL constraints (strict), else 0.0."""
        if not self.supports(example):
            return 0.0
        inp = self._make_input(example)
        out = self._el.test_instruction_following_strict(inp, {example.prompt: response})
        return 1.0 if out.follow_all_instructions else 0.0

    def instruction_loose(self, example: IFBenchOfficialExample, response: str) -> float:
        """Fraction of individual constraints satisfied (loose)."""
        if not self.supports(example):
            return 0.0
        inp = self._make_input(example)
        out = self._el.test_instruction_following_loose(inp, {example.prompt: response})
        n   = len(out.follow_instruction_list)
        return sum(out.follow_instruction_list) / n if n else 0.0

    def per_instruction(
        self, example: IFBenchOfficialExample, response: str
    ) -> List[bool]:
        """Per-constraint pass/fail list (loose mode)."""
        if not self.supports(example):
            return [False] * len(example.instruction_id_list)
        inp = self._make_input(example)
        out = self._el.test_instruction_following_loose(inp, {example.prompt: response})
        return list(out.follow_instruction_list)

    def feedback(self, example: IFBenchOfficialExample, response: str) -> str:
        """
        Natural-language feedback string for GEPA's reflection step.
        Describes which constraints failed and why the prompt should be revised.
        """
        if not self.supports(example):
            return (
                "The local IFBench verifier does not support this training "
                "example's constraint IDs. Prefer general instruction-following "
                "changes: explicitly identify every stated requirement in the "
                "prompt, satisfy them in the final answer, and avoid extra "
                "commentary unless requested."
            )
        inp  = self._make_input(example)
        out  = self._el.test_instruction_following_loose(inp, {example.prompt: response})
        passed = sum(out.follow_instruction_list)
        total  = len(out.follow_instruction_list)

        if passed == total:
            return (
                "The response correctly satisfies all constraints. "
                "The current prompt instruction is effective."
            )

        failed_ids = [
            inst_id
            for inst_id, ok in zip(out.instruction_id_list, out.follow_instruction_list)
            if not ok
        ]
        return (
            f"The response failed {total - passed} of {total} constraint(s): "
            f"{', '.join(failed_ids)}. "
            "The stage-2 rewrite instruction should more explicitly tell the model "
            "to check and satisfy every listed constraint before finalising its answer. "
            "Consider adding a step-by-step constraint verification pass."
        )

    # ------------------------------------------------------------------
    # Batch evaluation helper
    # ------------------------------------------------------------------

    def batch_prompt_loose(
        self,
        examples:  List[IFBenchOfficialExample],
        responses: List[str],
    ) -> float:
        """Mean prompt_loose over a list of (example, response) pairs."""
        if not examples:
            return 0.0
        scores = [
            self.prompt_loose(ex, resp)
            for ex, resp in zip(examples, responses)
        ]
        return sum(scores) / len(scores)


# ──────────────────────────────────────────────────────────────────────────────
# Data loaders
# ──────────────────────────────────────────────────────────────────────────────

def _parse_serialized(value: Any, default: Any) -> Any:
    """Parse JSON or Python-literal strings used by different HF schemas."""
    if value is None:
        return default
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            return default


def _extract_prompt(row: Dict[str, Any]) -> str:
    """Extract the user prompt across official JSONL and HF chat schemas."""
    if row.get("prompt"):
        return str(row["prompt"])
    messages = row.get("messages")
    if isinstance(messages, str):
        messages = _parse_serialized(messages, [])
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content", ""))
        for message in messages:
            if isinstance(message, dict) and message.get("content"):
                return str(message.get("content", ""))
    return str(row.get("input", ""))


def _extract_ground_truth(row: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Extract instruction IDs and kwargs across official IFBench and IF-RLVR schemas.

    The IF-RLVR HuggingFace pool stores verifier metadata under `ground_truth`,
    e.g. "[{'instruction_id': [...], 'kwargs': [...]}]".
    """
    inst_ids = row.get("instruction_id_list", row.get("instruction_id", []))
    kw_list = row.get("kwargs", row.get("kwarg", []))

    if inst_ids or kw_list:
        inst_ids = _parse_serialized(inst_ids, [])
        kw_list = _parse_serialized(kw_list, [])
    else:
        gt = _parse_serialized(row.get("ground_truth"), [])
        if isinstance(gt, dict):
            gt = [gt]
        inst_ids = []
        kw_list = []
        if isinstance(gt, list):
            for item in gt:
                if not isinstance(item, dict):
                    continue
                ids = item.get("instruction_id_list", item.get("instruction_id", []))
                kws = item.get("kwargs", item.get("kwarg", []))
                ids = _parse_serialized(ids, [])
                kws = _parse_serialized(kws, [])
                if isinstance(ids, str):
                    ids = [ids]
                if not isinstance(kws, list):
                    kws = [kws]
                inst_ids.extend(list(ids))
                kw_list.extend(kws)

    if isinstance(inst_ids, str):
        inst_ids = [inst_ids]
    if not isinstance(kw_list, list):
        kw_list = [kw_list]
    kw_list = [dict(kw) if isinstance(kw, dict) else {} for kw in kw_list]
    while len(kw_list) < len(inst_ids):
        kw_list.append({})

    return list(inst_ids), kw_list[: len(inst_ids)]


def _row_to_example(row: Dict[str, Any], fallback_key: int) -> IFBenchOfficialExample:
    """Convert a HuggingFace / JSONL row to IFBenchOfficialExample."""
    # Handle both int and str keys
    key = str(row.get("key", fallback_key))

    inst_ids, kw_list = _extract_ground_truth(row)

    return IFBenchOfficialExample(
        key                 = key,
        prompt              = _extract_prompt(row),
        instruction_id_list = list(inst_ids),
        kwargs              = kw_list,
        constraint_text     = row.get("constraint_text") or row.get("constraint") or None,
    )


def load_ifbench_test() -> List[IFBenchOfficialExample]:
    """
    Load the official IFBench test set (300 prompts) from vendor/ifbench/data/.

    Returns
    ───────
    List[IFBenchOfficialExample]
    """
    test_path = _VENDOR_DIR / "data" / "IFBench_test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(
            f"IFBench test file not found at {test_path}. "
            "Run:  curl -fsSL https://raw.githubusercontent.com/allenai/IFBench/main/data/IFBench_test.jsonl "
            f"-o {test_path}"
        )
    examples = []
    with open(test_path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if line:
                examples.append(_row_to_example(json.loads(line), i))
    return examples


def load_ifbench_train_val(
    train_size: int = 300,
    val_size:   int = 100,
    seed:       int = 42,
) -> Tuple[List[IFBenchOfficialExample], List[IFBenchOfficialExample]]:
    """
    Load train and validation splits from allenai/IF_multi_constraints_upto5
    on HuggingFace (the IF-RLVR training pool).

    Parameters
    ──────────
    train_size : int
        Number of training examples (default 300).
    val_size : int
        Number of validation examples (default 100).
    seed : int
        Random seed for reproducible shuffling.

    Returns
    ───────
    (train_examples, val_examples)

    Requirements
    ────────────
        pip install datasets
    """
    try:
        from datasets import load_dataset   # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "HuggingFace `datasets` is required to load IF-RLVR train data.\n"
            "Install with:  pip install datasets"
        ) from exc

    ds = load_dataset("allenai/IF_multi_constraints_upto5", split="train")

    rng     = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    train_idx = indices[:train_size]
    val_idx   = indices[train_size : train_size + val_size]

    train = [_row_to_example(dict(ds[i]), i) for i in train_idx]
    val   = [_row_to_example(dict(ds[i]), i) for i in val_idx]
    return train, val
