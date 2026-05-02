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
        parts = []
        for inst_id, kw in zip(self.instruction_id_list, self.kwargs):
            clean_kw = {k: v for k, v in kw.items() if v is not None}
            parts.append(f"{inst_id}: {json.dumps(clean_kw)}")
        return " | ".join(parts) if parts else "(no constraints)"


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

    # ------------------------------------------------------------------
    # Public scoring methods
    # ------------------------------------------------------------------

    def prompt_loose(self, example: IFBenchOfficialExample, response: str) -> float:
        """1.0 if response satisfies ALL constraints (loose), else 0.0."""
        inp = self._make_input(example)
        out = self._el.test_instruction_following_loose(inp, {example.prompt: response})
        return 1.0 if out.follow_all_instructions else 0.0

    def prompt_strict(self, example: IFBenchOfficialExample, response: str) -> float:
        """1.0 if response satisfies ALL constraints (strict), else 0.0."""
        inp = self._make_input(example)
        out = self._el.test_instruction_following_strict(inp, {example.prompt: response})
        return 1.0 if out.follow_all_instructions else 0.0

    def instruction_loose(self, example: IFBenchOfficialExample, response: str) -> float:
        """Fraction of individual constraints satisfied (loose)."""
        inp = self._make_input(example)
        out = self._el.test_instruction_following_loose(inp, {example.prompt: response})
        n   = len(out.follow_instruction_list)
        return sum(out.follow_instruction_list) / n if n else 0.0

    def per_instruction(
        self, example: IFBenchOfficialExample, response: str
    ) -> List[bool]:
        """Per-constraint pass/fail list (loose mode)."""
        inp = self._make_input(example)
        out = self._el.test_instruction_following_loose(inp, {example.prompt: response})
        return list(out.follow_instruction_list)

    def feedback(self, example: IFBenchOfficialExample, response: str) -> str:
        """
        Natural-language feedback string for GEPA's reflection step.
        Describes which constraints failed and why the prompt should be revised.
        """
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

def _row_to_example(row: Dict[str, Any], fallback_key: int) -> IFBenchOfficialExample:
    """Convert a HuggingFace / JSONL row to IFBenchOfficialExample."""
    # Handle both int and str keys
    key = str(row.get("key", fallback_key))

    # instruction_id_list may be a list or a JSON-encoded string
    inst_ids = row.get("instruction_id_list", [])
    if isinstance(inst_ids, str):
        inst_ids = json.loads(inst_ids)

    # kwargs may be a list of dicts or a JSON-encoded string
    kw_list = row.get("kwargs", [])
    if isinstance(kw_list, str):
        kw_list = json.loads(kw_list)
    # Normalise: ensure each entry is a dict
    kw_list = [dict(kw) if kw is not None else {} for kw in kw_list]

    return IFBenchOfficialExample(
        key                 = key,
        prompt              = str(row.get("prompt", "")),
        instruction_id_list = list(inst_ids),
        kwargs              = kw_list,
        constraint_text     = row.get("constraint_text") or None,
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
