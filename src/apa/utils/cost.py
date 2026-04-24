from __future__ import annotations

from typing import Any

from apa.types import CostSummary


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def estimate_lm_cost(lm: Any) -> CostSummary:
    """Best-effort extraction of token/cost accounting from DSPy LM history."""
    history = getattr(lm, "history", None)
    if not isinstance(history, list):
        return CostSummary()

    summary = CostSummary()
    for item in history:
        if not isinstance(item, dict):
            continue
        usage = item.get("usage") if isinstance(item.get("usage"), dict) else {}
        summary.input_tokens += int(_to_float(usage.get("prompt_tokens", usage.get("input_tokens", 0))))
        summary.output_tokens += int(_to_float(usage.get("completion_tokens", usage.get("output_tokens", 0))))
        summary.usd_cost += _to_float(item.get("cost", item.get("response_cost", 0)))
    return summary
