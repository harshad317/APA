from __future__ import annotations

import inspect
from dataclasses import asdict, is_dataclass
from typing import Any


def example_inputs(example: Any) -> dict[str, Any]:
    if hasattr(example, "inputs"):
        try:
            return dict(example.inputs())
        except Exception:
            pass
    if isinstance(example, dict):
        return dict(example)
    if hasattr(example, "items"):
        return dict(example.items())
    return {}


def _accepts_example_argument(program: Any) -> bool:
    target = getattr(program, "forward", None) or getattr(program, "__call__", None)
    if target is None:
        return False
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return False

    if "example" in sig.parameters:
        return True
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return False


def call_program(program: Any, example: Any, *, override_inputs: dict[str, Any] | None = None) -> Any:
    kwargs = dict(override_inputs) if override_inputs is not None else example_inputs(example)
    if _accepts_example_argument(program):
        kwargs = {**kwargs, "example": example}
    return program(**kwargs)


def prediction_to_dict(prediction: Any) -> dict[str, Any]:
    if prediction is None:
        return {}
    if isinstance(prediction, dict):
        return dict(prediction)
    if hasattr(prediction, "items"):
        try:
            return dict(prediction.items())
        except Exception:
            pass
    if is_dataclass(prediction):
        return asdict(prediction)
    return {"value": str(prediction)}


def extract_prediction_text(prediction: Any) -> str:
    payload = prediction_to_dict(prediction)
    if not payload:
        return ""

    preferred_keys = ["answer", "response", "output", "final_response", "query", "summary"]
    for key in preferred_keys:
        if key in payload:
            return str(payload[key])

    first_key = next(iter(payload.keys()))
    return str(payload[first_key])


def extract_program_prompts(program: Any) -> dict[str, str]:
    prompts: dict[str, str] = {}
    if not hasattr(program, "named_predictors"):
        return prompts

    for name, predictor in program.named_predictors():
        instruction = getattr(getattr(predictor, "signature", None), "instructions", None)
        prompts[name] = "" if instruction is None else str(instruction)
    return prompts


def collect_lms(root: Any, max_depth: int = 4) -> list[Any]:
    seen_obj: set[int] = set()
    seen_lm: set[int] = set()
    found: list[Any] = []

    def _walk(obj: Any, depth: int) -> None:
        if obj is None or depth > max_depth:
            return

        obj_id = id(obj)
        if obj_id in seen_obj:
            return
        seen_obj.add(obj_id)

        if hasattr(obj, "history") and callable(getattr(obj, "__call__", None)):
            if obj_id not in seen_lm:
                seen_lm.add(obj_id)
                found.append(obj)
            return

        if isinstance(obj, dict):
            for value in obj.values():
                _walk(value, depth + 1)
            return
        if isinstance(obj, (list, tuple, set)):
            for value in obj:
                _walk(value, depth + 1)
            return

        if hasattr(obj, "__dict__"):
            for value in vars(obj).values():
                _walk(value, depth + 1)

    _walk(root, 0)
    return found
