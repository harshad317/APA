from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from apa.benchmarks.base import BenchmarkSpec, DatasetSplits, mode_size
from apa.constants import OFFICIAL_INVOCATION_CAPS
from apa.types import RuntimeConfig

TRAIN_URL = "https://raw.githubusercontent.com/gepa-ai/gepa-artifact/main/gepa_artifact/benchmarks/IFBench/data/IFBench_train.jsonl"
TEST_URL = "https://raw.githubusercontent.com/gepa-ai/gepa-artifact/main/gepa_artifact/benchmarks/IFBench/data/IFBench_test.jsonl"


def _data_dir() -> Path:
    return Path.home() / ".cache" / "apa" / "ifbench"


def _ensure_ifbench_files() -> tuple[Path, Path]:
    data_dir = _data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "IFBench_train.jsonl"
    test_path = data_dir / "IFBench_test.jsonl"

    if not train_path.exists():
        train_path.write_bytes(urlopen(TRAIN_URL, timeout=30).read())
    if not test_path.exists():
        test_path.write_bytes(urlopen(TEST_URL, timeout=30).read())

    return train_path, test_path


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_splits(runtime: RuntimeConfig) -> DatasetSplits:
    import dspy

    train_path, test_path = _ensure_ifbench_files()
    train_val_rows = _load_jsonl(train_path)
    test_rows = _load_jsonl(test_path)

    train_val = [dspy.Example(**row).with_inputs("prompt") for row in train_val_rows]
    test = [dspy.Example(**row).with_inputs("prompt") for row in test_rows]

    # Matches official protocol: first 300 val, next 300 train.
    train = train_val[300:600]
    val = train_val[:300]

    t, v, s = mode_size(runtime)
    if t > 0:
        train, val, test = train[:t], val[:v], test[:s]

    return DatasetSplits(train=train, val=val, test=test)


def _program_factory(runtime: RuntimeConfig, retrieval: Any | None = None) -> Any:
    import dspy

    class GenerateResponse(dspy.Signature):
        """Respond to the query."""

        query = dspy.InputField()
        response = dspy.OutputField()

    class EnsureCorrectResponse(dspy.Signature):
        """Ensure the response is correct and adheres to constraints."""

        query = dspy.InputField()
        response = dspy.InputField()
        final_response = dspy.OutputField()

    class IFBenchProgram(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            self.generate_response_module = dspy.ChainOfThought(GenerateResponse)
            self.ensure_correct_response_module = dspy.ChainOfThought(EnsureCorrectResponse)

        def forward(self, prompt: str) -> Any:
            response = self.generate_response_module(query=prompt).response
            final = self.ensure_correct_response_module(query=prompt, response=response).final_response
            return dspy.Prediction(response=final)

    return IFBenchProgram()


def _augment_candidate_responses(response: str) -> list[str]:
    lines = response.split("\n")
    response_remove_first = "\n".join(lines[1:]).strip()
    response_remove_last = "\n".join(lines[:-1]).strip()
    response_remove_both = "\n".join(lines[1:-1]).strip()

    revised = response.replace("*", "")
    revised_remove_first = response_remove_first.replace("*", "")
    revised_remove_last = response_remove_last.replace("*", "")
    revised_remove_both = response_remove_both.replace("*", "")

    all_candidates = [
        response,
        revised,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_remove_first,
        revised_remove_last,
        revised_remove_both,
    ]
    return [candidate for candidate in all_candidates if candidate.strip()]


def _metric_with_feedback(example: Any, pred: Any, trace: Any | None = None) -> Any:
    import dspy

    from .ifbench_utils.instructions_registry import INSTRUCTION_DICT

    response = str(pred.response)
    candidate_responses = _augment_candidate_responses(response)

    instruction_ids = list(example.instruction_id_list)
    follow = []
    correct_feedback = []
    incorrect_feedback = []

    for idx, instruction_id in enumerate(instruction_ids):
        instruction_cls = INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)
        kwargs = dict(example.kwargs[idx]) if idx < len(example.kwargs) else {}
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        description = instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            description = instruction.build_description(prompt=example.prompt)

        is_following = any(candidate and instruction.check_following(candidate) for candidate in candidate_responses)
        follow.append(bool(is_following))

        if is_following:
            correct_feedback.append(description)
        else:
            incorrect_feedback.append(description)

    score = float(sum(follow) / max(1, len(follow)))

    feedback_parts = []
    if correct_feedback:
        feedback_parts.append(
            "Your response correctly followed the following instructions:\n" + "\n".join(correct_feedback)
        )
    if incorrect_feedback and correct_feedback:
        feedback_parts.append(
            "However, your response did not follow the following instructions properly:\n"
            + "\n".join(incorrect_feedback)
        )
    elif incorrect_feedback:
        feedback_parts.append(
            "Your response did not follow the following instructions properly:\n"
            + "\n".join(incorrect_feedback)
        )

    feedback = "\n".join(feedback_parts).strip()
    return dspy.Prediction(score=score, feedback=feedback)


def _metric(example: Any, pred: Any, trace: Any | None = None) -> float:
    return float(_metric_with_feedback(example, pred, trace).score)


def _feedback_map_factory() -> dict[str, Any]:
    import dspy

    def _feedback_for_response_module(
        predictor_output: dict[str, Any],
        predictor_inputs: dict[str, Any],
        module_inputs: Any,
        module_outputs: Any,
        captured_trace: Any,
    ) -> dict[str, Any]:
        del predictor_inputs, captured_trace
        pred = dspy.Prediction(response=predictor_output["response"])
        scored = _metric_with_feedback(module_inputs, pred, None)
        final_score = _metric_with_feedback(module_inputs, dspy.Prediction(**module_outputs), None).score
        return {"feedback_score": float(final_score), "feedback_text": str(scored.feedback)}

    def _feedback_for_ensure_module(
        predictor_output: dict[str, Any],
        predictor_inputs: dict[str, Any],
        module_inputs: Any,
        module_outputs: Any,
        captured_trace: Any,
    ) -> dict[str, Any]:
        del predictor_inputs, captured_trace
        pred = dspy.Prediction(response=predictor_output["final_response"])
        scored = _metric_with_feedback(module_inputs, pred, None)
        final_score = _metric_with_feedback(module_inputs, dspy.Prediction(**module_outputs), None).score
        return {"feedback_score": float(final_score), "feedback_text": str(scored.feedback)}

    return {
        "generate_response_module.predict": _feedback_for_response_module,
        "ensure_correct_response_module.predict": _feedback_for_ensure_module,
    }


def build_spec() -> BenchmarkSpec:
    return BenchmarkSpec(
        key="ifbench",
        name="IFBench",
        description="Instruction-following benchmark",
        invocation_cap=OFFICIAL_INVOCATION_CAPS["ifbench"],
        dataset_loader=_load_splits,
        program_factory=_program_factory,
        metric=_metric,
        metric_with_feedback=_metric_with_feedback,
        feedback_map_factory=_feedback_map_factory,
        retrieval_required=False,
    )
