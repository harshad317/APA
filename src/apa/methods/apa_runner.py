from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any

from apa.benchmarks.base import BenchmarkSpec
from apa.core.automaton import Automaton
from apa.core.executor import ExecutorConfig
from apa.core.optimizer import APAConfig, APAOptimizer
from apa.types import RuntimeConfig
from apa.utils import (
    EvaluationCache,
    call_program,
    extract_prediction_text,
    prediction_to_dict,
)

from .base import CompileResult


@dataclass(slots=True)
class APARunnerConfig:
    max_steps_per_episode: int = 3
    reflection_minibatch_size: int = 8


class APAMethodRunner:
    key = "apa"

    def __init__(self, config: APARunnerConfig | None = None) -> None:
        self.config = config or APARunnerConfig()

    def compile(
        self,
        *,
        spec: BenchmarkSpec,
        student: Any,
        trainset: list[Any],
        valset: list[Any],
        runtime: RuntimeConfig,
        task_lm: Any,
        reflection_lm: Any,
    ) -> CompileResult:
        del task_lm, reflection_lm

        if runtime.dry_run:
            return CompileResult(
                program={"base_program": student, "automaton": self._seed_automaton()},
                compile_invocations=0,
                artifacts={"mode": "dry_run"},
            )

        invocation_cap = runtime.invocation_cap or spec.invocation_cap
        cache_path = None
        if runtime.output_dir:
            cache_path = runtime.output_dir / "cache" / "apa_eval_cache.json"
        cache = EvaluationCache(cache_path=cache_path)

        optimizer = APAOptimizer(
            APAConfig(
                model_id=runtime.model,
                seed=runtime.seed,
                max_metric_calls=invocation_cap,
                reflection_minibatch_size=self.config.reflection_minibatch_size,
                max_steps_per_episode=self.config.max_steps_per_episode,
            )
        )

        def evaluate_episode(automaton: Automaton, example: Any, ex_idx: int):
            cache_key = self._cache_key(automaton, example, ex_idx)
            cached = cache.get(cache_key)
            if cached is not None:
                payload = cached.prediction
                return float(cached.score), dict(payload.get("features", {})), list(payload.get("path", []))

            pred, path, features = self._rollout(automaton, student, example)
            score = float(spec.metric(example, pred, trace=None))

            cache.set(
                cache_key,
                score=score,
                prediction={
                    "prediction": prediction_to_dict(pred),
                    "path": path,
                    "features": features,
                },
            )
            return score, features, path

        candidate = optimizer.compile(trainset=trainset, valset=valset, evaluate_episode=evaluate_episode)
        cache.persist()

        artifacts = {
            "train_score": candidate.train_score,
            "val_score": candidate.val_score,
            "metric_calls": candidate.metric_calls,
            "states": {sid: asdict(state) for sid, state in candidate.automaton.states.items()},
            "transitions": [asdict(t) for t in candidate.automaton.transitions],
            "traces": candidate.traces,
        }

        return CompileResult(
            program={"base_program": student, "automaton": candidate.automaton},
            compile_invocations=int(candidate.metric_calls),
            artifacts=artifacts,
        )

    def predict(self, compiled_program: Any, example: Any) -> Any:
        automaton = compiled_program["automaton"]
        student = compiled_program["base_program"]
        pred, _path, _features = self._rollout(automaton, student, example)
        return pred

    def _rollout(self, automaton: Automaton, student: Any, example: Any) -> tuple[Any, list[str], dict[str, Any]]:
        from apa.core.executor import AutomatonExecutor

        executor = AutomatonExecutor(automaton=automaton, config=ExecutorConfig(max_steps=self.config.max_steps_per_episode))
        state: dict[str, Any] = {"last_prediction": None, "last_quality": 0.0}

        def evaluate_step(prompt_fragment: str, context: dict[str, Any], step_index: int):
            del context
            pred = self._predict_with_prefix(student, example, prompt_fragment, step_index)
            text = extract_prediction_text(pred)
            quality = self._quality_proxy(text)
            state["last_prediction"] = pred
            state["last_quality"] = quality
            features = {
                "quality_proxy": quality,
                "text_len": float(len(text)),
                "non_empty": 1.0 if text.strip() else 0.0,
            }
            return quality, features

        trace = executor.run(context={}, evaluate_step=evaluate_step)
        prediction = state["last_prediction"]
        if prediction is None:
            prediction = self._predict_with_prefix(student, example, automaton.states[automaton.start_state].template, 0)

        features = {
            "quality_proxy": state["last_quality"],
            "steps": len(trace.steps),
        }
        return prediction, trace.path, features

    def _predict_with_prefix(self, student: Any, example: Any, prompt_fragment: str, step_index: int) -> Any:
        del step_index

        raw_inputs: dict[str, Any]
        if hasattr(example, "inputs"):
            raw_inputs = dict(example.inputs())
        elif isinstance(example, dict):
            raw_inputs = dict(example)
        else:
            raw_inputs = {}

        prefixed_inputs: dict[str, Any] = {}
        for key, value in raw_inputs.items():
            if isinstance(value, str):
                prefixed_inputs[key] = f"{prompt_fragment}\n\n{value}"
            else:
                prefixed_inputs[key] = value

        return call_program(student, example, override_inputs=prefixed_inputs)

    @staticmethod
    def _quality_proxy(text: str) -> float:
        cleaned = text.strip()
        if not cleaned:
            return 0.0

        proxy = min(1.0, max(0.15, len(cleaned) / 160.0))
        lowered = cleaned.lower()
        penalties = ["i don't know", "cannot answer", "unsure", "not enough information"]
        if any(token in lowered for token in penalties):
            proxy *= 0.5
        return float(max(0.0, min(1.0, proxy)))

    @staticmethod
    def _cache_key(automaton: Automaton, example: Any, ex_idx: int) -> str:
        payload = {
            "automaton": {
                "start_state": automaton.start_state,
                "states": {k: asdict(v) for k, v in automaton.states.items()},
                "transitions": [asdict(t) for t in automaton.transitions],
            },
            "example": dict(example) if hasattr(example, "items") else str(example),
            "input_keys": sorted(list(getattr(example, "_input_keys", []))),
            "idx": ex_idx,
        }
        raw = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
        return sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _seed_automaton() -> Automaton:
        # Used only for dry-run placeholders.
        return APAOptimizer(APAConfig())._seed_automaton()  # noqa: SLF001
