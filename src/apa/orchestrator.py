from __future__ import annotations

import dataclasses
import datetime as dt
import statistics
from pathlib import Path
from typing import Any

from apa.benchmarks.registry import get_benchmark_spec
from apa.constants import DEFAULT_COST_CAP_USD
from apa.methods.registry import get_method_runner
from apa.retrieval.bm25_wiki import BM25WikiRetriever
from apa.retrieval.fallback import FallbackRetriever
from apa.types import CostSummary, RunResult, RuntimeConfig
from apa.utils import (
    append_jsonl,
    collect_lms,
    estimate_lm_cost,
    extract_program_prompts,
    prediction_to_dict,
    seed_everything,
    write_json,
)


@dataclasses.dataclass(slots=True)
class MatrixRun:
    run_dir: Path
    results: list[RunResult]
    stopped_by_cost_cap: bool
    total_cost_usd: float


def run_single(benchmark: str, method: str, runtime: RuntimeConfig, *, run_dir: Path | None = None) -> RunResult:
    import dspy

    spec = get_benchmark_spec(benchmark)
    invocation_cap = runtime.invocation_cap or spec.invocation_cap

    if run_dir is None:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("runs") / f"{ts}_{spec.key}_{method}_seed{runtime.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    runtime = dataclasses.replace(runtime, invocation_cap=invocation_cap, output_dir=run_dir)
    seed_everything(runtime.seed)

    task_lm = dspy.LM(runtime.model, temperature=0.0, cache=True)
    reflection_lm = dspy.LM(runtime.model, temperature=1.0, cache=True)
    dspy.configure(lm=task_lm)

    splits = spec.dataset_loader(runtime)
    retriever, canonical_mode, retrieval_backend = _resolve_retriever(spec)
    student = spec.program_factory(runtime, retrieval=retriever)

    runner = get_method_runner(method)
    compile_result = runner.compile(
        spec=spec,
        student=student,
        trainset=splits.train,
        valset=splits.val,
        runtime=runtime,
        task_lm=task_lm,
        reflection_lm=reflection_lm,
    )

    scores: list[float] = []
    prediction_rows: list[dict[str, Any]] = []

    if not runtime.dry_run:
        max_test_examples = len(splits.test)
        if invocation_cap is not None:
            max_test_examples = min(max_test_examples, invocation_cap)

        for idx, example in enumerate(splits.test[:max_test_examples]):
            pred = runner.predict(compile_result.program, example)
            score = float(spec.metric(example, pred, trace=None))
            scores.append(score)

            prediction_rows.append(
                {
                    "index": idx,
                    "inputs": dict(example.inputs()) if hasattr(example, "inputs") else dict(example),
                    "prediction": prediction_to_dict(pred),
                    "score": score,
                }
            )

    score_mean = float(statistics.fmean(scores)) if scores else 0.0

    all_lms = _gather_cost_lms(task_lm, reflection_lm, compile_result.program, spec)
    cost = _sum_costs(estimate_lm_cost(lm) for lm in all_lms)

    run_result = RunResult(
        benchmark=spec.key,
        method=method.lower(),
        seed=runtime.seed,
        score=score_mean,
        canonical_mode=canonical_mode,
        retrieval_backend=retrieval_backend,
        invocation_count=int(compile_result.compile_invocations + len(scores)),
        cost=cost,
        artifacts={
            "run_dir": str(run_dir),
            "compile": compile_result.artifacts,
            "prompts": extract_program_prompts(compile_result.program),
            "num_test_examples": len(scores),
        },
    )

    _write_run_artifacts(run_dir, runtime, spec, run_result, prediction_rows)
    return run_result


def run_matrix(
    *,
    benchmarks: list[str],
    methods: list[str],
    runtime: RuntimeConfig,
    cost_cap_usd: float = DEFAULT_COST_CAP_USD,
    root_dir: Path | None = None,
) -> MatrixRun:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = root_dir or (Path("runs") / f"{ts}_matrix_seed{runtime.seed}")
    run_root.mkdir(parents=True, exist_ok=True)

    total_cost = 0.0
    stopped_by_cost_cap = False
    results: list[RunResult] = []

    for benchmark in benchmarks:
        for method in methods:
            if total_cost >= cost_cap_usd:
                stopped_by_cost_cap = True
                break

            target_dir = run_root / benchmark / method
            result = run_single(benchmark=benchmark, method=method, runtime=runtime, run_dir=target_dir)
            results.append(result)
            total_cost += result.cost.usd_cost

        if stopped_by_cost_cap:
            break

    summary = {
        "run_dir": str(run_root),
        "seed": runtime.seed,
        "model": runtime.model,
        "cost_cap_usd": cost_cap_usd,
        "total_cost_usd": total_cost,
        "stopped_by_cost_cap": stopped_by_cost_cap,
        "results": [dataclasses.asdict(r) for r in results],
    }
    write_json(run_root / "matrix_summary.json", summary)

    return MatrixRun(
        run_dir=run_root,
        results=results,
        stopped_by_cost_cap=stopped_by_cost_cap,
        total_cost_usd=total_cost,
    )


def _resolve_retriever(spec) -> tuple[Any | None, bool, str]:
    if not spec.retrieval_required:
        return None, True, "none"

    canonical = BM25WikiRetriever()
    if canonical.available():
        return canonical, True, canonical.backend_name

    fallback = FallbackRetriever()
    return fallback, False, fallback.backend_name


def _gather_cost_lms(task_lm: Any, reflection_lm: Any, compiled_program: Any, spec: Any) -> list[Any]:
    lms: list[Any] = [task_lm, reflection_lm]
    lms.extend(collect_lms(compiled_program))

    collector = spec.metadata.get("extra_lm_collectors") if isinstance(spec.metadata, dict) else None
    if callable(collector):
        try:
            extras = collector()
            if isinstance(extras, list):
                lms.extend(extras)
        except Exception:
            pass

    unique: list[Any] = []
    seen = set()
    for lm in lms:
        lm_id = id(lm)
        if lm_id not in seen:
            seen.add(lm_id)
            unique.append(lm)
    return unique


def _sum_costs(items: Any) -> CostSummary:
    summary = CostSummary()
    for item in items:
        if not isinstance(item, CostSummary):
            continue
        summary.input_tokens += int(item.input_tokens)
        summary.output_tokens += int(item.output_tokens)
        summary.usd_cost += float(item.usd_cost)
    return summary


def _write_run_artifacts(
    run_dir: Path,
    runtime: RuntimeConfig,
    spec: Any,
    run_result: RunResult,
    prediction_rows: list[dict[str, Any]],
) -> None:
    write_json(
        run_dir / "config.json",
        {
            "runtime": dataclasses.asdict(runtime),
            "benchmark": spec.key,
            "method": run_result.method,
            "invocation_cap": runtime.invocation_cap,
        },
    )
    write_json(run_dir / "metrics.json", dataclasses.asdict(run_result))
    write_json(run_dir / "compile_artifacts.json", run_result.artifacts.get("compile", {}))

    predictions_path = run_dir / "predictions.jsonl"
    predictions_path.unlink(missing_ok=True)
    append_jsonl(predictions_path, prediction_rows)

    write_json(run_dir / "run_log.json", {"timestamp": dt.datetime.now().isoformat(), "score": run_result.score})
