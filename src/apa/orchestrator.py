from __future__ import annotations

import dataclasses
import datetime as dt
import statistics
from pathlib import Path
from typing import Any, Iterable, Iterator

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tqdm.auto import tqdm

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


CONSOLE = Console()


@dataclasses.dataclass(slots=True)
class MatrixRun:
    run_dir: Path
    results: list[RunResult]
    stopped_by_cost_cap: bool
    total_cost_usd: float


def run_single(
    benchmark: str,
    method: str,
    runtime: RuntimeConfig,
    *,
    run_dir: Path | None = None,
    show_progress: bool = True,
) -> RunResult:
    import dspy

    spec = get_benchmark_spec(benchmark)
    invocation_cap = runtime.invocation_cap or spec.invocation_cap

    if run_dir is None:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("runs") / f"{ts}_{spec.key}_{method}_seed{runtime.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    runtime = dataclasses.replace(runtime, invocation_cap=invocation_cap, output_dir=run_dir)
    seed_everything(runtime.seed)

    if show_progress:
        _print_run_header(spec.key, method, runtime, run_dir)
        CONSOLE.log("[cyan]Initializing task/reflection models[/cyan]")

    task_lm = dspy.LM(runtime.model, temperature=0.0, cache=True)
    reflection_lm = dspy.LM(runtime.model, temperature=1.0, cache=True)
    dspy.configure(lm=task_lm)

    if show_progress:
        CONSOLE.log("[cyan]Loading benchmark dataset splits[/cyan]")
    splits = spec.dataset_loader(runtime)

    retriever, canonical_mode, retrieval_backend = _resolve_retriever(spec)
    if show_progress:
        CONSOLE.log(
            "[cyan]Retrieval backend:[/cyan] "
            f"{retrieval_backend} (canonical={canonical_mode})"
        )
        CONSOLE.log("[cyan]Building student program[/cyan]")

    student = spec.program_factory(runtime, retrieval=retriever)

    runner = get_method_runner(method)
    if show_progress:
        CONSOLE.log("[cyan]Compiling method[/cyan]")

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

        test_examples = splits.test[:max_test_examples]
        iterator = _tqdm_iter(
            test_examples,
            enabled=show_progress,
            desc=f"{spec.key}/{method} test",
            unit="ex",
        )

        for idx, example in enumerate(iterator):
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

            if show_progress and hasattr(iterator, "set_postfix"):
                running_mean = float(statistics.fmean(scores)) if scores else 0.0
                iterator.set_postfix(avg=f"{running_mean:.4f}")
    elif show_progress:
        CONSOLE.log("[yellow]Dry run enabled: skipping evaluation loop[/yellow]")

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
    if show_progress:
        _print_run_result(run_result)

    return run_result


def run_matrix(
    *,
    benchmarks: list[str],
    methods: list[str],
    runtime: RuntimeConfig,
    cost_cap_usd: float = DEFAULT_COST_CAP_USD,
    root_dir: Path | None = None,
    show_progress: bool = True,
) -> MatrixRun:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = root_dir or (Path("runs") / f"{ts}_matrix_seed{runtime.seed}")
    run_root.mkdir(parents=True, exist_ok=True)

    total_cost = 0.0
    stopped_by_cost_cap = False
    results: list[RunResult] = []

    planned_runs = len(benchmarks) * len(methods)
    matrix_bar = tqdm(
        total=planned_runs,
        desc="matrix",
        unit="run",
        dynamic_ncols=True,
        disable=not show_progress,
    )

    try:
        for benchmark in benchmarks:
            for method in methods:
                if total_cost >= cost_cap_usd:
                    stopped_by_cost_cap = True
                    break

                target_dir = run_root / benchmark / method
                result = run_single(
                    benchmark=benchmark,
                    method=method,
                    runtime=runtime,
                    run_dir=target_dir,
                    show_progress=show_progress,
                )
                results.append(result)
                total_cost += result.cost.usd_cost

                matrix_bar.update(1)
                matrix_bar.set_postfix(cost=f"${total_cost:.2f}", latest=f"{benchmark}/{method}")

            if stopped_by_cost_cap:
                break
    finally:
        matrix_bar.close()

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

    matrix = MatrixRun(
        run_dir=run_root,
        results=results,
        stopped_by_cost_cap=stopped_by_cost_cap,
        total_cost_usd=total_cost,
    )
    if show_progress:
        _print_matrix_result(matrix)

    return matrix


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


def _tqdm_iter(
    rows: Iterable[Any],
    *,
    enabled: bool,
    desc: str,
    unit: str,
) -> Iterator[Any]:
    row_list = list(rows)
    if not enabled:
        return iter(row_list)
    return iter(tqdm(row_list, desc=desc, unit=unit, dynamic_ncols=True))


def _print_run_header(benchmark: str, method: str, runtime: RuntimeConfig, run_dir: Path) -> None:
    body = (
        f"benchmark=[bold]{benchmark}[/bold]  method=[bold]{method}[/bold]\n"
        f"model={runtime.model}\n"
        f"seed={runtime.seed}  mode={runtime.dataset_mode}  dry_run={runtime.dry_run}\n"
        f"output={run_dir}"
    )
    CONSOLE.print(Panel.fit(body, title="Run Started", border_style="cyan"))


def _print_run_result(run_result: RunResult) -> None:
    table = Table(title="Run Summary")
    table.add_column("Field")
    table.add_column("Value", justify="right")
    table.add_row("benchmark", run_result.benchmark)
    table.add_row("method", run_result.method)
    table.add_row("score", f"{run_result.score:.4f}")
    table.add_row("invocations", str(run_result.invocation_count))
    table.add_row("cost_usd", f"{run_result.cost.usd_cost:.6f}")
    table.add_row("canonical", str(run_result.canonical_mode))
    table.add_row("retrieval", run_result.retrieval_backend)
    CONSOLE.print(table)


def _print_matrix_result(matrix: MatrixRun) -> None:
    table = Table(title="Matrix Summary")
    table.add_column("Benchmark")
    table.add_column("Method")
    table.add_column("Score", justify="right")
    table.add_column("Cost (USD)", justify="right")

    for result in matrix.results:
        table.add_row(
            result.benchmark,
            result.method,
            f"{result.score:.4f}",
            f"{result.cost.usd_cost:.6f}",
        )

    CONSOLE.print(table)
    CONSOLE.print(
        Panel.fit(
            f"run_dir={matrix.run_dir}\n"
            f"total_cost_usd={matrix.total_cost_usd:.6f}\n"
            f"stopped_by_cost_cap={matrix.stopped_by_cost_cap}",
            title="Matrix Completed",
            border_style="green",
        )
    )
