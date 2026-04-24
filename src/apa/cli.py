from __future__ import annotations

import dataclasses
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from apa.benchmarks.registry import list_benchmark_keys
from apa.constants import (
    DEFAULT_COST_CAP_USD,
    DEFAULT_METHODS,
    DEFAULT_MODEL,
    DEFAULT_SEED,
    GEPA6_BENCHMARK_ORDER,
)
from apa.methods.registry import list_methods
from apa.orchestrator import run_matrix, run_single
from apa.reporting import generate_aggregate_report
from apa.types import RuntimeConfig

console = Console()
app = typer.Typer(help="Adaptive Prompt Automaton benchmark CLI")
benchmarks_app = typer.Typer(help="Benchmark operations")
report_app = typer.Typer(help="Reporting operations")

app.add_typer(benchmarks_app, name="benchmarks")
app.add_typer(report_app, name="report")


def _parse_csv(raw: str, *, allowed: list[str], label: str) -> list[str]:
    items = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if not items:
        raise typer.BadParameter(f"No {label} provided")
    invalid = [x for x in items if x not in allowed]
    if invalid:
        raise typer.BadParameter(f"Invalid {label}: {', '.join(invalid)}. Allowed: {', '.join(allowed)}")
    return items


@benchmarks_app.command("list")
def benchmarks_list() -> None:
    table = Table(title="Available Benchmarks")
    table.add_column("Key")
    for key in list_benchmark_keys():
        table.add_row(key)
    console.print(table)


@app.command("run")
def run_command(
    benchmark: str = typer.Option(..., help="Benchmark key"),
    method: str = typer.Option("apa", help="Method: apa|gepa|miprov2"),
    seed: int = typer.Option(DEFAULT_SEED, help="Random seed"),
    model: str = typer.Option(DEFAULT_MODEL, help="Pinned model id"),
    dataset_mode: str = typer.Option("full", help="smoke|tiny|test|full"),
    invocation_cap: int | None = typer.Option(None, help="Override invocation cap"),
    output_dir: Path | None = typer.Option(None, help="Output run directory"),
    smoke: bool = typer.Option(False, help="Shortcut for smoke dataset mode"),
    dry_run: bool = typer.Option(False, help="Skip optimization/evaluation calls"),
    quiet: bool = typer.Option(False, "--quiet", help="Disable real-time progress output"),
) -> None:
    benchmark = benchmark.lower()
    method = method.lower()

    if benchmark not in list_benchmark_keys():
        raise typer.BadParameter(f"Unknown benchmark '{benchmark}'.")
    if method not in list_methods():
        raise typer.BadParameter(f"Unknown method '{method}'.")

    runtime = RuntimeConfig(
        model=model,
        seed=seed,
        smoke=smoke,
        dry_run=dry_run,
        dataset_mode=dataset_mode,
        invocation_cap=invocation_cap,
        output_dir=output_dir,
    )

    result = run_single(
        benchmark=benchmark,
        method=method,
        runtime=runtime,
        run_dir=output_dir,
        show_progress=not quiet,
    )
    console.print_json(data=dataclasses.asdict(result))


@app.command("run-all")
def run_all_command(
    seed: int = typer.Option(DEFAULT_SEED, help="Random seed"),
    model: str = typer.Option(DEFAULT_MODEL, help="Pinned model id"),
    dataset_mode: str = typer.Option("full", help="smoke|tiny|test|full"),
    cost_cap_usd: float = typer.Option(DEFAULT_COST_CAP_USD, help="Global cost cap for matrix run"),
    benchmarks: str = typer.Option(
        ",".join(GEPA6_BENCHMARK_ORDER),
        help="Comma separated benchmark keys",
    ),
    methods: str = typer.Option(
        ",".join(DEFAULT_METHODS),
        help="Comma separated method keys",
    ),
    invocation_cap: int | None = typer.Option(None, help="Override invocation cap for all runs"),
    output_dir: Path | None = typer.Option(None, help="Root output directory for matrix run"),
    smoke: bool = typer.Option(False, help="Shortcut for smoke dataset mode"),
    dry_run: bool = typer.Option(False, help="Skip optimization/evaluation calls"),
    quiet: bool = typer.Option(False, "--quiet", help="Disable real-time progress output"),
) -> None:
    benchmark_keys = _parse_csv(benchmarks, allowed=list_benchmark_keys(), label="benchmarks")
    method_keys = _parse_csv(methods, allowed=list_methods(), label="methods")

    runtime = RuntimeConfig(
        model=model,
        seed=seed,
        smoke=smoke,
        dry_run=dry_run,
        dataset_mode=dataset_mode,
        invocation_cap=invocation_cap,
        output_dir=None,
    )

    matrix = run_matrix(
        benchmarks=benchmark_keys,
        methods=method_keys,
        runtime=runtime,
        cost_cap_usd=cost_cap_usd,
        root_dir=output_dir,
        show_progress=not quiet,
    )

    payload = {
        "run_dir": str(matrix.run_dir),
        "stopped_by_cost_cap": matrix.stopped_by_cost_cap,
        "total_cost_usd": matrix.total_cost_usd,
        "results": [dataclasses.asdict(r) for r in matrix.results],
    }
    console.print_json(data=payload)


@report_app.command("aggregate")
def report_aggregate(
    root: Path = typer.Option(Path("runs"), help="Runs root directory"),
    output_dir: Path | None = typer.Option(None, help="Directory to write aggregate report"),
) -> None:
    payload = generate_aggregate_report(root=root, output_dir=output_dir)
    console.print_json(data=payload)



def main() -> None:
    app()


if __name__ == "__main__":
    main()
