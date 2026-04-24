# APA

Clean-room implementation of **Adaptive Prompt Automaton (APA)** with official baselines and benchmark harness.

## Scope

- Implements APA from scratch (`src/apa/core/*`, `src/apa/methods/apa_runner.py`).
- Integrates official pip baselines:
  - `dspy.MIPROv2(auto="heavy")`
  - `dspy.GEPA(...)` (backed by pip package `gepa`)
- Benchmarks the GEPA-6 matrix:
  - `hotpotqa`, `hover`, `pupa`, `ifbench`, `livebench_math`, `aime`
- Pins model defaults to `openai/gpt-4.1-mini-2025-04-14`.
- Applies official benchmark invocation caps by default.
- Enforces a default matrix cost guardrail of `$500` in `run-all`.

## Install

Core install:

```bash
pip install -e .
```

Full benchmark extras:

```bash
pip install -e '.[retrieval,ifbench,livebench_math,dev]'
```

## CLI

List benchmarks:

```bash
apa benchmarks list
```

Single run:

```bash
apa run --benchmark aime --method apa --seed 0
apa run --benchmark hover --method gepa --seed 0
apa run --benchmark pupa --method miprov2 --seed 0
```

Full matrix:

```bash
apa run-all --seed 0
```

Aggregate reports:

```bash
apa report aggregate
```

## Runtime policy

- All methods use the same model id unless you override `--model`.
- Retrieval benchmarks (`hotpotqa`, `hover`) try canonical BM25 wiki retrieval first.
- If canonical retrieval deps are unavailable, fallback retrieval is used and marked as non-canonical in run metadata.

## Artifacts

Each run writes:

- `config.json`
- `metrics.json`
- `compile_artifacts.json`
- `predictions.jsonl`
- `run_log.json`

Matrix runs also write `matrix_summary.json`. Aggregate reports write:

- `aggregate_report.json`
- `aggregate_report.md`

## Environment

Set credentials before real runs:

```bash
export OPENAI_API_KEY=...
```
