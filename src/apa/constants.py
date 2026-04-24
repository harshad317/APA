"""Global constants for APA benchmarking."""

from __future__ import annotations

DEFAULT_MODEL = "openai/gpt-4.1-mini-2025-04-14"
DEFAULT_SEED = 0
DEFAULT_COST_CAP_USD = 500.0

OFFICIAL_INVOCATION_CAPS: dict[str, int] = {
    "hotpotqa": 6871,
    "hover": 7051,
    "pupa": 2426,
    "ifbench": 3593,
    "livebench_math": 1839,
    "aime": 1839,
}

GEPA6_BENCHMARK_ORDER = [
    "hotpotqa",
    "hover",
    "pupa",
    "ifbench",
    "livebench_math",
    "aime",
]

DEFAULT_METHODS = ["apa", "gepa", "miprov2"]
