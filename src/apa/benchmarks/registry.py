from __future__ import annotations

from apa.benchmarks.base import BenchmarkSpec

from .aime import build_spec as build_aime
from .ifbench import build_spec as build_ifbench
from .hotpotqa import build_spec as build_hotpotqa
from .hover import build_spec as build_hover
from .livebench_math import build_spec as build_livebench_math
from .pupa import build_spec as build_pupa


SPEC_BUILDERS = {
    "aime": build_aime,
    "livebench_math": build_livebench_math,
    "ifbench": build_ifbench,
    "pupa": build_pupa,
    "hover": build_hover,
    "hotpotqa": build_hotpotqa,
}


def list_benchmark_keys() -> list[str]:
    return sorted(SPEC_BUILDERS.keys())


def get_benchmark_spec(key: str) -> BenchmarkSpec:
    normalized = key.lower()
    if normalized not in SPEC_BUILDERS:
        raise KeyError(f"Unknown benchmark '{key}'. Available: {', '.join(list_benchmark_keys())}")
    return SPEC_BUILDERS[normalized]()
