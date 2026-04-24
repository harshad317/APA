"""Benchmark adapters and registry."""

from .base import BenchmarkSpec, DatasetSplits
from .registry import get_benchmark_spec, list_benchmark_keys

__all__ = ["BenchmarkSpec", "DatasetSplits", "get_benchmark_spec", "list_benchmark_keys"]
