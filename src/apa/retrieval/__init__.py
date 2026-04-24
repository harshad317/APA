"""Retrieval backends for multi-hop benchmarks."""

from .base import RetrievalResult, Retriever
from .bm25_wiki import BM25WikiRetriever
from .fallback import FallbackRetriever

__all__ = ["RetrievalResult", "Retriever", "BM25WikiRetriever", "FallbackRetriever"]
