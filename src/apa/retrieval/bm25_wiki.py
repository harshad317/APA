from __future__ import annotations

import os
import tarfile
import threading
from pathlib import Path
from typing import Any

from .base import RetrievalResult


class BM25WikiRetriever:
    backend_name = "bm25_wiki2017"
    _init_lock = threading.Lock()

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or (Path.home() / ".cache" / "apa" / "wiki2017")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._ready = False
        self._stemmer = None
        self._bm25s = None
        self._retriever = None
        self._corpus: list[str] = []

    def available(self) -> bool:
        try:
            import bm25s  # noqa: F401
            import Stemmer  # noqa: F401
            import ujson  # noqa: F401
            return True
        except Exception:
            return False

    def search(self, query: str, k: int, example: Any | None = None) -> RetrievalResult:
        if not self.available():
            raise RuntimeError("BM25 dependencies are not installed. Install with: pip install -e '.[retrieval]'")
        self._ensure_ready()
        tokens = self._bm25s.tokenize(query, stopwords="en", stemmer=self._stemmer, show_progress=False)
        indices, scores = self._retriever.retrieve(tokens, k=k, n_threads=1, show_progress=False)
        passages = [self._corpus[idx] for idx in indices[0]]
        _ = scores  # kept for future diagnostics
        return RetrievalResult(passages=passages, canonical=True, backend=self.backend_name)

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        with self._init_lock:
            if self._ready:
                return

            import bm25s
            import Stemmer
            import ujson

            corpus_file = self.cache_dir / "wiki.abstracts.2017.jsonl"
            index_dir = self.cache_dir / "bm25s_retriever"
            tar_path = self.cache_dir / "wiki.abstracts.2017.tar.gz"

            if not corpus_file.exists() or not index_dir.exists():
                self._download_wiki_tarball(tar_path)
                with tarfile.open(tar_path, "r:gz") as tar:
                    tar.extractall(path=self.cache_dir)

            corpus: list[str] = []
            with corpus_file.open("r", encoding="utf-8") as f:
                for line in f:
                    row = ujson.loads(line)
                    corpus.append(f"{row['title']} | {' '.join(row['text'])}")

            stemmer = Stemmer.Stemmer("english")
            if index_dir.exists():
                retriever = bm25s.BM25.load(index_dir)
            else:
                corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer, show_progress=False)
                retriever = bm25s.BM25(k1=0.9, b=0.4)
                retriever.index(corpus_tokens)
                retriever.save(index_dir)

            self._bm25s = bm25s
            self._stemmer = stemmer
            self._retriever = retriever
            self._corpus = corpus
            self._ready = True

    def _download_wiki_tarball(self, target_path: Path) -> None:
        if target_path.exists():
            return
        try:
            from dspy.utils import download

            cwd = os.getcwd()
            os.chdir(self.cache_dir)
            try:
                download("https://huggingface.co/dspy/cache/resolve/main/wiki.abstracts.2017.tar.gz")
            finally:
                os.chdir(cwd)
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            raise RuntimeError("Failed to download wiki.abstracts.2017.tar.gz") from exc
