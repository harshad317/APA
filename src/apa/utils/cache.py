from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class CacheEntry:
    score: float
    prediction: dict[str, Any]


class EvaluationCache:
    """Simple disk-backed cache hook for expensive per-example evaluations."""

    def __init__(self, cache_path: Path | None = None) -> None:
        self._mem: dict[str, CacheEntry] = {}
        self.cache_path = cache_path
        if self.cache_path and self.cache_path.exists():
            try:
                data = json.loads(self.cache_path.read_text(encoding="utf-8"))
                for key, value in data.items():
                    self._mem[key] = CacheEntry(score=float(value["score"]), prediction=dict(value["prediction"]))
            except Exception:
                self._mem = {}

    @staticmethod
    def make_key(payload: dict[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> CacheEntry | None:
        return self._mem.get(key)

    def set(self, key: str, score: float, prediction: dict[str, Any]) -> None:
        self._mem[key] = CacheEntry(score=score, prediction=prediction)

    def persist(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            key: {"score": entry.score, "prediction": entry.prediction}
            for key, entry in self._mem.items()
        }
        self.cache_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
