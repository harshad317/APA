from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(slots=True)
class RetrievalResult:
    passages: list[str]
    canonical: bool
    backend: str


class Retriever(Protocol):
    backend_name: str

    def search(self, query: str, k: int, example: Any | None = None) -> RetrievalResult:
        ...
