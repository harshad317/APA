from __future__ import annotations

from typing import Any

from .base import RetrievalResult


class FallbackRetriever:
    backend_name = "fallback"

    def search(self, query: str, k: int, example: Any | None = None) -> RetrievalResult:
        docs = _docs_from_example(example)
        if not docs:
            docs = [f"fallback | No canonical retrieval available for query: {query}"]
        return RetrievalResult(passages=docs[:k], canonical=False, backend=self.backend_name)


def _docs_from_example(example: Any | None) -> list[str]:
    if example is None:
        return []

    docs: list[str] = []

    if isinstance(example, dict):
        context = example.get("context")
        supporting = example.get("supporting_facts")
    else:
        context = getattr(example, "context", None)
        supporting = getattr(example, "supporting_facts", None)

    if isinstance(context, dict):
        titles = context.get("title") or []
        sentences = context.get("sentences") or []
        for title, sent_list in zip(titles, sentences):
            text = " ".join(sent_list) if isinstance(sent_list, list) else str(sent_list)
            docs.append(f"{title} | {text}")

    if isinstance(supporting, list):
        for item in supporting:
            if isinstance(item, dict) and "key" in item:
                docs.append(f"{item['key']} | supporting fact")
            elif isinstance(item, dict) and "title" in item:
                docs.append(f"{item['title']} | supporting fact")
            elif isinstance(item, str):
                docs.append(f"{item} | supporting fact")

    # De-duplicate while preserving order
    seen = set()
    unique_docs = []
    for doc in docs:
        if doc not in seen:
            seen.add(doc)
            unique_docs.append(doc)

    return unique_docs
