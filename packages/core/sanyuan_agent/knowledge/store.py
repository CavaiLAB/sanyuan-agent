"""KnowledgeBase storage abstraction + an in-memory cosine-search implementation."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class KnowledgeChunk:
    """A stored, embedded chunk belonging to a knowledge collection."""

    id: str
    collection_id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class KnowledgeBase(ABC):
    """Storage + similarity search over knowledge chunks, scoped by collection."""

    @abstractmethod
    def add(self, chunks: Iterable[KnowledgeChunk]) -> None: ...

    @abstractmethod
    def search(
        self, collection_id: str, query_embedding: Sequence[float], *, top_k: int = 5
    ) -> list[tuple[KnowledgeChunk, float]]: ...

    @abstractmethod
    def delete_collection(self, collection_id: str) -> int: ...


class InMemoryKnowledgeBase(KnowledgeBase):
    """Dev/test store — brute-force cosine search. Not for production scale."""

    def __init__(self) -> None:
        self._by_collection: dict[str, list[KnowledgeChunk]] = {}

    def add(self, chunks: Iterable[KnowledgeChunk]) -> None:
        for chunk in chunks:
            self._by_collection.setdefault(chunk.collection_id, []).append(chunk)

    def search(
        self, collection_id: str, query_embedding: Sequence[float], *, top_k: int = 5
    ) -> list[tuple[KnowledgeChunk, float]]:
        scored = [
            (chunk, _cosine(query_embedding, chunk.embedding))
            for chunk in self._by_collection.get(collection_id, [])
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    def delete_collection(self, collection_id: str) -> int:
        removed = self._by_collection.pop(collection_id, [])
        return len(removed)
