"""Long-term semantic memory — pgvector-backed storage with scope isolation."""

from __future__ import annotations

from typing import Any

from sanyuan_agent.memory.longterm.embedding import (
    IEmbeddingProvider,
    LocalEmbeddingProvider,
    OpenAIEmbeddingProvider,
    create_embedding_provider,
)
from sanyuan_agent.memory.longterm.store import LongtermMemoryStore, MemoryEntry

__all__ = [
    "IEmbeddingProvider",
    "LocalEmbeddingProvider",
    "LongtermMemoryStore",
    "MemoryEntry",
    "OpenAIEmbeddingProvider",
    "build_memory_tools",
    "create_embedding_provider",
]


def __getattr__(name: str) -> Any:
    """Load tool wiring lazily to keep ``sanyuan_agent.memory`` import-safe.

    ``sanyuan_agent.tools`` exposes the memory provider from its package catalog;
    importing the builder eagerly here would make those two public packages
    initialize each other. The old ``from ...longterm import build_memory_tools``
    API remains available through this lazy attribute.
    """
    if name == "build_memory_tools":
        from sanyuan_agent.memory.longterm.tools import build_memory_tools

        return build_memory_tools
    raise AttributeError(name)
