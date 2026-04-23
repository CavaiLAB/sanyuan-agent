"""Long-term semantic memory — pgvector-backed storage with scope isolation."""

from neuroagent.memory.longterm.embedding import (
    IEmbeddingProvider,
    LocalEmbeddingProvider,
    OpenAIEmbeddingProvider,
    create_embedding_provider,
)
from neuroagent.memory.longterm.store import LongtermMemoryStore, MemoryEntry
from neuroagent.memory.longterm.tools import build_memory_tools

__all__ = [
    "IEmbeddingProvider",
    "LocalEmbeddingProvider",
    "LongtermMemoryStore",
    "MemoryEntry",
    "OpenAIEmbeddingProvider",
    "build_memory_tools",
    "create_embedding_provider",
]
