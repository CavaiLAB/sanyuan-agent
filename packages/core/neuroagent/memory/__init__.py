"""Agent memory layers — Sensory (raw) + Working (structured) + Episodic + Long-term (pgvector)."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from neuroagent.memory.context import (
    MemoryContext,
    MemoryLevel,
    memory_visible_in_context,
    validate_memory_context,
)
from neuroagent.memory.projection import (
    MemoryProjection,
    MemoryProjectionEntry,
    ProjectableMemory,
    materialize_memory_markdown,
    render_memory_markdown,
    stable_memory_token,
)

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "IEpisodicStore": ("neuroagent.memory.abc", "IEpisodicStore"),
    "IMemoryStore": ("neuroagent.memory.abc", "IMemoryStore"),
    "InMemoryStore": ("neuroagent.memory.abc", "InMemoryStore"),
    "Episode": ("neuroagent.memory.episodic", "Episode"),
    "EpisodicMemory": ("neuroagent.memory.episodic", "EpisodicMemory"),
    "IEmbeddingProvider": ("neuroagent.memory.longterm", "IEmbeddingProvider"),
    "LocalEmbeddingProvider": ("neuroagent.memory.longterm", "LocalEmbeddingProvider"),
    "LongtermMemoryStore": ("neuroagent.memory.longterm", "LongtermMemoryStore"),
    "MemoryEntry": ("neuroagent.memory.longterm", "MemoryEntry"),
    "OpenAIEmbeddingProvider": ("neuroagent.memory.longterm", "OpenAIEmbeddingProvider"),
    "create_embedding_provider": ("neuroagent.memory.longterm", "create_embedding_provider"),
    "ScopeKind": ("neuroagent.memory.longterm.store", "ScopeKind"),
    "EpisodicRecall": ("neuroagent.memory.recall", "EpisodicRecall"),
    "SensoryEvent": ("neuroagent.memory.sensory", "SensoryEvent"),
    "SensoryMemory": ("neuroagent.memory.sensory", "SensoryMemory"),
    "VectorRecall": ("neuroagent.memory.vector", "VectorRecall"),
    "WorkingMemory": ("neuroagent.memory.working", "WorkingMemory"),
    "WorkingStep": ("neuroagent.memory.working", "WorkingStep"),
}


def __getattr__(name: str) -> Any:
    """Keep lightweight primitives importable without loading every backend."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value

__all__ = [
    "Episode",
    "EpisodicMemory",
    "EpisodicRecall",
    "IEmbeddingProvider",
    "IEpisodicStore",
    "IMemoryStore",
    "InMemoryStore",
    "LocalEmbeddingProvider",
    "LongtermMemoryStore",
    "MemoryContext",
    "MemoryEntry",
    "MemoryLevel",
    "MemoryProjection",
    "MemoryProjectionEntry",
    "OpenAIEmbeddingProvider",
    "ProjectableMemory",
    "ScopeKind",
    "SensoryEvent",
    "SensoryMemory",
    "VectorRecall",
    "WorkingMemory",
    "WorkingStep",
    "create_embedding_provider",
    "materialize_memory_markdown",
    "memory_visible_in_context",
    "render_memory_markdown",
    "stable_memory_token",
    "validate_memory_context",
]
