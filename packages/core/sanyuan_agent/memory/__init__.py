"""Agent memory layers — Sensory (raw) + Working (structured) + Episodic + Long-term (pgvector)."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from sanyuan_agent.memory.context import (
    MemoryContext,
    MemoryLevel,
    memory_visible_in_context,
    validate_memory_context,
)
from sanyuan_agent.memory.projection import (
    MemoryProjection,
    MemoryProjectionEntry,
    ProjectableMemory,
    materialize_memory_markdown,
    render_memory_markdown,
    stable_memory_token,
)

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "IEpisodicStore": ("sanyuan_agent.memory.abc", "IEpisodicStore"),
    "IMemoryStore": ("sanyuan_agent.memory.abc", "IMemoryStore"),
    "InMemoryStore": ("sanyuan_agent.memory.abc", "InMemoryStore"),
    "Episode": ("sanyuan_agent.memory.episodic", "Episode"),
    "EpisodicMemory": ("sanyuan_agent.memory.episodic", "EpisodicMemory"),
    "IEmbeddingProvider": ("sanyuan_agent.memory.longterm", "IEmbeddingProvider"),
    "LocalEmbeddingProvider": ("sanyuan_agent.memory.longterm", "LocalEmbeddingProvider"),
    "LongtermMemoryStore": ("sanyuan_agent.memory.longterm", "LongtermMemoryStore"),
    "MemoryEntry": ("sanyuan_agent.memory.longterm", "MemoryEntry"),
    "OpenAIEmbeddingProvider": ("sanyuan_agent.memory.longterm", "OpenAIEmbeddingProvider"),
    "create_embedding_provider": ("sanyuan_agent.memory.longterm", "create_embedding_provider"),
    "ScopeKind": ("sanyuan_agent.memory.longterm.store", "ScopeKind"),
    "EpisodicRecall": ("sanyuan_agent.memory.recall", "EpisodicRecall"),
    "SensoryEvent": ("sanyuan_agent.memory.sensory", "SensoryEvent"),
    "SensoryMemory": ("sanyuan_agent.memory.sensory", "SensoryMemory"),
    "VectorRecall": ("sanyuan_agent.memory.vector", "VectorRecall"),
    "WorkingMemory": ("sanyuan_agent.memory.working", "WorkingMemory"),
    "WorkingStep": ("sanyuan_agent.memory.working", "WorkingStep"),
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
