"""MemoryToolProvider: the existing remember_this/recall_memories as a provider.

Pins that the provider/spec path yields the same two builder tools with memory
metadata and that they execute end-to-end against a duck-typed store — without
re-implementing the tools (no behavior/schema drift from the builder).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from sanyuan_agent.memory.context import MemoryLevel
from sanyuan_agent.memory.longterm.store import ScopeKind
from sanyuan_agent.tools.memory import MemoryToolProvider
from sanyuan_agent.tools.provider import register_provider
from sanyuan_agent.tools.registry import ToolRegistry


@dataclass
class _Entry:
    id: str
    content: str
    scope: ScopeKind
    importance: float
    level: MemoryLevel = MemoryLevel.PERSONAL
    project_id: str | None = None
    session_id: str | None = None
    similarity_score: float = 0.9
    created_at: str = "2026-06-28T00:00:00Z"


class _FakeStore:
    """Minimal duck-typed LongtermMemoryStore for the two tools."""

    def __init__(self) -> None:
        self.saved: list[_Entry] = []

    async def store(
        self,
        *,
        tenant_id,
        content,
        scope,
        scope_owner_id,
        importance,
        level,
        project_id,
        session_id,
    ) -> _Entry:
        entry = _Entry(
            id=f"m{len(self.saved)}",
            content=content,
            scope=scope,
            importance=importance,
            level=level,
            project_id=project_id,
            session_id=session_id,
        )
        self.saved.append(entry)
        return entry

    async def retrieve_semantic(
        self,
        *,
        tenant_id,
        query,
        scope,
        scope_owner_id,
        limit,
        importance_threshold,
        context,
    ) -> list[_Entry]:
        hits = [e for e in self.saved if e.importance >= importance_threshold]
        return hits[:limit]


class TestSpecs:
    def test_yields_both_memory_tools_with_metadata(self) -> None:
        specs = {s.name: s for s in MemoryToolProvider(_FakeStore()).provide()}
        assert set(specs) == {"remember_this", "recall_memories"}
        for spec in specs.values():
            assert spec.category == "memory"

        remember = specs["remember_this"]
        assert remember.side_effect == "write"
        assert remember.policy_hint == "confirm"
        assert remember.capabilities == ("memory", "write")

        recall = specs["recall_memories"]
        assert recall.side_effect == "read"
        assert recall.policy_hint is None


class TestExecution:
    async def test_remember_then_recall_roundtrip(self) -> None:
        store = _FakeStore()
        registry = ToolRegistry()
        names = register_provider(registry, MemoryToolProvider(store, tenant_id="t1", user_id="u1"))
        assert set(names) == {"remember_this", "recall_memories"}

        saved = await registry.execute(
            "remember_this",
            json.dumps({"content": "likes dark mode", "scope": "private"}),
        )
        assert saved["success"] is True
        assert saved["scope"] == "private"

        found = await registry.execute("recall_memories", json.dumps({"query": "preferences"}))
        assert found["success"] is True
        assert found["count"] == 1
        assert found["memories"][0]["content"] == "likes dark mode"
