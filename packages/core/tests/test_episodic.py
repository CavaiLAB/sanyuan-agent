"""Tests for EpisodicMemory — PG-backed long-term memory."""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from typing import Any

import pytest_asyncio
from neuroagent.memory.episodic import EpisodicMemory
from sqlalchemy import JSON, Column, DateTime, Index, MetaData, String, Table, text
from sqlalchemy.dialects.postgresql import UUID as _UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

# Schema-less table for SQLite test compatibility
test_metadata = MetaData()
test_episode_table = Table(
    "episodic_memory",
    test_metadata,
    Column("id", _UUID(as_uuid=True), primary_key=True),
    Column("tenant_id", _UUID(as_uuid=True), nullable=False),
    Column("conversation_id", _UUID(as_uuid=True), nullable=False),
    Column("user_query", String, nullable=False),
    Column("final_answer", String, nullable=False),
    Column("metadata", JSON, nullable=False, default=dict),
    Column(
        "created_at",
        DateTime(timezone=True),
        server_default=text("CURRENT_TIMESTAMP"),
    ),
    Index("ix_episodes_tenant_created", "tenant_id", "created_at"),
    Index("ix_episodes_conversation", "conversation_id", "created_at"),
)


@pytest_asyncio.fixture
async def session_factory() -> AsyncGenerator[async_sessionmaker[AsyncSession], None]:
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(test_metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


@pytest_asyncio.fixture
async def episodic(session_factory: async_sessionmaker[AsyncSession]) -> EpisodicMemory:
    return EpisodicMemory(session_factory=session_factory, table=test_episode_table)


class TestEpisodicMemory:
    async def test_add_episode_persists_row(self, episodic: EpisodicMemory) -> None:
        await episodic.add_episode(
            tenant_id=str(uuid.uuid4()),
            conversation_id=str(uuid.uuid4()),
            user_query="hello",
            final_answer="hi there",
        )
        episodes = await episodic.recent_episodes(
            tenant_id=str(uuid.uuid4()), limit=10
        )
        # Different tenant → empty
        assert episodes == []

    async def test_add_and_retrieve_same_tenant(
        self, episodic: EpisodicMemory
    ) -> None:
        tenant_id = str(uuid.uuid4())
        conv_id = str(uuid.uuid4())
        await episodic.add_episode(
            tenant_id=tenant_id,
            conversation_id=conv_id,
            user_query="q1",
            final_answer="a1",
        )
        episodes = await episodic.recent_episodes(
            tenant_id=tenant_id, limit=10
        )
        assert len(episodes) == 1
        assert episodes[0]["user_query"] == "q1"
        assert episodes[0]["final_answer"] == "a1"
        assert episodes[0]["tenant_id"] == tenant_id
        assert episodes[0]["conversation_id"] == conv_id

    async def test_tenant_isolation(self, episodic: EpisodicMemory) -> None:
        t1 = str(uuid.uuid4())
        t2 = str(uuid.uuid4())
        await episodic.add_episode(
            tenant_id=t1,
            conversation_id=str(uuid.uuid4()),
            user_query="t1-q",
            final_answer="t1-a",
        )
        await episodic.add_episode(
            tenant_id=t2,
            conversation_id=str(uuid.uuid4()),
            user_query="t2-q",
            final_answer="t2-a",
        )
        t1_eps = await episodic.recent_episodes(tenant_id=t1, limit=10)
        t2_eps = await episodic.recent_episodes(tenant_id=t2, limit=10)
        assert len(t1_eps) == 1
        assert len(t2_eps) == 1
        assert t1_eps[0]["user_query"] == "t1-q"
        assert t2_eps[0]["user_query"] == "t2-q"

    async def test_conversation_filter(self, episodic: EpisodicMemory) -> None:
        tenant_id = str(uuid.uuid4())
        c1 = str(uuid.uuid4())
        c2 = str(uuid.uuid4())
        await episodic.add_episode(
            tenant_id=tenant_id,
            conversation_id=c1,
            user_query="c1-q",
            final_answer="c1-a",
        )
        await episodic.add_episode(
            tenant_id=tenant_id,
            conversation_id=c2,
            user_query="c2-q",
            final_answer="c2-a",
        )
        c1_eps = await episodic.recent_episodes(
            tenant_id=tenant_id, conversation_id=c1, limit=10
        )
        assert len(c1_eps) == 1
        assert c1_eps[0]["user_query"] == "c1-q"

    async def test_limit_respected(self, episodic: EpisodicMemory) -> None:
        tenant_id = str(uuid.uuid4())
        for i in range(5):
            await episodic.add_episode(
                tenant_id=tenant_id,
                conversation_id=str(uuid.uuid4()),
                user_query=f"q{i}",
                final_answer=f"a{i}",
            )
        episodes = await episodic.recent_episodes(tenant_id=tenant_id, limit=3)
        assert len(episodes) == 3

    async def test_ordering_newest_first(self, episodic: EpisodicMemory) -> None:
        tenant_id = str(uuid.uuid4())
        conv = str(uuid.uuid4())
        for i in range(3):
            await episodic.add_episode(
                tenant_id=tenant_id,
                conversation_id=conv,
                user_query=f"q{i}",
                final_answer=f"a{i}",
            )
        episodes = await episodic.recent_episodes(tenant_id=tenant_id, limit=10)
        assert len(episodes) == 3
        # Newest first → reverse order of insertion
        assert episodes[0]["user_query"] == "q2"
        assert episodes[1]["user_query"] == "q1"
        assert episodes[2]["user_query"] == "q0"

    async def test_empty_result(self, episodic: EpisodicMemory) -> None:
        episodes = await episodic.recent_episodes(
            tenant_id=str(uuid.uuid4()), limit=10
        )
        assert episodes == []

    async def test_metadata_serialization(self, episodic: EpisodicMemory) -> None:
        tenant_id = str(uuid.uuid4())
        await episodic.add_episode(
            tenant_id=tenant_id,
            conversation_id=str(uuid.uuid4()),
            user_query="q",
            final_answer="a",
            metadata={"tool_call_count": 3, "duration": 12.34},
        )
        episodes = await episodic.recent_episodes(tenant_id=tenant_id, limit=10)
        assert len(episodes) == 1
        meta = episodes[0]["metadata"]
        assert meta["tool_call_count"] == 3
        assert meta["duration"] == 12.34

    async def test_large_json_metadata(self, episodic: EpisodicMemory) -> None:
        tenant_id = str(uuid.uuid4())
        big = {"items": [{"idx": i, "data": "x" * 100} for i in range(50)]}
        await episodic.add_episode(
            tenant_id=tenant_id,
            conversation_id=str(uuid.uuid4()),
            user_query="q",
            final_answer="a",
            metadata=big,
        )
        episodes = await episodic.recent_episodes(tenant_id=tenant_id, limit=10)
        assert len(episodes) == 1
        assert len(episodes[0]["metadata"]["items"]) == 50

    async def test_none_metadata_defaults_to_empty(
        self, episodic: EpisodicMemory
    ) -> None:
        tenant_id = str(uuid.uuid4())
        await episodic.add_episode(
            tenant_id=tenant_id,
            conversation_id=str(uuid.uuid4()),
            user_query="q",
            final_answer="a",
            metadata=None,
        )
        episodes = await episodic.recent_episodes(tenant_id=tenant_id, limit=10)
        assert episodes[0]["metadata"] == {}

    async def test_episode_has_id_and_timestamps(
        self, episodic: EpisodicMemory
    ) -> None:
        tenant_id = str(uuid.uuid4())
        await episodic.add_episode(
            tenant_id=tenant_id,
            conversation_id=str(uuid.uuid4()),
            user_query="q",
            final_answer="a",
        )
        episodes = await episodic.recent_episodes(tenant_id=tenant_id, limit=10)
        assert "id" in episodes[0]
        assert uuid.UUID(episodes[0]["id"])
        assert "created_at" in episodes[0]
        assert episodes[0]["created_at"] is not None


class TestAgentIntegration:
    """E2E-style integration for agent-level episodic hooks."""

    async def test_maybe_save_episode_no_op_without_memory(self) -> None:
        from neuroagent.agents.simple import _maybe_save_episode

        state: dict[str, Any] = {"messages": []}
        # No episodic memory → should return immediately without error
        await _maybe_save_episode(
            state,  # type: ignore[arg-type]
            None,
            "tenant-1",
            "conv-1",
            0.0,
        )

    async def test_maybe_save_episode_no_op_without_ids(self) -> None:
        from neuroagent.agents.simple import _maybe_save_episode

        state: dict[str, Any] = {"messages": []}
        fake_episodic = EpisodicMemory(
            session_factory=async_sessionmaker(
                bind=create_async_engine("sqlite+aiosqlite:///:memory:")
            )
        )
        # Empty tenant_id / conversation_id → should return immediately
        await _maybe_save_episode(
            state,  # type: ignore[arg-type]
            fake_episodic,
            "",
            "",
            0.0,
        )

    async def test_extract_user_query(self) -> None:
        from neuroagent.agents.simple import _extract_user_query

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello world"},
            {"role": "assistant", "content": "hi"},
        ]
        assert _extract_user_query(messages) == "hello world"

    async def test_extract_user_query_empty(self) -> None:
        from neuroagent.agents.simple import _extract_user_query

        assert _extract_user_query([]) == ""
        assert _extract_user_query([{"role": "assistant", "content": "x"}]) == ""
