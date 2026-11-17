"""Episodic long-term memory — PG-backed storage of session summaries."""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from sqlalchemy import JSON, Column, DateTime, Index, MetaData, String, Table, select, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from sanyuan_agent.memory.abc import IEpisodicStore

logger = logging.getLogger(__name__)

SCHEMA = "memory"

metadata = MetaData(schema=SCHEMA)

Episode = Table(
    "episodic_memory",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
    Column("tenant_id", UUID(as_uuid=True), nullable=False),
    Column("conversation_id", UUID(as_uuid=True), nullable=False),
    Column("user_query", String, nullable=False),
    Column("final_answer", String, nullable=False),
    Column("metadata", JSON, nullable=False, default=dict),
    Column("created_at", DateTime(timezone=True), server_default=text("now()")),
    Index("ix_episodes_tenant_created", "tenant_id", "created_at"),
    Index("ix_episodes_conversation", "conversation_id", "created_at"),
)


class EpisodicMemory(IEpisodicStore):
    """Async PG-backed episodic memory store.

    *session_factory* is an :class:`async_sessionmaker` bound to a PostgreSQL
    engine with the ``memory`` schema already created.

    *table* may be overridden for testing (e.g. schema-less SQLite).
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        table: Table | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._table = table if table is not None else Episode

    async def add_episode(
        self,
        tenant_id: str,
        conversation_id: str,
        user_query: str,
        final_answer: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist an episode summarising a completed agent turn."""
        meta = metadata or {}
        stmt = self._table.insert().values(
            id=uuid.uuid4(),
            tenant_id=uuid.UUID(tenant_id),
            conversation_id=uuid.UUID(conversation_id),
            user_query=user_query,
            final_answer=final_answer,
            metadata=meta,
        )
        async with self._session_factory() as session:
            async with session.begin():
                await session.execute(stmt)

    async def recent_episodes(
        self,
        tenant_id: str,
        conversation_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Return the most recent episodes, newest first."""
        stmt = (
            select(
                self._table.c.id,
                self._table.c.tenant_id,
                self._table.c.conversation_id,
                self._table.c.user_query,
                self._table.c.final_answer,
                self._table.c.metadata,
                self._table.c.created_at,
            )
            .where(self._table.c.tenant_id == uuid.UUID(tenant_id))
            .order_by(self._table.c.created_at.desc())
            .limit(limit)
        )
        if conversation_id is not None:
            stmt = stmt.where(
                self._table.c.conversation_id == uuid.UUID(conversation_id)
            )

        async with self._session_factory() as session:
            result = await session.execute(stmt)
            rows = result.mappings().all()

        return [
            {
                "id": str(row["id"]),
                "tenant_id": str(row["tenant_id"]),
                "conversation_id": str(row["conversation_id"]),
                "user_query": row["user_query"],
                "final_answer": row["final_answer"],
                "metadata": row["metadata"] or {},
                "created_at": (
                    row["created_at"].isoformat() if row["created_at"] else None
                ),
            }
            for row in rows
        ]

    async def episodes_stream(
        self,
        tenant_id: str,
        conversation_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield episodes lazily (newest first)."""
        stmt = (
            select(self._table)
            .where(self._table.c.tenant_id == uuid.UUID(tenant_id))
            .order_by(self._table.c.created_at.desc())
        )
        if conversation_id is not None:
            stmt = stmt.where(
                self._table.c.conversation_id == uuid.UUID(conversation_id)
            )

        async with self._session_factory() as session:
            result = await session.stream(stmt)
            async for row in result.mappings():
                yield {
                    "id": str(row["id"]),
                    "tenant_id": str(row["tenant_id"]),
                    "conversation_id": str(row["conversation_id"]),
                    "user_query": row["user_query"],
                    "final_answer": row["final_answer"],
                    "metadata": row["metadata"] or {},
                    "created_at": (
                        row["created_at"].isoformat() if row["created_at"] else None
                    ),
                }
