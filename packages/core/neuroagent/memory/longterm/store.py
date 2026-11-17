"""Long-term memory store — pgvector semantic retrieval with scope isolation."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import Column, Index, MetaData, String, Table, Text, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from neuroagent.memory.context import (
    MemoryContext,
    MemoryLevel,
    validate_memory_context,
)
from neuroagent.memory.longterm.embedding import IEmbeddingProvider

logger = logging.getLogger(__name__)

SCHEMA = "memory"
PGVECTOR_DIM = 1536

metadata = MetaData(schema=SCHEMA)


class ScopeKind(StrEnum):
    """Access scope for a memory entry."""

    PRIVATE = "private"
    TEAM = "team"
    TENANT = "tenant"


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    """A single long-term memory record."""

    id: str
    tenant_id: str
    scope: ScopeKind
    scope_owner_id: str | None
    content: str
    importance: float
    metadata: dict[str, Any]
    created_at: str | None
    expires_at: str | None
    level: MemoryLevel = MemoryLevel.PERSONAL
    project_id: str | None = None
    session_id: str | None = None
    similarity_score: float | None = None


def longterm_memory_table(dimension: int = PGVECTOR_DIM) -> Table:
    """Build the ``longterm_memory`` SQLAlchemy table.

    The ``embedding`` column uses raw DDL (pgvector) and is created via
    the SQL migration, not through SQLAlchemy DDL generation.
    """
    return Table(
        "longterm_memory",
        metadata,
        Column("id", UUID(as_uuid=True), primary_key=True),
        Column("tenant_id", UUID(as_uuid=True), nullable=False),
        Column("scope", String(16), nullable=False, default="private"),
        Column("scope_owner_id", UUID(as_uuid=True), nullable=True),
        Column("content", Text, nullable=False),
        Column("level", String(16), nullable=False, default="personal"),
        Column("project_id", Text, nullable=True),
        Column("session_id", Text, nullable=True),
        # embedding column managed by raw SQL (pgvector)
        Column("importance", String(), nullable=False, default="0.5"),
        Column("metadata", JSONB, nullable=False, default=dict),
        Column("created_at", String(), server_default=text("now()")),
        Column("expires_at", String(), nullable=True),
        Index(
            "ix_longterm_tenant_scope",
            "tenant_id",
            "scope",
            "scope_owner_id",
        ),
        extend_existing=True,
    )


class LongtermMemoryStore:
    """Async PG-backed long-term memory store with pgvector semantic search.

    Parameters
    ----------
    session_factory:
        An ``async_sessionmaker`` bound to a PostgreSQL engine with pgvector.
    embedding_provider:
        Provider used to generate embeddings for storage and queries.
    table:
        Optional table override (for testing with alternative schemas).
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        embedding_provider: IEmbeddingProvider,
        *,
        table: Table | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._embedding = embedding_provider
        self._table = table

    def _get_table(self) -> Table:
        if self._table is not None:
            return self._table
        return longterm_memory_table(self._embedding.dimension)

    async def store(
        self,
        *,
        tenant_id: str,
        content: str,
        scope: ScopeKind = ScopeKind.PRIVATE,
        scope_owner_id: str | None = None,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
        expires_at: datetime | None = None,
        level: MemoryLevel | str = MemoryLevel.PERSONAL,
        project_id: str | None = None,
        session_id: str | None = None,
    ) -> MemoryEntry:
        """Embed *content* and persist to the longterm_memory table."""
        normalized_level = validate_memory_context(
            level,
            project_id=project_id,
            session_id=session_id,
        )
        vec = await self._embedding.embed(content)
        entry_id = uuid.uuid4()
        meta = metadata or {}

        async with self._session_factory() as session:
            async with session.begin():
                await session.execute(
                    text(
                        "INSERT INTO memory.longterm_memory "
                        "(id, tenant_id, scope, scope_owner_id, content, "
                        "embedding, importance, metadata, expires_at, level, "
                        "project_id, session_id) "
                        "VALUES (:id, :tid, :scope, :owner, :content, "
                        ":embedding::vector, :importance, :meta, :expires, :level, "
                        ":project_id, :session_id)"
                    ),
                    {
                        "id": entry_id,
                        "tid": uuid.UUID(tenant_id),
                        "scope": scope.value,
                        "owner": (uuid.UUID(scope_owner_id) if scope_owner_id else None),
                        "content": content,
                        "embedding": _vec_to_pg(vec),
                        "importance": importance,
                        "meta": _json_to_str(meta),
                        "expires": expires_at,
                        "level": normalized_level.value,
                        "project_id": project_id,
                        "session_id": session_id,
                    },
                )

        return MemoryEntry(
            id=str(entry_id),
            tenant_id=tenant_id,
            scope=scope,
            scope_owner_id=scope_owner_id,
            content=content,
            importance=importance,
            metadata=meta,
            created_at=datetime.now(UTC).isoformat(),
            expires_at=expires_at.isoformat() if expires_at else None,
            level=normalized_level,
            project_id=project_id,
            session_id=session_id,
        )

    async def retrieve_semantic(
        self,
        *,
        tenant_id: str,
        query: str,
        scope: list[ScopeKind] | None = None,
        scope_owner_id: str | None = None,
        limit: int = 5,
        importance_threshold: float = 0.3,
        context: MemoryContext | None = None,
    ) -> list[MemoryEntry]:
        """Semantic search via cosine similarity on pgvector.

        Results are ordered by similarity (descending) and filtered by
        *importance_threshold* and *scope*.
        """
        vec = await self._embedding.embed(query)
        scopes = scope or [ScopeKind.PRIVATE, ScopeKind.TEAM, ScopeKind.TENANT]
        scope_values = [s.value for s in scopes]

        async with self._session_factory() as session:
            rows = await session.execute(
                text(
                    "SELECT id, tenant_id, scope, scope_owner_id, content, "
                    "importance, metadata, created_at, expires_at, level, "
                    "project_id, session_id, "
                    "1 - (embedding <=> :qvec::vector) AS similarity "
                    "FROM memory.longterm_memory "
                    "WHERE tenant_id = :tid "
                    "AND scope = ANY(:scopes) "
                    "AND importance >= :thresh "
                    "AND (expires_at IS NULL OR expires_at > now()) "
                    "AND (:owner_filter = false OR scope <> 'private' "
                    "OR scope_owner_id = CAST(:owner AS uuid)) "
                    "AND (:context_filter = false OR level = 'personal' "
                    "OR (level = 'project' AND project_id = :project_id) "
                    "OR (level = 'session' AND session_id = :session_id "
                    "AND (project_id IS NULL OR project_id = :project_id))) "
                    "ORDER BY embedding <=> :qvec::vector "
                    "LIMIT :lim"
                ),
                {
                    "qvec": _vec_to_pg(vec),
                    "tid": uuid.UUID(tenant_id),
                    "scopes": scope_values,
                    "thresh": importance_threshold,
                    "lim": limit,
                    "owner_filter": scope_owner_id is not None,
                    "owner": scope_owner_id,
                    "context_filter": context is not None,
                    "project_id": context.project_id if context else None,
                    "session_id": context.session_id if context else None,
                },
            )
            results = []
            for row in rows.mappings():
                results.append(
                    MemoryEntry(
                        id=str(row["id"]),
                        tenant_id=str(row["tenant_id"]),
                        scope=ScopeKind(row["scope"]),
                        scope_owner_id=(
                            str(row["scope_owner_id"]) if row["scope_owner_id"] else None
                        ),
                        content=row["content"],
                        importance=float(row["importance"]),
                        metadata=row["metadata"] or {},
                        created_at=(row["created_at"].isoformat() if row["created_at"] else None),
                        expires_at=(row["expires_at"].isoformat() if row["expires_at"] else None),
                        level=MemoryLevel(row.get("level") or "personal"),
                        project_id=row.get("project_id"),
                        session_id=row.get("session_id"),
                        similarity_score=float(row["similarity"]),
                    )
                )
            return results

    async def expire_old(self, cutoff: datetime) -> int:
        """Delete entries where ``expires_at < cutoff``. Returns count deleted."""
        async with self._session_factory() as session:
            async with session.begin():
                result = await session.execute(
                    text(
                        "DELETE FROM memory.longterm_memory "
                        "WHERE expires_at IS NOT NULL AND expires_at < :cutoff"
                    ),
                    {"cutoff": cutoff},
                )
                return int(result.rowcount) if result.rowcount is not None else 0  # type: ignore[attr-defined]

    async def enforce_capacity(self, max_entries_per_tenant: int) -> int:
        """Evict overflow beyond *max_entries_per_tenant* for every tenant.

        Keeps the highest-importance (newest first within a tie) entries;
        everything ranked past the cap is deleted. Returns count deleted.
        """
        if max_entries_per_tenant <= 0:
            raise ValueError("max_entries_per_tenant must be > 0")
        async with self._session_factory() as session:
            async with session.begin():
                result = await session.execute(
                    text(
                        "DELETE FROM memory.longterm_memory WHERE id IN ("
                        "SELECT id FROM ("
                        "SELECT id, row_number() OVER ("
                        "PARTITION BY tenant_id "
                        "ORDER BY importance DESC, created_at DESC"
                        ") AS rn FROM memory.longterm_memory"
                        ") ranked WHERE rn > :cap)"
                    ),
                    {"cap": max_entries_per_tenant},
                )
                return int(result.rowcount) if result.rowcount is not None else 0  # type: ignore[attr-defined]


def _vec_to_pg(vec: list[float]) -> str:
    """Format a float list as a pgvector literal."""
    return "[" + ",".join(str(v) for v in vec) + "]"


def _json_to_str(obj: dict[str, Any]) -> str:
    import json

    return json.dumps(obj)
