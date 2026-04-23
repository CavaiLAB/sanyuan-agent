"""Tests for long-term memory — embedding providers, store, tools (all mocked PG)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from neuroagent.memory.longterm.embedding import (
    LocalEmbeddingProvider,
    OpenAIEmbeddingProvider,
    create_embedding_provider,
)
from neuroagent.memory.longterm.store import LongtermMemoryStore, MemoryEntry, ScopeKind
from neuroagent.memory.longterm.tools import build_memory_tools
from neuroagent.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Fake embedding provider
# ---------------------------------------------------------------------------


class FakeEmbedding:
    """Deterministic embedding provider for tests."""

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    async def embed(self, text: str) -> list[float]:
        # Deterministic: hash-based vector, normalized
        raw = [float(ord(c) % 97) for c in (text + "\x00" * self._dim)[: self._dim]]
        norm = sum(x * x for x in raw) ** 0.5 or 1.0
        return [x / norm for x in raw]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_TENANT_A = str(uuid.uuid4())
_TENANT_B = str(uuid.uuid4())
_USER_A = str(uuid.uuid4())
_USER_B = str(uuid.uuid4())


class _FakeRow:
    """Mimics a SQLAlchemy mapping row — supports both attr and subscript access."""

    def __init__(self, **kwargs: Any) -> None:
        self._data = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]


class _FakeMappings:
    """Mimics SQLAlchemy result.mappings()."""

    def __init__(self, rows: list[_FakeRow]) -> None:
        self._rows = rows

    def __iter__(self) -> Any:
        return iter(self._rows)

    def all(self) -> list[_FakeRow]:
        return self._rows


def _make_session(
    rows: list[dict[str, Any]] | None = None,
    *,
    rowcount: int | None = None,
) -> AsyncMock:
    """Build a mock async session that returns *rows* from SELECT execute().

    For INSERT/DELETE execute(), returns a simple mock with rowcount.
    """
    session = AsyncMock()
    fake_rows = [_FakeRow(**r) for r in (rows or [])]

    result_mock = MagicMock()
    result_mock.mappings.return_value = fake_rows
    result_mock.rowcount = rowcount if rowcount is not None else len(fake_rows)

    # execute returns different things based on call
    execute_results = [result_mock]

    async def _execute(stmt: Any, params: Any = None) -> Any:
        return execute_results[0]

    session.execute = AsyncMock(side_effect=_execute)

    # Make session an async context manager
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    # Make session.begin() an async context manager
    begin_ctx = AsyncMock()
    begin_ctx.__aenter__ = AsyncMock(return_value=None)
    begin_ctx.__aexit__ = AsyncMock(return_value=False)
    session.begin = MagicMock(return_value=begin_ctx)

    return session


def _make_session_factory(session: AsyncMock) -> Any:
    def factory() -> Any:
        return session

    return factory


@pytest.fixture()
def embedding() -> FakeEmbedding:
    return FakeEmbedding()


@pytest.fixture()
def registry() -> ToolRegistry:
    return ToolRegistry()


# ---------------------------------------------------------------------------
# 1. Embedding: OpenAI provider (mocked HTTP)
# ---------------------------------------------------------------------------


async def test_openai_embedding_calls_api() -> None:
    provider = OpenAIEmbeddingProvider(dimension=3)
    assert provider.dimension == 3

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await provider.embed("hello")
        assert result == [0.1, 0.2, 0.3]
        mock_client.post.assert_awaited_once()


# ---------------------------------------------------------------------------
# 2. Embedding: local provider (mocked model)
# ---------------------------------------------------------------------------


async def test_local_embedding_returns_vector() -> None:
    provider = LocalEmbeddingProvider(dimension=4)
    mock_model = MagicMock()
    mock_model.encode.return_value = MagicMock(tolist=lambda: [1.0, 0.0, 0.5, 0.5])

    with patch.object(provider, "_load_model", return_value=mock_model):
        result = await provider.embed("test text")
        assert len(result) == 4
        assert result[0] == 1.0


# ---------------------------------------------------------------------------
# 3. Embedding: factory selects by name
# ---------------------------------------------------------------------------


def test_create_provider_openai() -> None:
    p = create_embedding_provider(provider="openai")
    assert isinstance(p, OpenAIEmbeddingProvider)


def test_create_provider_local() -> None:
    p = create_embedding_provider(provider="local")
    assert isinstance(p, LocalEmbeddingProvider)


# ---------------------------------------------------------------------------
# 4. Embedding: factory reads env
# ---------------------------------------------------------------------------


def test_create_provider_from_env() -> None:
    with patch("os.environ.get", return_value="local"):
        p = create_embedding_provider()
        assert isinstance(p, LocalEmbeddingProvider)


# ---------------------------------------------------------------------------
# 5. Store: store() embeds and inserts
# ---------------------------------------------------------------------------


async def test_store_inserts_with_embedding(embedding: FakeEmbedding) -> None:
    session = _make_session([])
    store = LongtermMemoryStore(
        _make_session_factory(session), embedding,
    )
    entry = await store.store(
        tenant_id=_TENANT_A,
        content="Important fact",
        scope=ScopeKind.PRIVATE,
        scope_owner_id=_USER_A,
        importance=0.8,
    )
    assert entry.content == "Important fact"
    assert entry.scope == ScopeKind.PRIVATE
    assert entry.importance == 0.8
    session.execute.assert_awaited_once()
    call_sql = str(session.execute.call_args[0][0])
    assert "INSERT INTO memory.longterm_memory" in call_sql


# ---------------------------------------------------------------------------
# 6. Store: retrieve_semantic returns results with similarity
# ---------------------------------------------------------------------------


async def test_retrieve_returns_similarity(embedding: FakeEmbedding) -> None:
    now = datetime.now(UTC)
    rows = [
        {
            "id": uuid.uuid4(),
            "tenant_id": uuid.UUID(_TENANT_A),
            "scope": "private",
            "scope_owner_id": uuid.UUID(_USER_A),
            "content": "Relevant fact",
            "importance": 0.9,
            "metadata": {},
            "created_at": now,
            "expires_at": None,
            "similarity": 0.87,
        },
    ]
    session = _make_session(rows)
    store = LongtermMemoryStore(
        _make_session_factory(session), embedding,
    )
    results = await store.retrieve_semantic(
        tenant_id=_TENANT_A,
        query="fact",
        limit=5,
        importance_threshold=0.3,
    )
    assert len(results) == 1
    assert results[0].similarity_score == 0.87
    assert results[0].content == "Relevant fact"


# ---------------------------------------------------------------------------
# 7. Store: tenant isolation (tenant B gets no results)
# ---------------------------------------------------------------------------


async def test_tenant_isolation(embedding: FakeEmbedding) -> None:
    session = _make_session([])
    store = LongtermMemoryStore(
        _make_session_factory(session), embedding,
    )
    results = await store.retrieve_semantic(
        tenant_id=_TENANT_B,
        query="anything",
    )
    assert results == []
    # Verify tenant_id was passed in query params
    call_kwargs = session.execute.call_args[0][1]
    assert call_kwargs["tid"] == uuid.UUID(_TENANT_B)


# ---------------------------------------------------------------------------
# 8. Store: scope filtering
# ---------------------------------------------------------------------------


async def test_scope_filtering(embedding: FakeEmbedding) -> None:
    session = _make_session([])
    store = LongtermMemoryStore(
        _make_session_factory(session), embedding,
    )
    await store.retrieve_semantic(
        tenant_id=_TENANT_A,
        query="test",
        scope=[ScopeKind.TEAM, ScopeKind.TENANT],
    )
    call_kwargs = session.execute.call_args[0][1]
    assert "team" in call_kwargs["scopes"]
    assert "tenant" in call_kwargs["scopes"]
    assert "private" not in call_kwargs["scopes"]


# ---------------------------------------------------------------------------
# 9. Store: importance threshold
# ---------------------------------------------------------------------------


async def test_importance_threshold(embedding: FakeEmbedding) -> None:
    session = _make_session([])
    store = LongtermMemoryStore(
        _make_session_factory(session), embedding,
    )
    await store.retrieve_semantic(
        tenant_id=_TENANT_A,
        query="test",
        importance_threshold=0.7,
    )
    call_kwargs = session.execute.call_args[0][1]
    assert call_kwargs["thresh"] == 0.7


# ---------------------------------------------------------------------------
# 10. Store: top-K limit
# ---------------------------------------------------------------------------


async def test_top_k_limit(embedding: FakeEmbedding) -> None:
    session = _make_session([])
    store = LongtermMemoryStore(
        _make_session_factory(session), embedding,
    )
    await store.retrieve_semantic(
        tenant_id=_TENANT_A,
        query="test",
        limit=3,
    )
    call_kwargs = session.execute.call_args[0][1]
    assert call_kwargs["lim"] == 3


# ---------------------------------------------------------------------------
# 11. Store: expire_old
# ---------------------------------------------------------------------------


async def test_expire_old(embedding: FakeEmbedding) -> None:
    session = _make_session([], rowcount=5)
    store = LongtermMemoryStore(
        _make_session_factory(session), embedding,
    )
    cutoff = datetime.now(UTC)
    deleted = await store.expire_old(cutoff)
    assert deleted == 5
    call_sql = str(session.execute.call_args[0][0])
    assert "DELETE FROM memory.longterm_memory" in call_sql


# ---------------------------------------------------------------------------
# 12. Store: MemoryEntry fields
# ---------------------------------------------------------------------------


def test_memory_entry_fields() -> None:
    entry = MemoryEntry(
        id="abc",
        tenant_id=_TENANT_A,
        scope=ScopeKind.TEAM,
        scope_owner_id=_USER_A,
        content="Test",
        importance=0.5,
        metadata={"key": "val"},
        created_at="2026-01-01T00:00:00",
        expires_at=None,
        similarity_score=0.95,
    )
    assert entry.scope == ScopeKind.TEAM
    assert entry.similarity_score == 0.95
    assert entry.metadata["key"] == "val"


# ---------------------------------------------------------------------------
# 13. Tools: remember_this success
# ---------------------------------------------------------------------------


async def test_remember_this_success(registry: ToolRegistry, embedding: FakeEmbedding) -> None:
    session = _make_session([])
    store = LongtermMemoryStore(
        _make_session_factory(session), embedding,
    )
    build_memory_tools(registry, store, tenant_id=_TENANT_A, user_id=_USER_A)
    fn = registry.get("remember_this")
    result = await fn(content="Remember this fact", scope="private", importance="0.8")
    assert result["success"] is True
    assert "memory_id" in result


# ---------------------------------------------------------------------------
# 14. Tools: remember_this invalid scope
# ---------------------------------------------------------------------------


async def test_remember_this_invalid_scope(registry: ToolRegistry, embedding: FakeEmbedding) -> None:
    session = _make_session([])
    store = LongtermMemoryStore(
        _make_session_factory(session), embedding,
    )
    build_memory_tools(registry, store, tenant_id=_TENANT_A, user_id=_USER_A)
    fn = registry.get("remember_this")
    result = await fn(content="Test", scope="invalid_scope")
    assert result["success"] is False
    assert "Invalid scope" in result["error"]


# ---------------------------------------------------------------------------
# 15. Tools: recall_memories success
# ---------------------------------------------------------------------------


async def test_recall_memories_success(registry: ToolRegistry, embedding: FakeEmbedding) -> None:
    now = datetime.now(UTC)
    rows = [
        {
            "id": uuid.uuid4(),
            "tenant_id": uuid.UUID(_TENANT_A),
            "scope": "tenant",
            "scope_owner_id": None,
            "content": "Global fact",
            "importance": 0.6,
            "metadata": {},
            "created_at": now,
            "expires_at": None,
            "similarity": 0.75,
        },
    ]
    session = _make_session(rows)
    store = LongtermMemoryStore(
        _make_session_factory(session), embedding,
    )
    build_memory_tools(registry, store, tenant_id=_TENANT_A, user_id=_USER_A)
    fn = registry.get("recall_memories")
    result = await fn(query="fact", scope="tenant", limit="5", importance_threshold="0.3")
    assert result["success"] is True
    assert result["count"] == 1
    assert result["memories"][0]["similarity"] == 0.75


# ---------------------------------------------------------------------------
# 16. Tools: recall_memories empty
# ---------------------------------------------------------------------------


async def test_recall_memories_empty(registry: ToolRegistry, embedding: FakeEmbedding) -> None:
    session = _make_session([])
    store = LongtermMemoryStore(
        _make_session_factory(session), embedding,
    )
    build_memory_tools(registry, store, tenant_id=_TENANT_A, user_id=_USER_A)
    fn = registry.get("recall_memories")
    result = await fn(query="nothing matches")
    assert result["success"] is True
    assert result["count"] == 0


# ---------------------------------------------------------------------------
# 17. ScopeKind enum values
# ---------------------------------------------------------------------------


def test_scope_kind_values() -> None:
    assert ScopeKind.PRIVATE == "private"
    assert ScopeKind.TEAM == "team"
    assert ScopeKind.TENANT == "tenant"
    assert ScopeKind("private") == ScopeKind.PRIVATE


# ---------------------------------------------------------------------------
# 18. Integration: longterm inject into messages
# ---------------------------------------------------------------------------


async def test_maybe_inject_longterm() -> None:
    from neuroagent.agents.simple import _maybe_inject_longterm

    now = datetime.now(UTC)
    rows = [
        {
            "id": uuid.uuid4(),
            "tenant_id": uuid.UUID(_TENANT_A),
            "scope": "private",
            "scope_owner_id": uuid.UUID(_USER_A),
            "content": "User fact",
            "importance": 0.8,
            "metadata": {},
            "created_at": now,
            "expires_at": None,
            "similarity": 0.91,
        },
    ]
    session = _make_session(rows)
    embedding = FakeEmbedding()
    store = LongtermMemoryStore(
        _make_session_factory(session), embedding,
    )

    messages: list[dict[str, Any]] = [{"role": "user", "content": "Tell me facts"}]
    result = await _maybe_inject_longterm(messages, store, _TENANT_A, _USER_A)
    assert len(result) == 2
    assert result[0]["role"] == "system"
    assert "Relevant memories" in result[0]["content"]
    assert "User fact" in result[0]["content"]


# ---------------------------------------------------------------------------
# 19. Integration: longterm inject skips when store is None
# ---------------------------------------------------------------------------


async def test_maybe_inject_longterm_none() -> None:
    from neuroagent.agents.simple import _maybe_inject_longterm

    messages: list[dict[str, Any]] = [{"role": "user", "content": "Hello"}]
    result = await _maybe_inject_longterm(messages, None, _TENANT_A, _USER_A)
    assert result is messages
