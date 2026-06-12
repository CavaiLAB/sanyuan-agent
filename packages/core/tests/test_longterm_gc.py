"""Long-term memory governance (015): TTL sweep + per-tenant capacity cap.

``LongtermCollector`` runs two passes per GC cycle against the long-term
store — delete expired entries, then evict the lowest-value overflow beyond
the per-tenant cap. ``LongtermMemoryStore.enforce_capacity`` ranks entries
per tenant by (importance desc, created_at desc) and deletes past the cap.
The collector plugs into ``GCScheduler`` like the sensory collector.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from neuroagent.gc import GCRunStats, GCScheduler, LongtermCollector
from neuroagent.memory.longterm.store import LongtermMemoryStore


class FakeGovernedStore:
    """Records governance calls; returns scripted counts."""

    def __init__(self, *, expired: int = 0, evicted: int = 0) -> None:
        self._expired = expired
        self._evicted = evicted
        self.expire_calls: list[datetime] = []
        self.capacity_calls: list[int] = []

    async def expire_old(self, cutoff: datetime) -> int:
        self.expire_calls.append(cutoff)
        return self._expired

    async def enforce_capacity(self, max_entries_per_tenant: int) -> int:
        self.capacity_calls.append(max_entries_per_tenant)
        return self._evicted


class TestLongtermCollector:
    async def test_runs_ttl_sweep_then_capacity_cap(self) -> None:
        store = FakeGovernedStore(expired=3, evicted=7)
        fixed_now = datetime(2026, 6, 13, 12, 0, tzinfo=UTC)
        collector = LongtermCollector(
            store, max_entries_per_tenant=100, clock=lambda: fixed_now
        )

        stats = await collector.collect()

        assert store.expire_calls == [fixed_now]
        assert store.capacity_calls == [100]
        assert stats.last_expired_count == 3
        assert stats.last_evicted_count == 7
        assert collector.stats is stats

    async def test_zero_cap_disables_eviction(self) -> None:
        store = FakeGovernedStore(expired=1, evicted=99)
        collector = LongtermCollector(store, max_entries_per_tenant=0)

        stats = await collector.collect()

        assert store.capacity_calls == []
        assert stats.last_evicted_count == 0
        assert stats.last_expired_count == 1

    def test_negative_cap_rejected(self) -> None:
        with pytest.raises(ValueError, match=">= 0"):
            LongtermCollector(FakeGovernedStore(), max_entries_per_tenant=-1)

    async def test_default_clock_is_utc_now(self) -> None:
        store = FakeGovernedStore()
        before = datetime.now(UTC)

        await LongtermCollector(store).collect()

        assert len(store.expire_calls) == 1
        assert before <= store.expire_calls[0] <= datetime.now(UTC)


class _StubSandboxCollector:
    """Minimal stand-in for SandboxCollector in scheduler tests."""

    def __init__(self) -> None:
        self._stats = GCRunStats()

    @property
    def stats(self) -> GCRunStats:
        return self._stats

    async def collect(self) -> GCRunStats:
        return self._stats


class TestSchedulerIntegration:
    async def test_run_once_propagates_longterm_counts(self) -> None:
        store = FakeGovernedStore(expired=2, evicted=5)
        scheduler = GCScheduler(
            _StubSandboxCollector(),  # type: ignore[arg-type]
            longterm_collector=LongtermCollector(store, max_entries_per_tenant=10),
        )

        stats = await scheduler.run_once()

        assert stats.last_longterm_expired_count == 2
        assert stats.last_longterm_evicted_count == 5

    async def test_without_longterm_collector_counts_stay_zero(self) -> None:
        stats = await GCScheduler(_StubSandboxCollector()).run_once()  # type: ignore[arg-type]

        assert stats.last_longterm_expired_count == 0
        assert stats.last_longterm_evicted_count == 0


class _FakeSqlResult:
    def __init__(self, rowcount: int | None) -> None:
        self.rowcount = rowcount


class _FakeSqlSession:
    """Async-context session capturing execute() statements."""

    def __init__(self, rowcount: int | None) -> None:
        self._rowcount = rowcount
        self.statements: list[tuple[str, dict[str, Any]]] = []

    async def __aenter__(self) -> _FakeSqlSession:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    def begin(self) -> _FakeSqlSession:
        return self

    async def execute(self, stmt: Any, params: dict[str, Any]) -> _FakeSqlResult:
        self.statements.append((str(stmt), params))
        return _FakeSqlResult(self._rowcount)


class _FakeEmbedding:
    @property
    def dimension(self) -> int:
        return 8

    async def embed(self, text: str) -> list[float]:
        return [0.0] * 8


def _store_with(session: _FakeSqlSession) -> LongtermMemoryStore:
    return LongtermMemoryStore(lambda: session, _FakeEmbedding())  # type: ignore[arg-type]


class TestEnforceCapacitySql:
    async def test_deletes_overflow_ranked_per_tenant(self) -> None:
        session = _FakeSqlSession(rowcount=4)
        deleted = await _store_with(session).enforce_capacity(500)

        assert deleted == 4
        sql, params = session.statements[0]
        assert "row_number() OVER" in sql
        assert "PARTITION BY tenant_id" in sql
        assert "ORDER BY importance DESC, created_at DESC" in sql
        assert "rn > :cap" in sql
        assert params == {"cap": 500}

    async def test_none_rowcount_returns_zero(self) -> None:
        session = _FakeSqlSession(rowcount=None)
        assert await _store_with(session).enforce_capacity(10) == 0

    async def test_non_positive_cap_rejected(self) -> None:
        session = _FakeSqlSession(rowcount=0)
        store = _store_with(session)
        for cap in (0, -5):
            with pytest.raises(ValueError, match="> 0"):
                await store.enforce_capacity(cap)
        assert session.statements == []
