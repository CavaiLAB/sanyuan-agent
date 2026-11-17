"""Long-term memory governance collector: TTL sweep + per-tenant capacity cap."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from opentelemetry import metrics

_meter = metrics.get_meter("sanyuan_agent.gc")

_longterm_expired_counter = _meter.create_counter(
    "neuroagent_longterm_expired_total",
    description="Total long-term memory entries deleted by TTL sweep",
    unit="1",
)
_longterm_evicted_counter = _meter.create_counter(
    "neuroagent_longterm_evicted_total",
    description="Total long-term memory entries evicted by the capacity cap",
    unit="1",
)


class GovernedMemoryStore(Protocol):
    """The slice of ``LongtermMemoryStore`` the collector needs."""

    async def expire_old(self, cutoff: datetime) -> int: ...

    async def enforce_capacity(self, max_entries_per_tenant: int) -> int: ...


@dataclass
class LongtermGCStats:
    """Counts from the most recent governance run."""

    last_expired_count: int = 0
    last_evicted_count: int = 0


class LongtermCollector:
    """Governs the long-term memory table on each GC cycle.

    Two passes: delete entries whose ``expires_at`` has passed (TTL sweep),
    then evict the lowest-value overflow beyond *max_entries_per_tenant*
    (capacity cap; the store keeps the highest-importance / newest entries).
    A cap of 0 disables eviction.
    """

    def __init__(
        self,
        store: GovernedMemoryStore,
        *,
        max_entries_per_tenant: int = 50_000,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if max_entries_per_tenant < 0:
            raise ValueError("max_entries_per_tenant must be >= 0")
        self._store = store
        self._cap = max_entries_per_tenant
        self._clock = clock or (lambda: datetime.now(UTC))
        self._stats = LongtermGCStats()

    @property
    def stats(self) -> LongtermGCStats:
        return self._stats

    async def collect(self) -> LongtermGCStats:
        expired = await self._store.expire_old(self._clock())
        evicted = 0
        if self._cap > 0:
            evicted = await self._store.enforce_capacity(self._cap)

        if expired > 0:
            _longterm_expired_counter.add(expired)
        if evicted > 0:
            _longterm_evicted_counter.add(evicted)

        self._stats = LongtermGCStats(
            last_expired_count=expired,
            last_evicted_count=evicted,
        )
        return self._stats
