"""UsageTracker — async queue + background worker for LLM token usage recording."""

from __future__ import annotations

import asyncio
import collections
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from opentelemetry import metrics

logger = logging.getLogger(__name__)

_MAX_PENDING = 10_000
_FLUSH_INTERVAL = 5.0  # seconds


@dataclass(frozen=True, slots=True)
class LLMCallUsage:
    """Token-usage record for a single LLM completion call."""

    run_id: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    timestamp: float
    tenant_id: str
    # Usage-ledger attribution (ADR di-saas-billing-metering D-3). project_id /
    # consumer_id may be unknown at enqueue time — the flush side falls back to
    # the tenant's default project. record_id is minted once at creation and
    # survives flush retries, so ledger idempotency keys derived from it dedupe
    # redeliveries instead of double-metering.
    project_id: str | None = None
    consumer_id: str | None = None
    trace_id: str | None = None
    record_id: str = field(default_factory=lambda: uuid.uuid4().hex)


# OTel metric -----------------------------------------------------------------

_meter = metrics.get_meter("sanyuan_agent.usage")
_tokens_counter = _meter.create_counter(
    name="neuroagent_llm_tokens_total",
    description="Total LLM tokens consumed, labelled by tenant/model/provider",
    unit="tokens",
)


# Flush callback type ---------------------------------------------------------

FlushCallback = Callable[[list[LLMCallUsage]], Awaitable[None]]


# Tracker ---------------------------------------------------------------------


class UsageTracker:
    """Thread-safe token-usage tracker with async background flush.

    The synchronous ``enqueue`` method is safe to call from LiteLLM
    callbacks (which run in any thread).  A background asyncio task
    periodically drains the internal deque and invokes the *flush_callback*
    for batched writes.

    Parameters
    ----------
    flush_callback:
        Async function that receives a list of :class:`LLMCallUsage` records
        and persists them (e.g. to ``llm_call_costs`` DB table).
    flush_interval:
        Seconds between background flush cycles.
    max_pending:
        Max pending records before a WARN is logged.  Records are **not**
        dropped — the enqueue is always non-blocking.
    """

    def __init__(
        self,
        flush_callback: FlushCallback | None = None,
        *,
        flush_interval: float = _FLUSH_INTERVAL,
        max_pending: int = _MAX_PENDING,
    ) -> None:
        self._flush_callback = flush_callback
        self._flush_interval = flush_interval
        self._max_pending = max_pending
        # Thread-safe deque for cross-thread enqueue (from litellm callback)
        self._pending: collections.deque[LLMCallUsage] = collections.deque()
        self._running = False
        self._task: asyncio.Task[None] | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._flush_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        # Final flush
        await self.flush()

    # -- enqueue (thread-safe, non-blocking) -------------------------------

    def enqueue(self, record: LLMCallUsage) -> None:
        """Add a usage record.  Safe to call from any thread."""
        self._pending.append(record)
        size = len(self._pending)
        if size >= self._max_pending:
            logger.warning("usage tracker queue full (%d pending)", size)

        # OTel metric
        _tokens_counter.add(
            record.total_tokens,
            attributes={
                "tenant": record.tenant_id,
                "model": record.model,
                "provider": record.provider,
            },
        )

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    # -- flush --------------------------------------------------------------

    async def flush(self) -> int:
        """Drain pending records and invoke the flush callback.

        Returns the number of records flushed.
        """
        batch: list[LLMCallUsage] = []
        while self._pending:
            batch.append(self._pending.popleft())
        if batch and self._flush_callback is not None:
            try:
                await self._flush_callback(batch)
            except Exception:
                logger.exception("usage tracker flush error")
                # Re-enqueue on failure so records aren't lost
                for r in reversed(batch):
                    self._pending.appendleft(r)
        return len(batch)

    async def _flush_loop(self) -> None:
        while self._running:
            try:
                await self.flush()
            except Exception:
                logger.exception("usage tracker flush loop error")
            await asyncio.sleep(self._flush_interval)
