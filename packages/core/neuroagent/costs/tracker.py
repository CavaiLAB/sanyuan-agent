"""CostTracker — async queue + background worker for LLM call cost recording."""

from __future__ import annotations

import asyncio
import collections
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from opentelemetry import metrics

logger = logging.getLogger(__name__)

_MAX_PENDING = 10_000
_FLUSH_INTERVAL = 5.0  # seconds


@dataclass(frozen=True, slots=True)
class LLMCallCost:
    """Cost record for a single LLM completion call."""

    run_id: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_cents: float  # cost in cents (USD)
    timestamp: float
    tenant_id: str


# OTel metric -----------------------------------------------------------------

_meter = metrics.get_meter("neuroagent.costs")
_cost_counter = _meter.create_counter(
    name="neuroagent_llm_cost_cents_total",
    description="Total LLM cost in cents, labelled by tenant/model/provider",
    unit="cents",
)


# Flush callback type ---------------------------------------------------------

FlushCallback = Callable[[list[LLMCallCost]], Awaitable[None]]


# Tracker ---------------------------------------------------------------------


class CostTracker:
    """Thread-safe cost tracker with async background flush.

    The synchronous ``enqueue`` method is safe to call from LiteLLM
    callbacks (which run in any thread).  A background asyncio task
    periodically drains the internal deque and invokes the *flush_callback*
    for batched writes.

    Parameters
    ----------
    flush_callback:
        Async function that receives a list of :class:`LLMCallCost` records
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
        self._pending: collections.deque[LLMCallCost] = collections.deque()
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

    def enqueue(self, record: LLMCallCost) -> None:
        """Add a cost record.  Safe to call from any thread."""
        self._pending.append(record)
        size = len(self._pending)
        if size >= self._max_pending:
            logger.warning("cost tracker queue full (%d pending)", size)

        # OTel metric
        _cost_counter.add(
            record.cost_cents,
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
        batch: list[LLMCallCost] = []
        while self._pending:
            batch.append(self._pending.popleft())
        if batch and self._flush_callback is not None:
            try:
                await self._flush_callback(batch)
            except Exception:
                logger.exception("cost tracker flush error")
                # Re-enqueue on failure so records aren't lost
                for r in reversed(batch):
                    self._pending.appendleft(r)
        return len(batch)

    async def _flush_loop(self) -> None:
        while self._running:
            try:
                await self.flush()
            except Exception:
                logger.exception("cost tracker flush loop error")
            await asyncio.sleep(self._flush_interval)


# -- helpers ------------------------------------------------------------------


def compute_cost(
    *,
    input_tokens: int,
    output_tokens: int,
    cost_input_per_1k: float = 0.0,
    cost_output_per_1k: float = 0.0,
    litellm_cost: float | None = None,
) -> float:
    """Compute cost in cents, preferring LiteLLM's built-in cost.

    Falls back to ``input_tokens * cost_input_per_1k / 1000 +
    output_tokens * cost_output_per_1k / 1000`` when *litellm_cost* is
    ``None`` or zero.
    """
    if litellm_cost is not None and litellm_cost > 0:
        return litellm_cost * 100  # LiteLLM returns USD
    # Manual calculation: rates are per-1k tokens in USD
    manual = (input_tokens * cost_input_per_1k + output_tokens * cost_output_per_1k) / 1000
    return manual * 100  # convert USD to cents
