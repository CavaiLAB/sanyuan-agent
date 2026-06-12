"""Background GC scheduler — asyncio periodic task."""

from __future__ import annotations

import asyncio
import logging

from neuroagent.gc.collector import (
    GCRunStats,
    SandboxCollector,
    SensoryCollector,
)
from neuroagent.gc.longterm import LongtermCollector

logger = logging.getLogger(__name__)


class GCConfig:
    """GC tuning knobs — initialise from env or defaults."""

    def __init__(
        self,
        *,
        interval_seconds: float = 300.0,
        soft_threshold_seconds: float = 1800.0,
        hard_threshold_seconds: float = 604800.0,
    ) -> None:
        self.interval_seconds = interval_seconds
        self.soft_threshold_seconds = soft_threshold_seconds
        self.hard_threshold_seconds = hard_threshold_seconds


class GCScheduler:
    """Runs collectors on a fixed interval."""

    def __init__(
        self,
        sandbox_collector: SandboxCollector,
        sensory_collector: SensoryCollector | None = None,
        *,
        longterm_collector: LongtermCollector | None = None,
        config: GCConfig | None = None,
    ) -> None:
        self._sandbox = sandbox_collector
        self._sensory = sensory_collector
        self._longterm = longterm_collector
        self._config = config or GCConfig()
        self._task: asyncio.Task[None] | None = None
        self._running = False

    @property
    def stats(self) -> GCRunStats:
        return self._sandbox.stats

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            "GC scheduler started (interval=%ss)", self._config.interval_seconds,
        )

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("GC scheduler stopped")

    async def run_once(self) -> GCRunStats:
        stats = await self._sandbox.collect()
        if self._sensory is not None:
            expired = await self._sensory.collect()
            stats.last_sensory_expired_count = expired
        if self._longterm is not None:
            lt_stats = await self._longterm.collect()
            stats.last_longterm_expired_count = lt_stats.last_expired_count
            stats.last_longterm_evicted_count = lt_stats.last_evicted_count
        return stats

    async def _run_loop(self) -> None:
        while self._running:
            try:
                await self.run_once()
            except Exception:
                logger.exception("GC run failed")
            await asyncio.sleep(self._config.interval_seconds)
