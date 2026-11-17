"""Per-tenant/run Web tool budgets, separate from LLM token quotas."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class WebRunBudget:
    """Shared counters and concurrency slots for one tenant + app run."""

    def __init__(
        self,
        *,
        max_searches: int = 5,
        max_firecrawl_pages: int = 3,
        max_browser_sessions: int = 1,
        max_concurrency: int = 4,
    ) -> None:
        limits = {
            "search": max_searches,
            "firecrawl": max_firecrawl_pages,
            "browser": max_browser_sessions,
        }
        if any(value < 0 for value in limits.values()):
            raise ValueError("Web run limits must not be negative")
        if max_concurrency < 1:
            raise ValueError("Web max_concurrency must be positive")
        self._limits = limits
        self._counts = {name: 0 for name in limits}
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def reserve(self, resource: str) -> bool:
        if resource not in self._limits:
            raise ValueError(f"unknown Web budget resource: {resource}")
        async with self._lock:
            if self._counts[resource] >= self._limits[resource]:
                return False
            self._counts[resource] += 1
            return True

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        async with self._semaphore:
            yield

    async def snapshot(self) -> dict[str, int]:
        async with self._lock:
            return dict(self._counts)


__all__ = ["WebRunBudget"]
