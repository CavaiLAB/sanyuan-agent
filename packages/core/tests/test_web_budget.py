"""Shared Web run quota and concurrency tests."""

from __future__ import annotations

import asyncio

from sanyuan_agent.tools.web_budget import WebRunBudget


async def test_resource_limits_are_independent_and_fail_closed() -> None:
    budget = WebRunBudget(
        max_searches=1,
        max_firecrawl_pages=1,
        max_browser_sessions=1,
    )

    for resource in ("search", "firecrawl", "browser"):
        assert await budget.reserve(resource) is True
        assert await budget.reserve(resource) is False

    assert await budget.snapshot() == {"search": 1, "firecrawl": 1, "browser": 1}


async def test_concurrency_slot_is_shared() -> None:
    budget = WebRunBudget(max_concurrency=1)
    active = 0
    peak = 0

    async def work() -> None:
        nonlocal active, peak
        async with budget.slot():
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0)
            active -= 1

    await asyncio.gather(work(), work(), work())

    assert peak == 1
