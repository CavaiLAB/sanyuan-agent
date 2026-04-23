"""Async budget client that talks to the C# Platform service.

Both functions are **fail-open**: if the Platform service is unreachable
or mis-configured the agent is allowed to continue and a warning is logged.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_PLATFORM_URL = os.environ.get("NEUROAGENT_PLATFORM_URL", "")


class BudgetExceeded(Exception):
    """Raised when the budget hard-stop (100 %) has been reached."""

    def __init__(self, used_percent: float, state: str | None) -> None:
        self.used_percent = used_percent
        self.state = state
        super().__init__(
            f"Budget exceeded: {used_percent:.1f}% used (state={state})"
        )


async def check_budget(
    tenant_id: str,
    user_id: str,
    run_id: str,
    platform_url: str = DEFAULT_PLATFORM_URL,
) -> dict[str, Any]:
    """Check budget before an LLM call.

    Returns a dict with keys ``allowed``, ``usedPercent``,
    ``warningEmitted``, and ``state``.  If the service is down the
    function returns ``{"allowed": True, ...}`` so the agent is not
    blocked by a transient failure.
    """
    if not platform_url or not run_id:
        return {
            "allowed": True,
            "usedPercent": 0.0,
            "warningEmitted": False,
            "state": None,
        }

    url = f"{platform_url}/api/v1/budget/check"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                url,
                json={"tenantId": tenant_id, "userId": user_id, "runId": run_id},
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        logger.warning("budget check failed (fail-open): %s", exc)
        return {
            "allowed": True,
            "usedPercent": 0.0,
            "warningEmitted": False,
            "state": None,
        }


def accumulate_budget(
    tenant_id: str,
    user_id: str,
    run_id: str,
    cost: float,
    platform_url: str = DEFAULT_PLATFORM_URL,
) -> None:
    """Add *cost* to the budget counters.

    This is a **fire-and-forget** synchronous call intended for
    LiteLLM success callbacks.  Errors are logged and swallowed.
    """
    if not platform_url or not run_id:
        return

    url = f"{platform_url}/api/v1/budget/accumulate"
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(
                url,
                json={
                    "tenantId": tenant_id,
                    "userId": user_id,
                    "runId": run_id,
                    "cost": cost,
                },
            )
            resp.raise_for_status()
    except Exception as exc:
        logger.warning("budget accumulate failed: %s", exc)
