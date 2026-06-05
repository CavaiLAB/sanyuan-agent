"""CDP client — connects to sandbox Chromium via CDP WebSocket."""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_CDP_LAUNCH_CMD = [
    "chromium",
    "--headless",
    "--no-sandbox",
    "--disable-gpu",
    "--remote-debugging-port=9222",
    "--disable-dev-shm-usage",
]
_FALLBACK_CDP_PORT = 9222
_CONNECT_TIMEOUT_S = 15


class CDPClient:
    """Obtains a CDP WebSocket URL from the sandbox supervisor and connects.

    Flow:
    1. POST /v2/sandboxes/{id}/browser/cdp → get ``cdp_ws_url``.
    2. If the supervisor returns 501 (not yet implemented), fall back to
       launching Chromium inside the sandbox via ``exec_process`` and
       constructing the CDP URL from the container's mapped port.
    """

    def __init__(
        self,
        *,
        supervisor_url: str,
        sandbox_id: str,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._supervisor_url = supervisor_url.rstrip("/")
        self._sandbox_id = sandbox_id
        self._http = http_client

    async def _get_http_client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=30.0)
        return self._http

    async def resolve_cdp_url(self) -> str:
        """Return a usable CDP WebSocket URL for the sandbox's Chromium."""
        client = await self._get_http_client()
        url = f"{self._supervisor_url}/v2/sandboxes/{self._sandbox_id}/browser/cdp"

        try:
            resp = await client.post(url)
            if resp.status_code == 200:
                data = resp.json()
                # The supervisor returns its *proxy* ws url (camelCase API), so
                # Playwright connects through the audited proxy, not the container.
                cdp_url: str = data["cdpWsUrl"]
                logger.info("CDP URL from supervisor proxy: %s", cdp_url)
                return cdp_url
        except httpx.HTTPError as exc:
            logger.warning("CDP endpoint request failed: %s", exc)

        # Fallback: exec chromium in sandbox, then build CDP URL
        return await self._fallback_launch(client)

    async def _fallback_launch(self, client: httpx.AsyncClient) -> str:
        """Start Chromium inside the sandbox and return CDP URL."""
        exec_url = (
            f"{self._supervisor_url}/v2/sandboxes/{self._sandbox_id}/processes"
        )
        payload = {
            "cmd": _CDP_LAUNCH_CMD,
            "cwd": "/tmp",
        }
        resp = await client.post(exec_url, json=payload)
        resp.raise_for_status()

        cdp_url = f"ws://localhost:{_FALLBACK_CDP_PORT}/devtools/browser/"
        logger.info("CDP fallback URL (sandbox exec): %s", cdp_url)
        return cdp_url

    async def connect_over_cdp(self, cdp_url: str | None = None) -> Any:
        """Connect Playwright to Chromium via CDP.

        Returns the Playwright ``Browser`` object.
        """
        from playwright.async_api import async_playwright

        if cdp_url is None:
            cdp_url = await self.resolve_cdp_url()

        pw = await async_playwright().start()
        browser = await pw.chromium.connect_over_cdp(
            cdp_url,
            timeout=_CONNECT_TIMEOUT_S * 1000,
        )
        return browser, pw

    async def close(self) -> None:
        """Shut down the internal httpx client if we own it."""
        if self._http is not None:
            await self._http.aclose()
            self._http = None
