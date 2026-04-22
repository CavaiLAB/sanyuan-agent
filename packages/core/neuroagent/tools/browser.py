"""Playwright browser tools — navigate, click, type, screenshot with SSRF guard."""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from neuroagent.artifacts.client import StorageArtifactClient
from neuroagent.tools.registry import ToolRegistry, callable_to_schema
from neuroagent.tools.web_fetch import _ssrf_guard

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 10_000  # ms
_VIEWPORT: dict[str, int] = {"width": 1280, "height": 720}
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)


class BrowserSession:
    """Manages a single headless Chromium context via Playwright.

    Lazily created on first tool call and reused until ``cleanup()``.
    """

    def __init__(
        self,
        *,
        artifact_client: StorageArtifactClient | None = None,
        tenant_id: str = "",
        timeout_ms: int = _DEFAULT_TIMEOUT,
    ) -> None:
        self._artifact_client = artifact_client
        self._tenant_id = tenant_id
        self._timeout_ms = timeout_ms
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None

    async def _ensure_page(self) -> Any:
        if self._page is not None:
            return self._page
        from playwright.async_api import async_playwright  # type: ignore[import-not-found]

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=True)
        self._context = await self._browser.new_context(
            viewport=_VIEWPORT,
            user_agent=_USER_AGENT,
        )
        self._page = await self._context.new_page()
        return self._page

    async def cleanup(self) -> None:
        if self._context is not None:
            await self._context.close()
        if self._browser is not None:
            await self._browser.close()
        if self._playwright is not None:
            await self._playwright.stop()
        self._page = self._context = self._browser = self._playwright = None

    def _safe_url(self, url: str) -> None:
        _ssrf_guard(url)
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Blocked: only http(s) URLs allowed, got {url}")

    async def navigate(self, url: str) -> dict[str, Any]:
        self._safe_url(url)
        page = await self._ensure_page()
        try:
            await page.goto(url, timeout=self._timeout_ms, wait_until="domcontentloaded")
        except Exception as exc:
            return {"success": False, "error": str(exc)}
        return {
            "success": True,
            "title": await page.title(),
            "url": page.url,
        }

    async def click(self, selector: str) -> dict[str, Any]:
        page = await self._ensure_page()
        try:
            await page.click(selector, timeout=self._timeout_ms)
        except Exception as exc:
            return {"success": False, "error": str(exc)}
        return {
            "success": True,
            "title": await page.title(),
            "url": page.url,
        }

    async def type_text(self, selector: str, text: str) -> dict[str, Any]:
        page = await self._ensure_page()
        try:
            await page.fill(selector, "", timeout=self._timeout_ms)
            await page.type(selector, text, timeout=self._timeout_ms)
        except Exception as exc:
            return {"success": False, "error": str(exc)}
        return {
            "success": True,
            "title": await page.title(),
            "url": page.url,
        }

    async def screenshot(self) -> dict[str, Any]:
        page = await self._ensure_page()
        try:
            png_bytes = await page.screenshot(timeout=self._timeout_ms, type="png")
        except Exception as exc:
            return {"success": False, "error": str(exc)}

        artifact_id = ""
        if self._artifact_client and self._tenant_id:
            ts = int(time.time())
            ref = await self._artifact_client.create_artifact(
                tenant_id=self._tenant_id,
                content_bytes=png_bytes,
                content_type="image/png",
                filename=f"screenshot-{ts}.png",
            )
            artifact_id = ref.id

        return {
            "success": True,
            "title": await page.title(),
            "url": page.url,
            "screenshot_artifact_id": artifact_id,
        }


SessionFactory = Callable[[], Awaitable[BrowserSession]]


def build_browser_tools(
    registry: ToolRegistry,
    session_factory: SessionFactory,
) -> None:
    """Register browser tools in *registry* backed by *session_factory*.

    Each tool lazily obtains a ``BrowserSession`` from the factory on first
    call, enabling deferred Playwright import and avoiding Chromium launch
    until actually needed.
    """
    _session: BrowserSession | None = None

    async def _get_session() -> BrowserSession:
        nonlocal _session
        if _session is None:
            _session = await session_factory()
        return _session

    async def browser_navigate(url: str) -> dict[str, Any]:
        """Navigate the browser to the given URL. Only http(s) URLs are allowed."""
        sess = await _get_session()
        try:
            return await sess.navigate(url)
        except Exception as exc:
            logger.warning("browser_navigate failed: %s", exc)
            return {"success": False, "error": str(exc)}

    async def browser_click(selector: str) -> dict[str, Any]:
        """Click the first element matching the CSS selector."""
        sess = await _get_session()
        try:
            return await sess.click(selector)
        except Exception as exc:
            logger.warning("browser_click failed: %s", exc)
            return {"success": False, "error": str(exc)}

    async def browser_type(selector: str, text: str) -> dict[str, Any]:
        """Type text into the element matching the CSS selector."""
        sess = await _get_session()
        try:
            return await sess.type_text(selector, text)
        except Exception as exc:
            logger.warning("browser_type failed: %s", exc)
            return {"success": False, "error": str(exc)}

    async def browser_screenshot() -> dict[str, Any]:
        """Take a screenshot and upload it as a PNG artifact."""
        sess = await _get_session()
        try:
            return await sess.screenshot()
        except Exception as exc:
            logger.warning("browser_screenshot failed: %s", exc)
            return {"success": False, "error": str(exc)}

    for name, fn in [
        ("browser_navigate", browser_navigate),
        ("browser_click", browser_click),
        ("browser_type", browser_type),
        ("browser_screenshot", browser_screenshot),
    ]:
        registry.register(name, fn, callable_to_schema(fn))
