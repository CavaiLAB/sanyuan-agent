"""BrowserSession — lifecycle management with thread_id-based reuse."""

from __future__ import annotations

import base64
import logging
import time
from typing import Any

from neuroagent.artifacts.client import StorageArtifactClient
from neuroagent.tools.web_fetch import _ssrf_guard

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 10_000  # ms
_VIEWPORT: dict[str, int] = {"width": 1280, "height": 720}
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)

# Module-level session cache: thread_id → BrowserSession
_session_cache: dict[str, BrowserSession] = {}


def _safe_url(url: str) -> None:
    """Validate URL for browser navigation — SSRF + scheme check."""
    _ssrf_guard(url)
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"Blocked: only http(s) URLs allowed, got {url}")


class BrowserSession:
    """Manages a headless Chromium context via Playwright.

    Supports two connection modes:
    - **CDP mode**: Connect to a remote Chromium via CDP WebSocket (sandbox).
    - **Local mode**: Launch a local Chromium instance (dev/test).

    Sessions are cached by ``thread_id`` via ``get_or_create()``.
    """

    def __init__(
        self,
        *,
        artifact_client: StorageArtifactClient | None = None,
        tenant_id: str = "",
        timeout_ms: int = _DEFAULT_TIMEOUT,
        cdp_url: str | None = None,
    ) -> None:
        self._artifact_client = artifact_client
        self._tenant_id = tenant_id
        self._timeout_ms = timeout_ms
        self._cdp_url = cdp_url
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None

    @classmethod
    def get_or_create(
        cls,
        thread_id: str,
        **kwargs: Any,
    ) -> BrowserSession:
        """Return cached session for *thread_id*, or create a new one."""
        if thread_id not in _session_cache:
            _session_cache[thread_id] = cls(**kwargs)
        return _session_cache[thread_id]

    @classmethod
    def _cached(cls, thread_id: str) -> BrowserSession | None:
        return _session_cache.get(thread_id)

    @classmethod
    def _clear_cache(cls, thread_id: str) -> None:
        _session_cache.pop(thread_id, None)

    @classmethod
    def _cache_size(cls) -> int:
        return len(_session_cache)

    async def _ensure_page(self) -> Any:
        """Lazily create/reuse a browser page."""
        if self._page is not None:
            return self._page

        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()

        if self._cdp_url:
            self._browser = await self._playwright.chromium.connect_over_cdp(
                self._cdp_url,
                timeout=_CONNECT_TIMEOUT,
            )
            contexts = self._browser.contexts
            if contexts:
                self._context = contexts[0]
            else:
                self._context = await self._browser.new_context(
                    viewport=_VIEWPORT,
                    user_agent=_USER_AGENT,
                )
        else:
            self._browser = await self._playwright.chromium.launch(headless=True)
            self._context = await self._browser.new_context(
                viewport=_VIEWPORT,
                user_agent=_USER_AGENT,
            )

        pages = self._context.pages
        if pages:
            self._page = pages[0]
        else:
            self._page = await self._context.new_page()
        return self._page

    # ---- Actions ----

    async def navigate(self, url: str) -> dict[str, Any]:
        _safe_url(url)
        page = await self._ensure_page()
        try:
            await page.goto(
                url, timeout=self._timeout_ms, wait_until="domcontentloaded"
            )
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
        """Take a screenshot. Returns base64 PNG and optionally uploads as artifact."""
        page = await self._ensure_page()
        try:
            png_bytes = await page.screenshot(
                timeout=self._timeout_ms, type="png"
            )
        except Exception as exc:
            return {"success": False, "error": str(exc)}

        b64 = base64.b64encode(png_bytes).decode("ascii")

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
            "screenshot_base64": b64,
            "screenshot_artifact_id": artifact_id,
        }

    async def read_dom(self, selector: str = "body") -> dict[str, Any]:
        """Read the inner HTML/text of the first element matching *selector*."""
        page = await self._ensure_page()
        try:
            element = await page.query_selector(selector)
            if element is None:
                return {
                    "success": False,
                    "error": f"Element not found: {selector}",
                }
            inner_text = await element.inner_text()
            html = await element.inner_html()
        except Exception as exc:
            return {"success": False, "error": str(exc)}
        return {
            "success": True,
            "title": await page.title(),
            "url": page.url,
            "text": inner_text,
            "html": html,
        }

    # ---- Lifecycle ----

    async def close(self) -> None:
        """Release all browser resources."""
        if self._context is not None:
            try:
                await self._context.close()
            except Exception:
                pass
        if self._browser is not None:
            try:
                await self._browser.close()
            except Exception:
                pass
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception:
                pass
        self._page = self._context = self._browser = self._playwright = None


_CONNECT_TIMEOUT = 15_000  # ms
