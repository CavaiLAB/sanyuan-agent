"""BrowserSession — lifecycle management with thread_id-based reuse."""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

from sanyuan_agent.artifacts.client import StorageArtifactClient
from sanyuan_agent.tools.web_fetch import _ssrf_guard

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 10_000  # ms
_VIEWPORT: dict[str, int] = {"width": 1280, "height": 720}
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)
_DEFAULT_QUARANTINE_DIR = "/tmp/iclaw-quarantine"

# Module-level session cache: tenant_id + run/thread_id → BrowserSession
_session_cache: dict[str, BrowserSession] = {}

URLValidator = Callable[[str], Awaitable[str | None]]
CloseCallback = Callable[[], Awaitable[None]]


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
        quarantine_dir: str | None = None,
        url_validator: URLValidator | None = None,
        close_callback: CloseCallback | None = None,
    ) -> None:
        self._artifact_client = artifact_client
        self._tenant_id = tenant_id
        self._timeout_ms = timeout_ms
        self._cdp_url = cdp_url
        self._url_validator = url_validator
        self._close_callback = close_callback
        self._last_policy_error: str | None = None
        # Downloads never land in the workspace directly: they are written to an
        # isolated quarantine dir and flagged unscanned (Sandbox Spec — download
        # quarantine, Phase 1 safety baseline). Release is an explicit step.
        self._quarantine_dir = quarantine_dir or _DEFAULT_QUARANTINE_DIR
        self._downloads: list[dict[str, Any]] = []
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
        """Return a tenant+thread scoped session, or create a new one."""
        key = cls._cache_key(thread_id, str(kwargs.get("tenant_id") or ""))
        if key not in _session_cache:
            _session_cache[key] = cls(**kwargs)
        return _session_cache[key]

    @classmethod
    def _cached(cls, thread_id: str, tenant_id: str = "") -> BrowserSession | None:
        return _session_cache.get(cls._cache_key(thread_id, tenant_id))

    @classmethod
    def _clear_cache(cls, thread_id: str, tenant_id: str = "") -> None:
        _session_cache.pop(cls._cache_key(thread_id, tenant_id), None)

    @classmethod
    def _cache_size(cls) -> int:
        return len(_session_cache)

    @staticmethod
    def _cache_key(thread_id: str, tenant_id: str) -> str:
        return f"{tenant_id}:{thread_id}" if tenant_id else thread_id

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
                    accept_downloads=True,
                )
        else:
            self._browser = await self._playwright.chromium.launch(headless=True)
            self._context = await self._browser.new_context(
                viewport=_VIEWPORT,
                user_agent=_USER_AGENT,
                accept_downloads=True,
            )

        if getattr(self, "_url_validator", None) is not None:
            await self._context.route("**/*", self._route_with_policy)

        pages = self._context.pages
        if pages:
            self._page = pages[0]
        else:
            self._page = await self._context.new_page()
        return self._page

    async def _validate_url(self, url: str) -> None:
        if not url.startswith(("http://", "https://")):
            raise ValueError("Blocked: only http(s) URLs allowed")
        validator = getattr(self, "_url_validator", None)
        if validator is None:
            _safe_url(url)
            return
        reason = await validator(url)
        if reason:
            self._last_policy_error = reason
            raise ValueError(f"Blocked by URL policy: {reason}")

    async def _route_with_policy(self, route: Any, request: Any) -> None:
        url = str(request.url)
        if not url.startswith(("http://", "https://")):
            await route.continue_()
            return
        try:
            await self._validate_url(url)
        except ValueError:
            await route.abort("blockedbyclient")
            return
        await route.continue_()

    # ---- Actions ----

    async def navigate(self, url: str) -> dict[str, Any]:
        self._last_policy_error = None
        try:
            await self._validate_url(url)
        except ValueError:
            if getattr(self, "_url_validator", None) is None:
                raise
            return {
                "success": False,
                "policy_denied": True,
                "error_code": self._last_policy_error or "browser_url_policy_denied",
            }
        page = await self._ensure_page()
        try:
            await page.goto(url, timeout=self._timeout_ms, wait_until="domcontentloaded")
        except Exception as exc:
            policy_error = self._current_policy_error()
            if policy_error:
                return {
                    "success": False,
                    "policy_denied": True,
                    "error_code": policy_error,
                }
            return {"success": False, "error": str(exc)}
        return {
            "success": True,
            "title": await page.title(),
            "url": page.url,
        }

    def _current_policy_error(self) -> str | None:
        return getattr(self, "_last_policy_error", None)

    async def wait_until_ready(self, timeout_ms: int = _DEFAULT_TIMEOUT) -> dict[str, Any]:
        """Best-effort network-idle wait; a busy analytics stream is not fatal."""
        page = await self._ensure_page()
        try:
            await page.wait_for_load_state("networkidle", timeout=timeout_ms)
            return {"success": True, "timed_out": False, "url": page.url}
        except Exception:
            return {"success": True, "timed_out": True, "url": page.url}

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
            png_bytes = await page.screenshot(timeout=self._timeout_ms, type="png")
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

    async def accessibility_tree(self) -> dict[str, Any]:
        """Return the page's accessibility tree (structured page model)."""
        page = await self._ensure_page()
        try:
            snapshot = await page.accessibility.snapshot()
        except Exception as exc:
            return {"success": False, "error": str(exc)}
        return {
            "success": True,
            "title": await page.title(),
            "url": page.url,
            "tree": snapshot,
        }

    # ---- Downloads (quarantined) ----

    async def download(self, trigger_selector: str) -> dict[str, Any]:
        """Click *trigger_selector* and capture the resulting download into the
        quarantine dir — never the workspace. Returns the quarantine record."""
        page = await self._ensure_page()
        try:
            async with page.expect_download(timeout=self._timeout_ms) as dl_info:
                await page.click(trigger_selector, timeout=self._timeout_ms)
            download = await dl_info.value
            record = await self._quarantine_download(download)
        except Exception as exc:
            return {"success": False, "error": str(exc)}
        return {"success": True, **record}

    async def _quarantine_download(self, download: Any) -> dict[str, Any]:
        """Save a Playwright download into the quarantine dir, flagged unscanned.

        The file is hashed and recorded; it is NOT placed in the workspace, so a
        run cannot use downloaded bytes until an explicit (Phase 2) release step.
        """
        os.makedirs(self._quarantine_dir, exist_ok=True)
        suggested = download.suggested_filename or "download"
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", suggested) or "download"
        dest = os.path.join(self._quarantine_dir, f"{len(self._downloads):04d}-{safe}")
        await download.save_as(dest)
        try:
            with open(dest, "rb") as fh:
                data = fh.read()
        except OSError:
            data = b""
        record = {
            "suggestedFilename": suggested,
            "quarantinePath": dest,
            "sizeBytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "status": "quarantined",
            "scanned": False,
        }
        self._downloads.append(record)
        return record

    def list_downloads(self) -> list[dict[str, Any]]:
        """Return all downloads captured into quarantine this session."""
        return list(self._downloads)

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
        close_callback = getattr(self, "_close_callback", None)
        if close_callback is not None:
            callback, self._close_callback = close_callback, None
            try:
                await callback()
            except Exception:
                logger.warning("browser sandbox cleanup failed", exc_info=True)


_CONNECT_TIMEOUT = 15_000  # ms
