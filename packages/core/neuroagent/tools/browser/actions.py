"""Browser tool registration — wires BrowserSession into the ToolRegistry."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from neuroagent.tools.browser.session import BrowserSession
from neuroagent.tools.registry import ToolRegistry, callable_to_schema

logger = logging.getLogger(__name__)

SessionFactory = Callable[[str], Awaitable[BrowserSession]]


def build_browser_tools(
    registry: ToolRegistry,
    session_factory: SessionFactory,
) -> None:
    """Register browser tools in *registry* backed by *session_factory*.

    *session_factory* receives a ``thread_id`` string and must return a
    ``BrowserSession``.  The factory is called lazily on first tool invocation
    and the session is reused for subsequent calls within the same closure.
    """

    _session: BrowserSession | None = None
    _thread_id: str | None = None

    async def _get_session(thread_id: str) -> BrowserSession:
        nonlocal _session, _thread_id
        if _session is None or _thread_id != thread_id:
            _session = await session_factory(thread_id)
            _thread_id = thread_id
        return _session

    async def browser_navigate(url: str, thread_id: str = "default") -> dict[str, Any]:
        """Navigate the browser to the given URL. Only http(s) URLs allowed."""
        sess = await _get_session(thread_id)
        try:
            return await sess.navigate(url)
        except Exception as exc:
            logger.warning("browser_navigate failed: %s", exc)
            return {"success": False, "error": str(exc)}

    async def browser_click(
        selector: str, thread_id: str = "default"
    ) -> dict[str, Any]:
        """Click the first element matching the CSS selector."""
        sess = await _get_session(thread_id)
        try:
            return await sess.click(selector)
        except Exception as exc:
            logger.warning("browser_click failed: %s", exc)
            return {"success": False, "error": str(exc)}

    async def browser_type(
        selector: str, text: str, thread_id: str = "default"
    ) -> dict[str, Any]:
        """Type text into the element matching the CSS selector."""
        sess = await _get_session(thread_id)
        try:
            return await sess.type_text(selector, text)
        except Exception as exc:
            logger.warning("browser_type failed: %s", exc)
            return {"success": False, "error": str(exc)}

    async def browser_screenshot(thread_id: str = "default") -> dict[str, Any]:
        """Take a screenshot. Returns base64 PNG and optional artifact ID."""
        sess = await _get_session(thread_id)
        try:
            return await sess.screenshot()
        except Exception as exc:
            logger.warning("browser_screenshot failed: %s", exc)
            return {"success": False, "error": str(exc)}

    async def browser_read_dom(
        selector: str = "body", thread_id: str = "default"
    ) -> dict[str, Any]:
        """Read the inner text and HTML of the element matching the selector."""
        sess = await _get_session(thread_id)
        try:
            return await sess.read_dom(selector)
        except Exception as exc:
            logger.warning("browser_read_dom failed: %s", exc)
            return {"success": False, "error": str(exc)}

    async def browser_accessibility(thread_id: str = "default") -> dict[str, Any]:
        """Return the page's accessibility tree (structured, non-visual page model)."""
        sess = await _get_session(thread_id)
        try:
            return await sess.accessibility_tree()
        except Exception as exc:
            logger.warning("browser_accessibility failed: %s", exc)
            return {"success": False, "error": str(exc)}

    async def browser_download(
        trigger_selector: str, thread_id: str = "default"
    ) -> dict[str, Any]:
        """Click the selector and capture the download into quarantine (never the
        workspace). Returns the quarantine record (path, sha256, status)."""
        sess = await _get_session(thread_id)
        try:
            return await sess.download(trigger_selector)
        except Exception as exc:
            logger.warning("browser_download failed: %s", exc)
            return {"success": False, "error": str(exc)}

    async def browser_downloads(thread_id: str = "default") -> dict[str, Any]:
        """List files this browser session captured into download quarantine."""
        sess = await _get_session(thread_id)
        return {"success": True, "downloads": sess.list_downloads()}

    _tools = [
        ("browser_navigate", browser_navigate),
        ("browser_click", browser_click),
        ("browser_type", browser_type),
        ("browser_screenshot", browser_screenshot),
        ("browser_read_dom", browser_read_dom),
        ("browser_accessibility", browser_accessibility),
        ("browser_download", browser_download),
        ("browser_downloads", browser_downloads),
    ]
    for name, fn in _tools:
        registry.register(name, fn, callable_to_schema(fn))  # type: ignore[arg-type]
