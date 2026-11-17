"""Browser tool registration — wires BrowserSession into the ToolRegistry."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from neuroagent.tools.block_detect import detect_block
from neuroagent.tools.browser.session import BrowserSession
from neuroagent.tools.provider import ToolSpec, spec_from_callable
from neuroagent.tools.registry import ToolRegistry, callable_to_schema

logger = logging.getLogger(__name__)

SessionFactory = Callable[[str], Awaitable[BrowserSession]]


def _annotate_block(result: dict[str, Any]) -> dict[str, Any]:
    """Tag a successful page result with ``blocked``/``block_reason``.

    A soft anti-bot page navigates fine (success + HTTP 200) but its content is a
    captcha/拦截 wall — surface that structurally instead of as a clean page.
    """
    if not isinstance(result, dict) or not result.get("success"):
        return result
    blocked, reason = detect_block(
        url=str(result.get("url") or ""),
        title=str(result.get("title") or ""),
        content=str(result.get("text") or result.get("content") or ""),
        html=str(result.get("html") or ""),
    )
    return {**result, "blocked": blocked, "block_reason": reason}


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
            return _annotate_block(await sess.navigate(url))
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
            return _annotate_block(await sess.read_dom(selector))
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


# Descriptive provider metadata per browser tool (PR7). Per provider.py this is
# advisory only - it informs policy building / UI / audit and never gates
# execution. Reads stay read (no hint); page-mutating interactions are write, and
# the genuinely risky ones (click / type / download) default to confirm.
_BROWSER_TOOL_META: dict[str, dict[str, Any]] = {
    "browser_navigate": {"capabilities": ("browser", "navigate"), "side_effect": "write"},
    "browser_click": {
        "capabilities": ("browser", "click"),
        "side_effect": "write",
        "policy_hint": "confirm",
    },
    "browser_type": {
        "capabilities": ("browser", "type"),
        "side_effect": "write",
        "policy_hint": "confirm",
    },
    "browser_screenshot": {
        "capabilities": ("browser", "screenshot"),
        "side_effect": "read",
        "artifact_output": True,  # may emit an artifact id
    },
    "browser_read_dom": {"capabilities": ("browser", "read"), "side_effect": "read"},
    "browser_accessibility": {"capabilities": ("browser", "read"), "side_effect": "read"},
    "browser_download": {
        "capabilities": ("browser", "download"),
        "side_effect": "write",
        "policy_hint": "confirm",  # captures an untrusted download into quarantine
    },
    "browser_downloads": {"capabilities": ("browser", "list"), "side_effect": "read"},
}


class BrowserToolProvider:
    """Thin ToolProvider wrapper over :func:`build_browser_tools` (PR7).

    Yields the existing browser tools as specs with ``category="browser"`` plus
    per-tool side-effect / policy-hint metadata. It changes **no behavior and no
    schema**: each tool's callable and its reflected schema come straight from
    ``build_browser_tools``; only descriptive metadata is attached, and per PR1
    that metadata never gates execution (``ToolPolicy`` / ``RiskGate`` remain the
    runtime authority).

    Selectable through the provider path: reference the provider in
    ``AgentSpec.providers`` and enable individual ``browser_*`` names via
    ``AgentSpec.tools`` / ``tool_sets``.
    """

    def __init__(self, session_factory: SessionFactory) -> None:
        self._session_factory = session_factory

    def provide(self) -> list[ToolSpec]:
        # Build into a scratch registry, then re-emit the registered callables +
        # schemas verbatim as specs (schema passed through, never re-derived).
        scratch = ToolRegistry()
        build_browser_tools(scratch, self._session_factory)
        specs: list[ToolSpec] = []
        for name in scratch.list_tools():
            meta = _BROWSER_TOOL_META.get(name, {})
            specs.append(
                spec_from_callable(
                    scratch.get(name),
                    name=name,
                    schema=scratch.get_schema(name),
                    category="browser",
                    capabilities=meta.get("capabilities", ("browser",)),
                    side_effect=meta.get("side_effect", "write"),
                    artifact_output=meta.get("artifact_output", False),
                    policy_hint=meta.get("policy_hint"),
                )
            )
        return specs
