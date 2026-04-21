"""Web-fetch tool builder — httpx GET + trafilatura extraction."""

from __future__ import annotations

import ipaddress
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from neuroagent.tools.registry import ToolRegistry, callable_to_schema

WebFetchTool = Callable[[str], Awaitable[dict[str, Any]]]


def build_web_fetch_tool(
    registry: ToolRegistry,
    *,
    timeout: int = 20,
    max_bytes: int = 2_000_000,
) -> WebFetchTool:
    """Return a bound web-fetch tool with SSRF guard.

    The tool is automatically registered in *registry* under ``"web_fetch"``.
    """

    async def web_fetch(url: str) -> dict[str, Any]:
        _ssrf_guard(url)
        return await _fetch(url, timeout, max_bytes)

    registry.register("web_fetch", web_fetch, callable_to_schema(web_fetch))
    return web_fetch


def _ssrf_guard(url: str) -> None:
    """Reject private / loopback / internal URLs."""
    parsed = httpx.URL(url)
    host = parsed.host
    if not host:
        raise ValueError("Invalid URL: missing host")

    # Block localhost variants
    if host.lower() in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("SSRF: localhost access denied")

    # Block private IP ranges (RFC 1918 + link-local)
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        # Not an IP address — resolve is deferred to httpx (DNS rebinding risk
        # remains but mitigated by not resolving ourselves).  We intentionally
        # do *not* block *.internal or *.local here to avoid false positives.
        pass
    else:
        if ip.is_private or ip.is_loopback or ip.is_link_local:
            raise ValueError(f"SSRF: private IP access denied ({host})")


def _is_allowed_content_type(content_type: str | None) -> bool:
    if content_type is None:
        return False
    return content_type.startswith("text/")


async def _fetch(
    url: str,
    timeout: int,
    max_bytes: int,
) -> dict[str, Any]:
    from trafilatura import extract

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
        if not _is_allowed_content_type(content_type):
            raise ValueError(
                f"Non-text content-type refused: {content_type}"
            )

        raw = resp.content[:max_bytes]
        decoded = raw.decode("utf-8", errors="replace")

        if content_type.startswith("text/plain"):
            text: str | None = decoded
        else:
            text = extract(
                decoded,
                url=url,
                include_comments=False,
                include_tables=False,
                no_fallback=True,
            )

        safe_text = text or ""
        return {
            "url": str(resp.url),
            "title": safe_text.strip().splitlines()[0] if safe_text else "",
            "content": safe_text,
            "status_code": resp.status_code,
        }
