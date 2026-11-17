"""Web-fetch tool builder — httpx GET + trafilatura extraction."""

from __future__ import annotations

import ipaddress
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import httpx

from sanyuan_agent.tools.block_detect import detect_block
from sanyuan_agent.tools.registry import ToolRegistry, callable_to_schema
from sanyuan_agent.tools.web_reader import ReaderRouter
from sanyuan_agent.tools.web_reader_models import FetchDocument, FetchRequest

WebFetchTool = Callable[[str], Awaitable[dict[str, Any]]]
FetchDocumentTransform = Callable[[FetchDocument], FetchDocument]


def build_web_fetch_tool(
    registry: ToolRegistry,
    *,
    timeout: int = 20,
    max_bytes: int = 2_000_000,
    reader_router: ReaderRouter | None = None,
    document_transform: FetchDocumentTransform | None = None,
) -> WebFetchTool:
    """Return a bound web-fetch tool with SSRF guard.

    The tool is automatically registered in *registry* under ``"web_fetch"``.
    """

    async def web_fetch(url: str) -> dict[str, Any]:
        if reader_router is not None:
            document = await reader_router.fetch(
                FetchRequest(
                    url=url,
                    timeout_seconds=timeout,
                    max_bytes=max_bytes,
                )
            )
            if document_transform is not None:
                document = document_transform(document)
            return document.to_tool_result()
        _ssrf_guard(url)
        return await _fetch(url, timeout, max_bytes)

    registry.register("web_fetch", web_fetch, callable_to_schema(web_fetch))
    return web_fetch


def _ssrf_guard(url: str) -> None:
    """Synchronous compatibility guard; production also uses URLClassifier."""
    try:
        parsed = urlsplit(url.strip())
        host = parsed.hostname
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("Invalid URL") from exc
    if not host:
        raise ValueError("Invalid URL: missing host")
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("SSRF: only http(s) URLs are allowed")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("SSRF: URL userinfo is denied")

    # Block localhost variants
    lowered_host = host.lower().rstrip(".")
    if lowered_host in {"localhost", "127.0.0.1", "::1"} or lowered_host.endswith(
        (".internal", ".local", ".localhost", ".home", ".lan")
    ):
        raise ValueError("SSRF: localhost access denied")
    query_names = {
        name.casefold() for name, _value in parse_qsl(parsed.query, keep_blank_values=True)
    }
    sensitive_names = {
        "access_token",
        "apikey",
        "api_key",
        "auth",
        "authorization",
        "key",
        "ossaccesskeyid",
        "password",
        "secret",
        "sig",
        "signature",
        "token",
    }
    if query_names & sensitive_names or any(name.startswith("x-amz-") for name in query_names):
        raise ValueError("SSRF: credential-bearing URLs are denied")

    # Block private IP ranges (RFC 1918 + link-local)
    try:
        ip = ipaddress.ip_address(lowered_host)
    except ValueError:
        # Not an IP address — resolve is deferred to httpx (DNS rebinding risk
        # remains but mitigated by not resolving ourselves).  We intentionally
        # do *not* block *.internal or *.local here to avoid false positives.
        pass
    else:
        if not ip.is_global:
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
    from trafilatura import extract, html2txt

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
                fast=True,
            )
            if not text:
                text = html2txt(decoded)

        safe_text = text or ""
        final_url = str(resp.url)
        page_title = safe_text.strip().splitlines()[0] if safe_text else ""
        # Soft blocks (anti-bot / captcha pages) come back as HTTP 200, so
        # raise_for_status above can't catch them. Flag them structurally for the
        # UI; the front-end heuristic is only a fallback for this signal.
        blocked, block_reason = detect_block(
            url=final_url, title=page_title, content=safe_text
        )
        return {
            "url": final_url,
            "title": page_title,
            "content": safe_text,
            "status_code": resp.status_code,
            "blocked": blocked,
            "block_reason": block_reason,
        }
