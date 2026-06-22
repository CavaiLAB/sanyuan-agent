"""Link-reader tool: fetch + extract several links in one call, over web_fetch.

link_reader is a low-risk aggregation built on the existing web_fetch primitive
(httpx GET + trafilatura extraction + SSRF guard + soft-block detection). It
reads a small batch of URLs and returns each page's extracted content, isolating
per-URL failures so one bad link does not sink the whole batch.

Two deliberate properties keep it low-risk:

1. No new egress surface and no vendor API. Every URL goes through web_fetch, so
   the SSRF guard and content-type rules are inherited unchanged. This module
   never reads a secret and imports nothing from neuroagent_app.
2. Bounded fan-out. The URL list is de-duplicated and capped (default 10), and a
   per-link error is reported in that link's entry rather than raised.

It is exposed two equivalent ways, mirroring the rest of the tool layer:

- the legacy builder ``build_link_reader_tool(registry, ...)`` (registers under
  ``"link_reader"``), and
- :class:`LinkReaderToolProvider`, so it is selectable through the provider path
  (reference the provider in ``AgentSpec.providers`` and enable ``"link_reader"``
  via ``AgentSpec.tools`` / ``tool_sets``).
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import Any

from neuroagent.tools.provider import ToolSpec, spec_from_callable
from neuroagent.tools.registry import ToolRegistry, callable_to_schema
from neuroagent.tools.web_fetch import build_web_fetch_tool

# A single-URL fetch primitive (web_fetch's shape): url -> result dict.
FetchFn = Callable[[str], Awaitable[dict[str, Any]]]
LinkReaderTool = Callable[[str], Awaitable[dict[str, Any]]]

_DEFAULT_MAX_URLS = 10
_SPLIT_RE = re.compile(r"[\s,]+")


def _check_max_urls(max_urls: int) -> int:
    """Reject a non-positive cap (it would disable the tool entirely)."""
    if max_urls < 1:
        raise ValueError(f"max_urls must be >= 1, got {max_urls}")
    return max_urls


def _parse_urls(urls: str, max_urls: int) -> tuple[list[str], bool]:
    """Split a free-form url list, dedupe (order-preserving), cap at *max_urls*.

    Returns the capped list plus a ``truncated`` flag so the caller can tell the
    model some links were dropped rather than silently losing them.
    """
    candidates = [u for u in _SPLIT_RE.split(urls.strip()) if u]
    seen: set[str] = set()
    ordered: list[str] = []
    for u in candidates:
        if u not in seen:
            seen.add(u)
            ordered.append(u)
    truncated = len(ordered) > max_urls
    return ordered[:max_urls], truncated


def _make_link_reader(fetch: FetchFn, max_urls: int) -> LinkReaderTool:
    """Build the bound link_reader callable over a single-URL *fetch*."""

    async def link_reader(urls: str) -> dict[str, Any]:
        """Read several web links and return each page's extracted text.

        urls is a whitespace/comma-separated list. Each link is fetched and
        extracted like web_fetch; a per-link failure is reported in that link's
        entry (an error field) without failing the batch.
        """
        parsed, truncated = _parse_urls(urls, max_urls)
        links: list[dict[str, Any]] = []
        for url in parsed:
            try:
                links.append(await fetch(url))
            except Exception as exc:  # per-link isolation: one bad link != batch fail
                links.append({"url": url, "error": str(exc)})
        return {"links": links, "count": len(parsed), "truncated": truncated}

    return link_reader


def build_link_reader_tool(
    registry: ToolRegistry,
    *,
    fetch: FetchFn | None = None,
    max_urls: int = _DEFAULT_MAX_URLS,
    timeout: int = 20,
    max_bytes: int = 2_000_000,
) -> LinkReaderTool:
    """Return a bound link_reader tool, registered under ``"link_reader"``.

    fetch is the single-URL primitive link_reader delegates to; it defaults to a
    fresh web_fetch (built into a scratch registry so *registry* receives only
    ``link_reader`` and the production web_fetch registration is untouched).
    Injecting fetch is for tests or a custom egress policy.
    """
    _check_max_urls(max_urls)
    if fetch is None:
        fetch = build_web_fetch_tool(ToolRegistry(), timeout=timeout, max_bytes=max_bytes)
    tool = _make_link_reader(fetch, max_urls)
    registry.register("link_reader", tool, callable_to_schema(tool))
    return tool


class LinkReaderToolProvider:
    """Yields the low-risk link_reader tool (fetch + extract over web_fetch).

    Selectable through the provider path: reference the provider in
    ``AgentSpec.providers`` and enable ``"link_reader"`` via ``AgentSpec.tools``
    or a ``tool_set``. The yielded spec carries descriptive metadata only
    (category ``"web"``, read-only side effect); it never gates execution.
    """

    def __init__(
        self,
        *,
        fetch: FetchFn | None = None,
        max_urls: int = _DEFAULT_MAX_URLS,
        timeout: int = 20,
        max_bytes: int = 2_000_000,
    ) -> None:
        self._fetch = fetch
        self._max_urls = _check_max_urls(max_urls)
        self._timeout = timeout
        self._max_bytes = max_bytes

    def provide(self) -> list[ToolSpec]:
        fetch = self._fetch
        if fetch is None:
            fetch = build_web_fetch_tool(
                ToolRegistry(), timeout=self._timeout, max_bytes=self._max_bytes
            )
        tool = _make_link_reader(fetch, self._max_urls)
        return [
            spec_from_callable(
                tool,
                category="web",
                capabilities=("link", "read", "extract"),
                side_effect="read",
            )
        ]
