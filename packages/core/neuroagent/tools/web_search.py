"""Web-search tool builder — DuckDuckGo via duckduckgo_search."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from neuroagent.tools.registry import ToolRegistry, callable_to_schema


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A single web-search result."""

    title: str
    url: str
    snippet: str


WebSearchTool = Callable[[str, int], Awaitable[list[SearchResult]]]


def _to_search_results(ddg_items: list[dict[str, Any]]) -> list[SearchResult]:
    out: list[SearchResult] = []
    for item in ddg_items:
        out.append(
            SearchResult(
                title=item.get("title", ""),
                url=item.get("href", ""),
                snippet=item.get("body", ""),
            )
        )
    return out


def build_web_search_tool(
    registry: ToolRegistry,
    api_key_env: str = "NEUROAGENT_WEB_SEARCH_API_KEY",
) -> WebSearchTool:
    """Return a bound web-search tool using DuckDuckGo.

    If *api_key_env* points to a non-empty environment variable, Brave Search
    API is used instead of DuckDuckGo (future enhancement).

    The tool is automatically registered in *registry* under ``"web_search"``.
    """
    _api_key = os.environ.get(api_key_env)

    async def web_search(query: str, max_results: int = 5) -> list[SearchResult]:
        if _api_key:
            return await _brave_search(query, max_results, _api_key)
        return await _ddg_search(query, max_results)

    registry.register("web_search", web_search, callable_to_schema(web_search))
    return web_search


async def _ddg_search(query: str, max_results: int) -> list[SearchResult]:
    from duckduckgo_search import DDGS

    def _sync_search() -> list[dict[str, Any]]:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=min(max_results, 10)))

    items = await asyncio.to_thread(_sync_search)
    return _to_search_results(items)


async def _brave_search(query: str, max_results: int, api_key: str) -> list[SearchResult]:
    import httpx

    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": api_key,
    }
    params: dict[str, str | int] = {
        "q": query,
        "count": min(max_results, 10),
        "offset": 0,
    }
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()

    web_results = data.get("web", {}).get("results", [])
    return [
        SearchResult(
            title=r.get("title", ""),
            url=r.get("url", ""),
            snippet=r.get("description", ""),
        )
        for r in web_results[: min(max_results, 10)]
    ]
