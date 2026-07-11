"""Provider-routed ``web_search`` tool with an unchanged public call shape."""

from __future__ import annotations

import asyncio
import inspect
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from time import perf_counter
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from neuroagent.tools.registry import ToolRegistry, callable_to_schema
from neuroagent.tools.web_budget import WebRunBudget
from neuroagent.tools.web_models import WebStatus
from neuroagent.tools.web_search_models import (
    ExaSearchClient,
    SearchAttempt,
    SearchBackend,
    SearchBackendError,
    SearchBackendResult,
    SearchItem,
    SearchRequest,
    SearchResponse,
    SearchTrace,
)

_TRACKING_QUERY_KEYS = frozenset({"fbclid", "gclid", "mc_cid", "mc_eid"})


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Public result returned by ``web_search``; original fields stay required."""

    title: str
    url: str
    snippet: str
    published_date: str | None = None
    content: str = ""
    highlights: tuple[str, ...] = ()
    provider: str = ""
    request_id: str | None = None
    cost_usd: float | None = None


WebSearchTool = Callable[[str, int], Awaitable[list[SearchResult]]]
SearchObserver = Callable[[SearchAttempt], Awaitable[None] | None]
SearchRequestTransform = Callable[[SearchRequest], SearchRequest]


def normalize_search_url(url: str) -> str:
    """Canonicalize result URLs for stable deduplication and citations."""
    parsed = urlsplit(url.strip())
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return ""
    scheme = parsed.scheme.lower()
    host = parsed.hostname.lower()
    try:
        port = parsed.port
    except ValueError:
        return ""
    netloc = host
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{host}:{port}"
    path = parsed.path
    if path == "/":
        path = ""
    elif path:
        path = path.rstrip("/")
    query = urlencode(
        sorted(
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if not key.lower().startswith("utm_") and key.lower() not in _TRACKING_QUERY_KEYS
        )
    )
    return urlunsplit((scheme, netloc, path, query, ""))


def _host_matches(url: str, domains: tuple[str, ...]) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    return any(host == domain.lower() or host.endswith(f".{domain.lower()}") for domain in domains)


def _parse_published_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    except ValueError:
        return None


def _item_allowed(item: SearchItem, request: SearchRequest) -> bool:
    if request.include_domains and not _host_matches(item.url, request.include_domains):
        return False
    if request.exclude_domains and _host_matches(item.url, request.exclude_domains):
        return False
    if request.start_published_at is None and request.end_published_at is None:
        return True
    published = _parse_published_date(item.published_date)
    if published is None:
        return False
    if request.start_published_at is not None and published < request.start_published_at:
        return False
    return not (request.end_published_at is not None and published > request.end_published_at)


def _normalise_items(items: Sequence[SearchItem], request: SearchRequest) -> tuple[SearchItem, ...]:
    output: list[SearchItem] = []
    seen: set[str] = set()
    for item in items:
        url = normalize_search_url(item.url)
        if not url or url in seen:
            continue
        normalized = replace(item, url=url)
        if not _item_allowed(normalized, request):
            continue
        seen.add(url)
        output.append(normalized)
        if len(output) >= request.max_results:
            break
    return tuple(output)


class SearchRouter:
    """Try configured backends in order and return the first non-empty result set."""

    def __init__(
        self,
        backends: Sequence[SearchBackend],
        *,
        observer: SearchObserver | None = None,
        budget: WebRunBudget | None = None,
        max_results: int = 100,
        max_content_results: int = 3,
        request_transform: SearchRequestTransform | None = None,
    ) -> None:
        if not backends:
            raise ValueError("SearchRouter requires at least one backend")
        if max_results < 1:
            raise ValueError("max_results must be positive")
        if max_content_results < 0:
            raise ValueError("max_content_results must not be negative")
        self._backends = tuple(backends)
        self._observer = observer
        self._budget = budget
        self._max_results = max_results
        self._max_content_results = max_content_results
        self._request_transform = request_transform

    async def search(self, request: SearchRequest) -> SearchResponse:
        if self._request_transform is not None:
            request = self._request_transform(request)
        if self._budget is not None and not await self._budget.reserve("search"):
            attempt = SearchAttempt(
                provider="quota",
                status="rate_limited",
                latency_ms=0,
                error_code="web_search_run_budget_exhausted",
            )
            await self._record(attempt)
            return SearchResponse(
                items=(), attempts=(attempt,), final_provider=None
            )
        limited_request = replace(
            request,
            max_results=min(request.max_results, self._max_results),
            content_results=min(
                request.content_results,
                self._max_content_results,
                request.max_results,
            ),
        )
        if self._budget is None:
            return await self._search(limited_request)
        async with self._budget.slot():
            return await self._search(limited_request)

    async def _search(self, request: SearchRequest) -> SearchResponse:
        attempts: list[SearchAttempt] = []
        fallback_reason: str | None = None
        for backend in self._backends:
            started = perf_counter()
            if not backend.available:
                attempt = SearchAttempt(
                    provider=backend.name,
                    status="provider_unavailable",
                    latency_ms=0,
                    fallback_reason=fallback_reason,
                    error_code="startup_preflight_unavailable",
                )
                await self._record(attempt)
                attempts.append(attempt)
                fallback_reason = attempt.status
                continue
            try:
                backend_result = await backend.search(request)
            except SearchBackendError as exc:
                attempt = SearchAttempt(
                    provider=backend.name,
                    status=exc.error.status,
                    latency_ms=(perf_counter() - started) * 1000,
                    fallback_reason=fallback_reason,
                    error_code=exc.error.code,
                )
            except TimeoutError:
                attempt = SearchAttempt(
                    provider=backend.name,
                    status="timeout",
                    latency_ms=(perf_counter() - started) * 1000,
                    fallback_reason=fallback_reason,
                    error_code="timeout",
                )
            except Exception as exc:  # stable tool behavior; details stay out of observations
                attempt = SearchAttempt(
                    provider=backend.name,
                    status="provider_unavailable",
                    latency_ms=(perf_counter() - started) * 1000,
                    fallback_reason=fallback_reason,
                    error_code=type(exc).__name__,
                )
            else:
                items = _normalise_items(backend_result.items, request)
                status: WebStatus = "success" if items else "empty_content"
                attempt = SearchAttempt(
                    provider=backend.name,
                    status=status,
                    latency_ms=(perf_counter() - started) * 1000,
                    result_count=len(items),
                    fallback_reason=fallback_reason,
                    request_id=backend_result.request_id,
                    cost_usd=backend_result.cost_usd,
                )
                await self._record(attempt)
                attempts.append(attempt)
                if items:
                    return SearchResponse(
                        items=items,
                        attempts=tuple(attempts),
                        final_provider=backend.name,
                    )
                fallback_reason = status
                continue

            await self._record(attempt)
            attempts.append(attempt)
            fallback_reason = attempt.status

        return SearchResponse(items=(), attempts=tuple(attempts), final_provider=None)

    async def _record(self, attempt: SearchAttempt) -> None:
        if self._observer is None:
            return
        try:
            result = self._observer(attempt)
            if inspect.isawaitable(result):
                await result
        except Exception:
            return


class ExaSearchBackend:
    """Exa Search mapping over an app-injected transport client."""

    name = "exa"

    def __init__(self, client: ExaSearchClient | None) -> None:
        self._client = client

    @property
    def available(self) -> bool:
        return self._client is not None

    async def search(self, request: SearchRequest) -> SearchBackendResult:
        if self._client is None:
            return SearchBackendResult(items=())
        payload: dict[str, Any] = {
            "query": request.query,
            "numResults": request.max_results,
            "type": "auto",
        }
        if request.content_results:
            # Search content is discovery context; selected sources are fetched
            # in full through web_fetch before they can become evidence. Bound
            # each result so a 10-result search cannot transfer multi-megabyte
            # page bodies only to discard seven of them below.
            payload["contents"] = {
                "text": {"maxCharacters": 4_000},
                "highlights": True,
            }
        if request.include_domains:
            payload["includeDomains"] = list(request.include_domains)
        if request.exclude_domains:
            payload["excludeDomains"] = list(request.exclude_domains)
        if request.start_published_at is not None:
            payload["startPublishedDate"] = request.start_published_at.isoformat()
        if request.end_published_at is not None:
            payload["endPublishedDate"] = request.end_published_at.isoformat()

        raw = await self._client.search(payload)
        request_id = _optional_string(raw.get("requestId"))
        cost_usd = _exa_cost(raw)
        raw_items = raw.get("results")
        if not isinstance(raw_items, list):
            raw_items = []
        items: list[SearchItem] = []
        for raw_item in raw_items:
            if not isinstance(raw_item, Mapping):
                continue
            keep_content = len(items) < request.content_results
            highlights_raw = raw_item.get("highlights")
            highlights = (
                tuple(str(value) for value in highlights_raw if isinstance(value, str))
                if keep_content and isinstance(highlights_raw, list)
                else ()
            )
            content = (
                (_optional_string(raw_item.get("text")) or "")
                if keep_content
                else ""
            )
            snippet = highlights[0] if highlights else content[:500]
            items.append(
                SearchItem(
                    title=_optional_string(raw_item.get("title")) or "",
                    url=_optional_string(raw_item.get("url")) or "",
                    snippet=snippet,
                    published_date=_optional_string(raw_item.get("publishedDate")),
                    content=content,
                    highlights=highlights,
                    provider=self.name,
                    request_id=request_id,
                    cost_usd=cost_usd,
                )
            )
        return SearchBackendResult(
            items=tuple(items),
            request_id=request_id,
            cost_usd=cost_usd,
        )


class LegacySearchBackend:
    """Compatibility backend for Brave (when configured) or DuckDuckGo."""

    name = "legacy"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key

    @property
    def available(self) -> bool:
        return True

    async def search(self, request: SearchRequest) -> SearchBackendResult:
        if self._api_key:
            results = await _brave_search(request.query, request.max_results, self._api_key)
            provider = "brave"
        else:
            results = await _ddg_search(request.query, request.max_results)
            provider = "duckduckgo"
        return SearchBackendResult(
            items=tuple(
                SearchItem(
                    title=result.title,
                    url=result.url,
                    snippet=result.snippet,
                    provider=provider,
                )
                for result in results
            )
        )


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _exa_cost(data: Mapping[str, Any]) -> float | None:
    raw = data.get("costDollars")
    if not isinstance(raw, Mapping):
        return None
    total = raw.get("total")
    if isinstance(total, int | float) and total >= 0:
        return float(total)
    return None


def _to_search_results(ddg_items: list[dict[str, Any]]) -> list[SearchResult]:
    return [
        SearchResult(
            title=str(item.get("title") or ""),
            url=str(item.get("href") or ""),
            snippet=str(item.get("body") or ""),
        )
        for item in ddg_items
    ]


def build_web_search_tool(
    registry: ToolRegistry,
    api_key_env: str = "NEUROAGENT_WEB_SEARCH_API_KEY",
    *,
    router: SearchRouter | None = None,
) -> WebSearchTool:
    """Register the compatible tool, optionally backed by an injected Router."""
    selected_router = router or SearchRouter(
        [LegacySearchBackend(os.environ.get(api_key_env))]
    )

    async def web_search(query: str, max_results: int = 5) -> list[SearchResult]:
        response = await selected_router.search(
            SearchRequest(
                query=query,
                max_results=min(max(max_results, 1), 100),
                content_results=min(3, max(max_results, 1)),
            )
        )
        return [
            SearchResult(
                title=item.title,
                url=item.url,
                snippet=item.snippet,
                published_date=item.published_date,
                content=item.content,
                highlights=item.highlights,
                provider=item.provider,
                request_id=item.request_id,
                cost_usd=item.cost_usd,
            )
            for item in response.items
        ]

    registry.register("web_search", web_search, callable_to_schema(web_search))
    return web_search


async def _ddg_search(query: str, max_results: int) -> list[SearchResult]:
    from ddgs import DDGS

    def _sync_search() -> list[dict[str, Any]]:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=min(max_results, 10)))

    items = await asyncio.to_thread(_sync_search)
    return _to_search_results(items)


async def _brave_search(query: str, max_results: int, api_key: str) -> list[SearchResult]:
    import httpx

    headers = {"Accept": "application/json", "X-Subscription-Token": api_key}
    params: dict[str, str | int] = {
        "q": query,
        "count": min(max_results, 10),
        "offset": 0,
    }
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers=headers,
            params=params,
        )
        response.raise_for_status()
        data = response.json()

    web_results = data.get("web", {}).get("results", [])
    return [
        SearchResult(
            title=result.get("title", ""),
            url=result.get("url", ""),
            snippet=result.get("description", ""),
        )
        for result in web_results[: min(max_results, 10)]
    ]


__all__ = [
    "ExaSearchBackend",
    "LegacySearchBackend",
    "SearchResult",
    "SearchRouter",
    "SearchTrace",
    "build_web_search_tool",
    "normalize_search_url",
]
