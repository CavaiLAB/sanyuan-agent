"""SearchRouter, Exa mapping, fallback, filtering, and compatibility tests."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from neuroagent.tools.registry import ToolRegistry
from neuroagent.tools.web_budget import WebRunBudget
from neuroagent.tools.web_search import (
    ExaSearchBackend,
    SearchResult,
    SearchRouter,
    build_web_search_tool,
    normalize_search_url,
)
from neuroagent.tools.web_search_models import (
    SearchBackendError,
    SearchBackendResult,
    SearchItem,
    SearchProviderError,
    SearchRequest,
    SearchTrace,
    WebStatus,
)


class FakeBackend:
    def __init__(
        self,
        name: str,
        *,
        items: tuple[SearchItem, ...] = (),
        status: WebStatus | None = None,
        available: bool = True,
    ) -> None:
        self.name = name
        self.available = available
        self._items = items
        self._status = status
        self.calls: list[SearchRequest] = []

    async def search(self, request: SearchRequest) -> SearchBackendResult:
        self.calls.append(request)
        if self._status is not None:
            raise SearchBackendError(
                SearchProviderError(
                    provider=self.name,
                    status=self._status,
                    code=f"fake_{self._status}",
                    retryable=self._status in {"timeout", "rate_limited"},
                )
            )
        return SearchBackendResult(items=self._items, request_id=f"req-{self.name}")


class FakeExaClient:
    def __init__(self, response: Mapping[str, Any]) -> None:
        self.response = response
        self.payloads: list[Mapping[str, Any]] = []

    async def search(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        self.payloads.append(payload)
        return self.response


async def test_duckduckgo_legacy_uses_ddgs_successfully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from neuroagent.tools import web_search as module

    class FakeDDGS:
        def __enter__(self) -> FakeDDGS:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def text(self, query: str, *, max_results: int) -> list[dict[str, str]]:
            assert query == "fallback query"
            assert max_results == 2
            return [
                {
                    "title": "Recovered",
                    "href": "https://example.com/recovered",
                    "body": "legacy evidence",
                }
            ]

    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS=FakeDDGS))

    results = await module._ddg_search("fallback query", 2)

    assert results == [
        SearchResult(
            title="Recovered",
            url="https://example.com/recovered",
            snippet="legacy evidence",
        )
    ]


@pytest.mark.parametrize("status", ["timeout", "rate_limited", "provider_unavailable"])
async def test_structured_failure_falls_back(status: WebStatus) -> None:
    trace = SearchTrace()
    primary = FakeBackend("primary", status=status)
    fallback = FakeBackend(
        "legacy",
        items=(SearchItem(title="ok", url="https://example.com", provider="legacy"),),
    )

    response = await SearchRouter([primary, fallback], observer=trace).search(
        SearchRequest(query="q")
    )

    assert response.final_provider == "legacy"
    assert response.items[0].url == "https://example.com"
    assert [attempt.status for attempt in trace.attempts] == [status, "success"]
    assert trace.attempts[1].fallback_reason == status


async def test_unavailable_preflight_and_empty_results_both_fall_back() -> None:
    trace = SearchTrace()
    unavailable = FakeBackend("exa", available=False)
    empty = FakeBackend("empty")
    legacy = FakeBackend(
        "legacy",
        items=(SearchItem(title="ok", url="https://example.com"),),
    )

    response = await SearchRouter([unavailable, empty, legacy], observer=trace).search(
        SearchRequest(query="q")
    )

    assert response.final_provider == "legacy"
    assert [attempt.status for attempt in response.attempts] == [
        "provider_unavailable",
        "empty_content",
        "success",
    ]
    assert unavailable.calls == []


async def test_shared_run_budget_caps_search_calls_and_request_size() -> None:
    backend = FakeBackend(
        "exa",
        items=tuple(
            SearchItem(title=str(index), url=f"https://example.com/{index}") for index in range(10)
        ),
    )
    router = SearchRouter(
        [backend],
        budget=WebRunBudget(max_searches=1),
        max_results=4,
        max_content_results=2,
    )

    first = await router.search(SearchRequest(query="q", max_results=9, content_results=8))
    second = await router.search(SearchRequest(query="q2"))

    assert len(first.items) == 4
    assert backend.calls[0].max_results == 4
    assert backend.calls[0].content_results == 2
    assert second.items == ()
    assert second.attempts[0].provider == "quota"
    assert second.attempts[0].error_code == "web_search_run_budget_exhausted"
    assert len(backend.calls) == 1


async def test_search_router_awaits_async_observer() -> None:
    attempts: list[str] = []

    async def observe(attempt: Any) -> None:
        attempts.append(attempt.provider)

    router = SearchRouter([FakeBackend("empty")], observer=observe)
    await router.search(SearchRequest(query="q"))

    assert attempts == ["empty"]


async def test_search_router_applies_request_transform_before_backend() -> None:
    backend = FakeBackend(
        "exa",
        items=(SearchItem(title="official", url="https://www.fifa.com/result"),),
    )

    def transform(request: SearchRequest) -> SearchRequest:
        return SearchRequest(
            query=f"official {request.query}",
            max_results=request.max_results,
            include_domains=("fifa.com",),
            content_results=request.content_results,
        )

    response = await SearchRouter([backend], request_transform=transform).search(
        SearchRequest(query="bracket")
    )

    assert response.items[0].title == "official"
    assert backend.calls[0].query == "official bracket"
    assert backend.calls[0].include_domains == ("fifa.com",)


async def test_normalization_dedupe_domain_and_date_filters() -> None:
    backend = FakeBackend(
        "exa",
        items=(
            SearchItem(
                title="first",
                url="HTTPS://Docs.Example.com/a/?utm_source=x&b=2&a=1#part",
                published_date="2026-06-01T00:00:00Z",
            ),
            SearchItem(
                title="duplicate",
                url="https://docs.example.com/a?a=1&b=2",
                published_date="2026-06-01T00:00:00Z",
            ),
            SearchItem(
                title="excluded",
                url="https://blocked.example.com/a",
                published_date="2026-06-01T00:00:00Z",
            ),
            SearchItem(
                title="old",
                url="https://docs.example.com/old",
                published_date="2025-01-01T00:00:00Z",
            ),
        ),
    )
    request = SearchRequest(
        query="q",
        include_domains=("example.com",),
        exclude_domains=("blocked.example.com",),
        start_published_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    response = await SearchRouter([backend]).search(request)

    assert [item.title for item in response.items] == ["first"]
    assert response.items[0].url == "https://docs.example.com/a?a=1&b=2"
    assert normalize_search_url("ftp://example.com/a") == ""
    assert normalize_search_url("https://example.com:not-a-port/a") == ""


def test_request_allows_small_result_sets_and_requires_timezone() -> None:
    assert SearchRequest(query="q", max_results=1).max_results == 1
    with pytest.raises(ValueError, match="timezone"):
        SearchRequest(query="q", start_published_at=datetime(2026, 1, 1))


async def test_exa_maps_contents_request_id_and_cost() -> None:
    client = FakeExaClient(
        {
            "requestId": "req-123",
            "costDollars": {"total": 0.007},
            "results": [
                {
                    "title": "Result",
                    "url": "https://example.com/article",
                    "publishedDate": "2026-07-01T00:00:00Z",
                    "text": "full retrieved text",
                    "highlights": ["important excerpt"],
                }
            ],
        }
    )
    backend = ExaSearchBackend(client)
    request = SearchRequest(
        query="latest",
        max_results=7,
        include_domains=("example.com",),
        start_published_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    result = await backend.search(request)

    assert result.request_id == "req-123"
    assert result.cost_usd == 0.007
    assert result.items[0].content == "full retrieved text"
    assert result.items[0].highlights == ("important excerpt",)
    assert client.payloads[0]["numResults"] == 7
    assert client.payloads[0]["includeDomains"] == ["example.com"]
    assert client.payloads[0]["contents"] == {
        "text": {"maxCharacters": 4_000},
        "highlights": True,
    }


async def test_exa_keeps_content_only_for_configured_result_budget() -> None:
    client = FakeExaClient(
        {
            "results": [
                {
                    "title": str(index),
                    "url": f"https://example.com/{index}",
                    "text": f"content-{index}",
                    "highlights": [f"highlight-{index}"],
                }
                for index in range(4)
            ]
        }
    )

    result = await ExaSearchBackend(client).search(
        SearchRequest(query="q", max_results=4, content_results=2)
    )

    assert [item.content for item in result.items] == [
        "content-0",
        "content-1",
        "",
        "",
    ]
    assert result.items[2].highlights == ()


async def test_tool_request_transform_applies_before_selected_router() -> None:
    backend = FakeBackend(
        "legacy",
        items=(SearchItem(title="official", url="https://www.fifa.com/bracket"),),
    )
    registry = ToolRegistry()

    def transform(request: SearchRequest) -> SearchRequest:
        return SearchRequest(
            query=f"official {request.query}",
            max_results=request.max_results,
            include_domains=("fifa.com",),
            content_results=request.content_results,
        )

    tool = build_web_search_tool(
        registry,
        router=SearchRouter([backend]),
        request_transform=transform,
    )
    results = await tool("World Cup bracket", 3)

    assert results[0].url == "https://www.fifa.com/bracket"
    assert backend.calls[0].query == "official World Cup bracket"
    assert backend.calls[0].include_domains == ("fifa.com",)


async def test_public_tool_signature_unchanged_and_metadata_additive() -> None:
    router = SearchRouter(
        [
            FakeBackend(
                "exa",
                items=(
                    SearchItem(
                        title="Result",
                        url="https://example.com",
                        snippet="snippet",
                        content="evidence",
                        provider="exa",
                        request_id="req-1",
                    ),
                ),
            )
        ]
    )
    registry = ToolRegistry()
    tool = build_web_search_tool(registry, router=router)

    results = await tool("query", 5)

    assert results == [
        SearchResult(
            title="Result",
            url="https://example.com",
            snippet="snippet",
            content="evidence",
            provider="exa",
            request_id="req-1",
        )
    ]
    parameters = registry.get_schema("web_search")["parameters"]
    assert set(parameters["properties"]) == {"query", "max_results"}
