"""link_reader (PR5): low-risk fetch+extract aggregation over web_fetch.

Pins three things:

1. Behavior - parse/dedupe/cap a batch of URLs, delegate each to a single-URL
   fetch primitive, isolate per-link failures, and report truncation.
2. Provider path - LinkReaderToolProvider yields a clean (non-degenerate) spec
   with descriptive web/read metadata, registers + executes, and is selectable
   through load_agent_from_spec (providers + tools / tool_sets).
3. web_fetch is untouched - the default fetch is built into a scratch registry,
   so the caller's registry receives only link_reader, and the SSRF guard is
   inherited (a private URL fails as a per-link error, no network).
"""

from __future__ import annotations

from typing import Any

import pytest
from sanyuan_agent.config import AgentSpec, load_agent_from_spec
from sanyuan_agent.config.loader import _resolve_tools
from sanyuan_agent.models import LLMProvider, LLMProviderRegistry
from sanyuan_agent.tools import (
    LinkReaderToolProvider,
    build_link_reader_tool,
    build_web_fetch_tool,
)
from sanyuan_agent.tools.provider import ToolProvider, register_provider
from sanyuan_agent.tools.registry import ToolRegistry
from sanyuan_agent.tools.web_reader_models import FetchDocument


class _FakeFetch:
    """Records calls; returns a web_fetch-shaped dict (no network)."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def __call__(self, url: str) -> dict[str, Any]:
        self.calls.append(url)
        return {
            "url": url,
            "title": f"title:{url}",
            "content": f"content:{url}",
            "status_code": 200,
            "blocked": False,
            "block_reason": None,
        }


# ---------------------------------------------------------------------------
# Behavior
# ---------------------------------------------------------------------------


class TestBehavior:
    async def test_reads_multiple_links_and_aggregates(self) -> None:
        fetch = _FakeFetch()
        tool = build_link_reader_tool(ToolRegistry(), fetch=fetch)

        result = await tool("https://a.test https://b.test")

        assert fetch.calls == ["https://a.test", "https://b.test"]
        assert result["count"] == 2
        assert result["truncated"] is False
        assert [link["url"] for link in result["links"]] == [
            "https://a.test",
            "https://b.test",
        ]
        assert result["links"][0]["content"] == "content:https://a.test"

    async def test_comma_and_newline_separated_and_deduped(self) -> None:
        fetch = _FakeFetch()
        tool = build_link_reader_tool(ToolRegistry(), fetch=fetch)

        # commas, whitespace, newlines, and a duplicate
        result = await tool("https://a.test, https://b.test\nhttps://a.test")

        assert fetch.calls == ["https://a.test", "https://b.test"]  # deduped, ordered
        assert result["count"] == 2

    async def test_empty_input_reads_nothing(self) -> None:
        fetch = _FakeFetch()
        tool = build_link_reader_tool(ToolRegistry(), fetch=fetch)

        result = await tool("   \n  ")

        assert fetch.calls == []
        assert result == {"links": [], "count": 0, "truncated": False}

    async def test_caps_at_max_urls_and_flags_truncated(self) -> None:
        fetch = _FakeFetch()
        tool = build_link_reader_tool(ToolRegistry(), fetch=fetch, max_urls=2)

        result = await tool("https://a.test https://b.test https://c.test")

        assert fetch.calls == ["https://a.test", "https://b.test"]  # third dropped
        assert result["count"] == 2
        assert result["truncated"] is True

    async def test_per_link_failure_is_isolated(self) -> None:
        class _Flaky:
            async def __call__(self, url: str) -> dict[str, Any]:
                if "bad" in url:
                    raise ValueError("boom")
                return {"url": url, "content": "ok"}

        tool = build_link_reader_tool(ToolRegistry(), fetch=_Flaky())
        result = await tool("https://good.test https://bad.test")

        assert result["links"][0] == {"url": "https://good.test", "content": "ok"}
        assert result["links"][1] == {"url": "https://bad.test", "error": "boom"}
        assert result["count"] == 2


# ---------------------------------------------------------------------------
# max_urls validation (fail fast on a non-positive cap)
# ---------------------------------------------------------------------------


class TestMaxUrlsValidation:
    @pytest.mark.parametrize("bad", [0, -1, -10])
    def test_builder_rejects_non_positive(self, bad: int) -> None:
        with pytest.raises(ValueError, match="max_urls must be >= 1"):
            build_link_reader_tool(ToolRegistry(), fetch=_FakeFetch(), max_urls=bad)

    @pytest.mark.parametrize("bad", [0, -1, -10])
    def test_provider_rejects_non_positive_at_construction(self, bad: int) -> None:
        with pytest.raises(ValueError, match="max_urls must be >= 1"):
            LinkReaderToolProvider(fetch=_FakeFetch(), max_urls=bad)

    async def test_min_valid_cap_of_one_works(self) -> None:
        fetch = _FakeFetch()
        tool = build_link_reader_tool(ToolRegistry(), fetch=fetch, max_urls=1)
        result = await tool("https://a.test https://b.test")
        assert fetch.calls == ["https://a.test"]  # second dropped at the cap
        assert result["count"] == 1
        assert result["truncated"] is True


# ---------------------------------------------------------------------------
# Provider path
# ---------------------------------------------------------------------------


class TestProvider:
    def test_provider_is_runtime_checkable(self) -> None:
        assert isinstance(LinkReaderToolProvider(), ToolProvider)

    def test_provide_yields_one_clean_spec_with_metadata(self) -> None:
        (spec,) = LinkReaderToolProvider(fetch=_FakeFetch()).provide()
        assert spec.name == "link_reader"
        assert spec.category == "web"
        assert spec.capabilities == ("link", "read", "extract")
        assert spec.side_effect == "read"
        assert spec.artifact_output is False
        assert spec.policy_hint is None
        assert spec.required_secrets == ()  # no secrets
        # non-degenerate schema reflected from the real signature
        props = spec.schema["parameters"]["properties"]
        assert set(props) == {"urls"}
        assert "arguments" not in props
        assert spec.schema["parameters"]["required"] == ["urls"]

    async def test_register_provider_then_execute(self) -> None:
        fetch = _FakeFetch()
        registry = ToolRegistry()
        names = register_provider(registry, LinkReaderToolProvider(fetch=fetch))

        assert names == ["link_reader"]
        result = await registry.execute("link_reader", '{"urls": "https://a.test"}')
        assert result["count"] == 1
        assert result["links"][0]["url"] == "https://a.test"
        assert fetch.calls == ["https://a.test"]

    def test_builder_and_provider_schemas_are_equivalent(self) -> None:
        fetch = _FakeFetch()

        legacy = ToolRegistry()
        build_link_reader_tool(legacy, fetch=fetch)

        provider_reg = ToolRegistry()
        register_provider(provider_reg, LinkReaderToolProvider(fetch=fetch))

        assert legacy.get_schema("link_reader") == provider_reg.get_schema("link_reader")
        assert legacy.tool_defs() == provider_reg.tool_defs()


# ---------------------------------------------------------------------------
# Selectable through the loader (provider path)
# ---------------------------------------------------------------------------


def _llm_registry() -> LLMProviderRegistry:
    reg = LLMProviderRegistry()
    reg.register(
        LLMProvider(
            name="higress",
            provider_kind="openai_compatible",
            api_base="http://higress.test/v1",
            api_key_env="SANYUAN_AGENT_LLM_API_KEY",
            default_model="openai/deepseek-v4-flash",
        )
    )
    return reg


def _spec(**over: Any) -> AgentSpec:
    base = {
        "name": "reader",
        "system_prompt": "You read links.",
        "llm_provider": "higress",
    }
    base.update(over)
    return AgentSpec(**base)


class TestLoaderSelection:
    def test_candidate_not_auto_enabled(self) -> None:
        providers = {"links": LinkReaderToolProvider(fetch=_FakeFetch())}
        spec = _spec(providers=["links"])  # referenced but nothing selected
        assert _resolve_tools(spec, ToolRegistry(), None, providers) == []

    def test_selected_via_tools(self) -> None:
        providers = {"links": LinkReaderToolProvider(fetch=_FakeFetch())}
        spec = _spec(providers=["links"], tools=["link_reader"])
        resolved = _resolve_tools(spec, ToolRegistry(), None, providers)
        assert [t.__name__ for t in resolved] == ["link_reader"]

    def test_selected_via_tool_set(self) -> None:
        providers = {"links": LinkReaderToolProvider(fetch=_FakeFetch())}
        sets = {"reading": ["link_reader"]}
        spec = _spec(providers=["links"], tool_sets=["reading"])
        resolved = _resolve_tools(spec, ToolRegistry(), sets, providers)
        assert [t.__name__ for t in resolved] == ["link_reader"]

    def test_load_agent_builds_graph(self) -> None:
        providers = {"links": LinkReaderToolProvider(fetch=_FakeFetch())}
        spec = _spec(providers=["links"], tools=["link_reader"])
        loaded = load_agent_from_spec(
            spec,
            tool_registry=ToolRegistry(),
            llm_registry=_llm_registry(),
            tool_providers=providers,
        )
        assert loaded.graph is not None


# ---------------------------------------------------------------------------
# web_fetch behavior preserved
# ---------------------------------------------------------------------------


class TestWebFetchPreserved:
    def test_default_builder_does_not_register_web_fetch_in_caller_registry(self) -> None:
        registry = ToolRegistry()
        build_link_reader_tool(registry)  # default fetch -> scratch web_fetch
        assert registry.list_tools() == ["link_reader"]
        assert not registry.has_tool("web_fetch")

    def test_web_fetch_schema_unchanged(self) -> None:
        registry = ToolRegistry()
        build_web_fetch_tool(registry)
        schema = registry.get_schema("web_fetch")
        assert schema["name"] == "web_fetch"
        assert set(schema["parameters"]["properties"]) == {"url"}
        assert schema["parameters"]["required"] == ["url"]

    async def test_reader_document_transform_applies_before_serialization(self) -> None:
        class FakeReaderRouter:
            async def fetch(self, request):
                return FetchDocument(
                    url=request.url,
                    final_url=request.url,
                    provider="fake",
                    status="success",
                    content="original bulk",
                )

        def focus(document: FetchDocument) -> FetchDocument:
            return FetchDocument(
                url=document.url,
                final_url=document.final_url,
                provider=document.provider,
                status=document.status,
                content="focused evidence",
            )

        tool = build_web_fetch_tool(
            ToolRegistry(),
            reader_router=FakeReaderRouter(),
            document_transform=focus,
        )

        result = await tool("https://example.com/page")

        assert "focused evidence" in result["content"]
        assert "original bulk" not in result["content"]

    async def test_ssrf_guard_inherited_as_per_link_error(self) -> None:
        # Default fetch is the real web_fetch, whose SSRF guard rejects localhost
        # before any network call; link_reader surfaces it as a per-link error.
        tool = build_link_reader_tool(ToolRegistry())
        result = await tool("http://127.0.0.1/secret")
        assert result["count"] == 1
        assert "SSRF" in result["links"][0]["error"]
