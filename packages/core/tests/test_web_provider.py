"""WebToolProvider (PR9): thin wrapper over the web_search + web_fetch builders.

Pins that the provider:

1. yields the two existing web tools as specs with category="web" /
   side_effect="read" metadata and no declared secrets;
2. changes NO schema - register_provider produces exactly the legacy
   build_web_search_tool + build_web_fetch_tool tool_defs;
3. changes NO behavior - the wrapped web_fetch keeps its SSRF guard (a private
   URL still raises, no network); and
4. is selectable through load_agent_from_spec (providers + tools / tool_sets).
"""

from __future__ import annotations

from typing import Any

import pytest
from neuroagent.config import AgentSpec, load_agent_from_spec
from neuroagent.config.loader import _resolve_tools
from neuroagent.models import LLMProvider, LLMProviderRegistry
from neuroagent.tools import (
    WebToolProvider,
    build_web_fetch_tool,
    build_web_search_tool,
)
from neuroagent.tools.provider import ToolProvider, register_provider
from neuroagent.tools.registry import ToolRegistry
from neuroagent.tools.web_search import SearchRouter
from neuroagent.tools.web_search_models import SearchBackendResult, SearchItem, SearchRequest

EXPECTED = {"web_search", "web_fetch"}


def _specs_by_name() -> dict[str, Any]:
    return {s.name: s for s in WebToolProvider().provide()}


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_provider_is_runtime_checkable(self) -> None:
        assert isinstance(WebToolProvider(), ToolProvider)

    def test_yields_search_and_fetch(self) -> None:
        assert set(_specs_by_name()) == EXPECTED

    def test_both_web_read_no_secrets(self) -> None:
        for spec in _specs_by_name().values():
            assert spec.category == "web"
            assert spec.side_effect == "read"
            assert spec.policy_hint is None
            assert spec.artifact_output is False
            assert spec.required_secrets == ()  # no new secret behavior

    def test_capabilities(self) -> None:
        specs = _specs_by_name()
        assert specs["web_search"].capabilities == ("web", "search")
        assert specs["web_fetch"].capabilities == ("web", "fetch")

    def test_routed_provider_declares_secret_names_without_values(self) -> None:
        specs = {
            spec.name: spec
            for spec in WebToolProvider(
                search_required_secrets=("EXA_API_KEY",),
                fetch_required_secrets=("EXA_API_KEY", "FIRECRAWL_API_KEY"),
            ).provide()
        }
        assert specs["web_search"].required_secrets == ("EXA_API_KEY",)
        assert specs["web_fetch"].required_secrets == (
            "EXA_API_KEY",
            "FIRECRAWL_API_KEY",
        )


# ---------------------------------------------------------------------------
# No schema change / no behavior change
# ---------------------------------------------------------------------------


class TestEquivalence:
    def test_schemas_identical_to_legacy_builders(self) -> None:
        legacy = ToolRegistry()
        build_web_search_tool(legacy)
        build_web_fetch_tool(legacy)

        provider_reg = ToolRegistry()
        names = register_provider(provider_reg, WebToolProvider())

        assert set(names) == EXPECTED
        assert provider_reg.list_tools() == legacy.list_tools()
        for name in legacy.list_tools():
            assert provider_reg.get_schema(name) == legacy.get_schema(name)
        # strongest proof: the OpenAI tool_defs are byte-identical
        assert provider_reg.tool_defs() == legacy.tool_defs()

    async def test_web_fetch_behavior_preserved_ssrf_guard(self) -> None:
        # The wrapped web_fetch keeps its SSRF guard: a localhost URL is rejected
        # before any network call - behavior unchanged, no network in the test.
        fetch = _specs_by_name()["web_fetch"]
        with pytest.raises(ValueError, match="SSRF"):
            await fetch.callable("http://127.0.0.1/secret")

    async def test_injected_search_router_is_used(self) -> None:
        class StaticBackend:
            name = "static"
            available = True

            async def search(self, request: SearchRequest) -> SearchBackendResult:
                return SearchBackendResult(
                    items=(SearchItem(title=request.query, url="https://example.com"),)
                )

        specs = {
            spec.name: spec
            for spec in WebToolProvider(SearchRouter([StaticBackend()])).provide()
        }
        results = await specs["web_search"].callable("routed", 5)
        assert results[0].title == "routed"


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
            api_key_env="NEUROAGENT_LLM_API_KEY",
            default_model="openai/deepseek-v4-flash",
        )
    )
    return reg


def _spec(**over: Any) -> AgentSpec:
    base = {
        "name": "web-agent",
        "system_prompt": "You search and read the web.",
        "llm_provider": "higress",
    }
    base.update(over)
    return AgentSpec(**base)


class TestLoaderSelection:
    def test_candidate_not_auto_enabled(self) -> None:
        providers = {"web": WebToolProvider()}
        spec = _spec(providers=["web"])  # referenced, nothing selected
        assert _resolve_tools(spec, ToolRegistry(), None, providers) == []

    def test_select_subset_via_tools(self) -> None:
        providers = {"web": WebToolProvider()}
        spec = _spec(providers=["web"], tools=["web_search"])
        resolved = _resolve_tools(spec, ToolRegistry(), None, providers)
        assert [t.__name__ for t in resolved] == ["web_search"]

    def test_select_via_tool_set(self) -> None:
        providers = {"web": WebToolProvider()}
        sets = {"web_bundle": ["web_search", "web_fetch"]}
        spec = _spec(providers=["web"], tool_sets=["web_bundle"])
        resolved = _resolve_tools(spec, ToolRegistry(), sets, providers)
        assert [t.__name__ for t in resolved] == ["web_search", "web_fetch"]

    def test_load_agent_builds_graph(self) -> None:
        providers = {"web": WebToolProvider()}
        spec = _spec(providers=["web"], tools=["web_search", "web_fetch"])
        loaded = load_agent_from_spec(
            spec,
            tool_registry=ToolRegistry(),
            llm_registry=_llm_registry(),
            tool_providers=providers,
        )
        assert loaded.graph is not None
