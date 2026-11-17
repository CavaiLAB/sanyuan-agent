"""BrowserToolProvider (PR7): thin provider wrapper over build_browser_tools.

Pins that the provider:

1. yields the existing 8 browser tools as specs with category="browser" and the
   right per-tool side-effect / policy-hint metadata (reads stay read; click /
   type / download default to confirm);
2. changes NO schema - register_provider produces exactly the legacy
   build_browser_tools tool_defs;
3. changes NO behavior - a wrapped tool executes the real browser action
   (including _annotate_block); and
4. is selectable through load_agent_from_spec (providers + tools / tool_sets).
"""

from __future__ import annotations

from typing import Any

from sanyuan_agent.config import AgentSpec, load_agent_from_spec
from sanyuan_agent.config.loader import _resolve_tools
from sanyuan_agent.models import LLMProvider, LLMProviderRegistry
from sanyuan_agent.tools import BrowserToolProvider, build_browser_tools
from sanyuan_agent.tools.provider import ToolProvider, register_provider
from sanyuan_agent.tools.registry import ToolRegistry

EXPECTED = {
    "browser_navigate",
    "browser_click",
    "browser_type",
    "browser_screenshot",
    "browser_read_dom",
    "browser_accessibility",
    "browser_download",
    "browser_downloads",
}
_READ_ONLY = {
    "browser_screenshot",
    "browser_read_dom",
    "browser_accessibility",
    "browser_downloads",
}
_CONFIRM = {"browser_click", "browser_type", "browser_download"}


class _FakeSession:
    """Minimal stand-in: only navigate() is exercised in the behavior test."""

    async def navigate(self, url: str) -> dict[str, Any]:
        return {"success": True, "url": url, "title": "T", "text": "hello world"}


async def _factory(thread_id: str) -> Any:
    return _FakeSession()


def _specs_by_name(provider: BrowserToolProvider) -> dict[str, Any]:
    return {s.name: s for s in provider.provide()}


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_provider_is_runtime_checkable(self) -> None:
        assert isinstance(BrowserToolProvider(_factory), ToolProvider)

    def test_yields_the_eight_browser_tools(self) -> None:
        specs = _specs_by_name(BrowserToolProvider(_factory))
        assert set(specs) == EXPECTED

    def test_all_category_browser_no_secrets(self) -> None:
        for spec in _specs_by_name(BrowserToolProvider(_factory)).values():
            assert spec.category == "browser"
            assert spec.required_secrets == ()

    def test_read_only_tools_are_read_with_no_hint(self) -> None:
        specs = _specs_by_name(BrowserToolProvider(_factory))
        for name in _READ_ONLY:
            assert specs[name].side_effect == "read"
            assert specs[name].policy_hint is None

    def test_risky_interactions_confirm(self) -> None:
        specs = _specs_by_name(BrowserToolProvider(_factory))
        for name in _CONFIRM:
            assert specs[name].side_effect == "write"
            assert specs[name].policy_hint == "confirm"

    def test_navigate_is_write_without_confirm(self) -> None:
        spec = _specs_by_name(BrowserToolProvider(_factory))["browser_navigate"]
        assert spec.side_effect == "write"
        assert spec.policy_hint is None

    def test_screenshot_flags_artifact_output(self) -> None:
        spec = _specs_by_name(BrowserToolProvider(_factory))["browser_screenshot"]
        assert spec.artifact_output is True


# ---------------------------------------------------------------------------
# No schema change / no behavior change
# ---------------------------------------------------------------------------


class TestEquivalence:
    def test_schemas_identical_to_legacy_builder(self) -> None:
        legacy = ToolRegistry()
        build_browser_tools(legacy, _factory)

        provider_reg = ToolRegistry()
        names = register_provider(provider_reg, BrowserToolProvider(_factory))

        assert set(names) == EXPECTED
        assert provider_reg.list_tools() == legacy.list_tools()
        for name in legacy.list_tools():
            assert provider_reg.get_schema(name) == legacy.get_schema(name)
        # strongest proof: the OpenAI tool_defs are byte-identical
        assert provider_reg.tool_defs() == legacy.tool_defs()

    async def test_behavior_preserved_navigate_executes(self) -> None:
        registry = ToolRegistry()
        register_provider(registry, BrowserToolProvider(_factory))

        result = await registry.execute(
            "browser_navigate", '{"url": "https://example.com"}'
        )
        assert result["success"] is True
        assert result["url"] == "https://example.com"
        # _annotate_block ran on the success path -> behavior unchanged
        assert "blocked" in result
        assert "block_reason" in result


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
        "name": "browser-agent",
        "system_prompt": "You browse.",
        "llm_provider": "higress",
    }
    base.update(over)
    return AgentSpec(**base)


class TestLoaderSelection:
    def test_candidate_not_auto_enabled(self) -> None:
        providers = {"browser": BrowserToolProvider(_factory)}
        spec = _spec(providers=["browser"])  # referenced, nothing selected
        assert _resolve_tools(spec, ToolRegistry(), None, providers) == []

    def test_select_a_subset_via_tools(self) -> None:
        providers = {"browser": BrowserToolProvider(_factory)}
        spec = _spec(providers=["browser"], tools=["browser_navigate", "browser_click"])
        resolved = _resolve_tools(spec, ToolRegistry(), None, providers)
        assert [t.__name__ for t in resolved] == ["browser_navigate", "browser_click"]

    def test_select_via_tool_set(self) -> None:
        providers = {"browser": BrowserToolProvider(_factory)}
        sets = {"browsing": ["browser_navigate", "browser_read_dom"]}
        spec = _spec(providers=["browser"], tool_sets=["browsing"])
        resolved = _resolve_tools(spec, ToolRegistry(), sets, providers)
        assert [t.__name__ for t in resolved] == ["browser_navigate", "browser_read_dom"]

    def test_load_agent_builds_graph(self) -> None:
        providers = {"browser": BrowserToolProvider(_factory)}
        spec = _spec(providers=["browser"], tools=["browser_navigate"])
        loaded = load_agent_from_spec(
            spec,
            tool_registry=ToolRegistry(),
            llm_registry=_llm_registry(),
            tool_providers=providers,
        )
        assert loaded.graph is not None
