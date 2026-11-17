"""Declarative AgentSpec + loader: build an agent from a config, not hardcode.

Closes the SAD §8.2 ``build_lead_agent_from_config`` placeholder — an Agent can
now be declared (YAML/dict) and instantiated by the loader, instead of the
hardcoded ``build_simple_agent`` call sites. Pure harness addition; existing
direct callers are untouched.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError
from sanyuan_agent.config import (
    AgentSpec,
    LoadedAgent,
    load_agent_from_spec,
    load_agent_from_yaml,
)
from sanyuan_agent.config.loader import _resolve_tools
from sanyuan_agent.models import LLMProvider, LLMProviderRegistry
from sanyuan_agent.skills.registry import Skill, SkillRegistry
from sanyuan_agent.tools.provider import ToolSpec, spec_from_callable
from sanyuan_agent.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def echo(text: str) -> str:
    """Echo the input back (a trivial fake tool)."""
    return text


@pytest.fixture()
def tool_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register("echo", echo, {"type": "function", "function": {"name": "echo"}})
    return reg


@pytest.fixture()
def skill_registry() -> SkillRegistry:
    reg = SkillRegistry()
    reg.register(
        Skill(
            name="summarize",
            version="1.0.0",
            description="summarize text",
            tools=[],
            prompt_template="Summarize: {input}",
            source_path=Path("/fake/summarize/SKILL.md"),
            trigger_keywords=["summarize"],
        )
    )
    return reg


@pytest.fixture()
def llm_registry() -> LLMProviderRegistry:
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


def _minimal_dict(**over: Any) -> dict[str, Any]:
    base = {
        "name": "researcher",
        "system_prompt": "You are a careful researcher.",
        "llm_provider": "higress",
    }
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# AgentSpec schema
# ---------------------------------------------------------------------------


class TestAgentSpec:
    def test_minimal_valid_with_defaults(self) -> None:
        spec = AgentSpec(**_minimal_dict())
        assert spec.name == "researcher"
        assert spec.model == ""  # inherit provider default
        assert spec.temperature == 0.7
        assert spec.max_tokens == 4096
        assert spec.tools == []
        assert spec.tool_sets == []
        assert spec.providers == []
        assert spec.skills == []
        assert spec.max_iterations == 10

    def test_tool_sets_and_providers_preserve_order(self) -> None:
        spec = AgentSpec(
            **_minimal_dict(
                tool_sets=["web", "sandbox"],
                providers=["mcp_wx", "multimodal"],
            )
        )
        assert spec.tool_sets == ["web", "sandbox"]
        assert spec.providers == ["mcp_wx", "multimodal"]

    def test_tool_sets_and_providers_must_be_lists(self) -> None:
        with pytest.raises(ValidationError):
            AgentSpec(**_minimal_dict(tool_sets="web"))  # must be a list
        with pytest.raises(ValidationError):
            AgentSpec(**_minimal_dict(providers="mcp_wx"))  # must be a list

    @pytest.mark.parametrize("missing", ["name", "system_prompt", "llm_provider"])
    def test_missing_required_raises(self, missing: str) -> None:
        data = _minimal_dict()
        del data[missing]
        with pytest.raises(ValidationError):
            AgentSpec(**data)

    def test_wrong_types_raise(self) -> None:
        with pytest.raises(ValidationError):
            AgentSpec(**_minimal_dict(temperature="hot"))
        with pytest.raises(ValidationError):
            AgentSpec(**_minimal_dict(tools="echo"))  # must be a list

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentSpec(**_minimal_dict(unknown_field=1))


# ---------------------------------------------------------------------------
# load_agent_from_spec — reference validation
# ---------------------------------------------------------------------------


class TestLoaderValidation:
    def test_unknown_provider_raises(
        self, tool_registry: ToolRegistry, llm_registry: LLMProviderRegistry
    ) -> None:
        spec = AgentSpec(**_minimal_dict(llm_provider="missing"))
        with pytest.raises((KeyError, ValueError)):
            load_agent_from_spec(spec, tool_registry=tool_registry, llm_registry=llm_registry)

    def test_unknown_tool_raises(
        self, tool_registry: ToolRegistry, llm_registry: LLMProviderRegistry
    ) -> None:
        spec = AgentSpec(**_minimal_dict(tools=["echo", "nope"]))
        with pytest.raises(ValueError, match="nope"):
            load_agent_from_spec(spec, tool_registry=tool_registry, llm_registry=llm_registry)

    def test_unknown_skill_raises(
        self,
        tool_registry: ToolRegistry,
        skill_registry: SkillRegistry,
        llm_registry: LLMProviderRegistry,
    ) -> None:
        spec = AgentSpec(**_minimal_dict(skills=["nope"]))
        with pytest.raises(ValueError, match="nope"):
            load_agent_from_spec(
                spec,
                tool_registry=tool_registry,
                skill_registry=skill_registry,
                llm_registry=llm_registry,
            )


# ---------------------------------------------------------------------------
# load_agent_from_spec — construction
# ---------------------------------------------------------------------------


class TestLoaderConstruction:
    def test_returns_loaded_agent(
        self, tool_registry: ToolRegistry, llm_registry: LLMProviderRegistry
    ) -> None:
        spec = AgentSpec(**_minimal_dict(tools=["echo"]))
        loaded = load_agent_from_spec(spec, tool_registry=tool_registry, llm_registry=llm_registry)
        assert isinstance(loaded, LoadedAgent)
        assert loaded.spec is spec
        assert loaded.graph is not None

    def test_initial_messages_prepends_system_prompt(
        self, tool_registry: ToolRegistry, llm_registry: LLMProviderRegistry
    ) -> None:
        spec = AgentSpec(**_minimal_dict())
        loaded = load_agent_from_spec(spec, tool_registry=tool_registry, llm_registry=llm_registry)
        msgs = loaded.initial_messages("hello")
        assert msgs[0] == {"role": "system", "content": "You are a careful researcher."}
        assert msgs[-1] == {"role": "user", "content": "hello"}

    def test_empty_system_prompt_yields_no_system_message(
        self, tool_registry: ToolRegistry, llm_registry: LLMProviderRegistry
    ) -> None:
        spec = AgentSpec(name="a", system_prompt="", llm_provider="higress")
        loaded = load_agent_from_spec(spec, tool_registry=tool_registry, llm_registry=llm_registry)
        msgs = loaded.initial_messages("hi")
        assert msgs == [{"role": "user", "content": "hi"}]

    @pytest.mark.asyncio
    async def test_builds_runnable_graph_applies_spec_to_llm(
        self,
        tool_registry: ToolRegistry,
        llm_registry: LLMProviderRegistry,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import litellm

        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="done", tool_calls=None))],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2),
        )
        acompletion = AsyncMock(return_value=response)
        monkeypatch.setattr(litellm, "acompletion", acompletion)

        spec = AgentSpec(
            **_minimal_dict(
                model="openai/deepseek-v4-pro",
                temperature=0.2,
                max_tokens=512,
            )
        )
        loaded = load_agent_from_spec(spec, tool_registry=tool_registry, llm_registry=llm_registry)

        state = {
            "messages": loaded.initial_messages("hi"),
            "next_action": None,
            "tool_results": [],
            "iteration": 0,
            "max_iter": spec.max_iterations,
            "sensory": None,
            "working": None,
            "usage": {},
        }
        result = await loaded.graph.ainvoke(state)

        assert result is not None
        assert acompletion.await_count >= 1
        kwargs = acompletion.await_args.kwargs
        # model override + generation params from the spec reach litellm
        assert kwargs["model"] == "openai/deepseek-v4-pro"
        assert kwargs["temperature"] == 0.2
        assert kwargs["max_tokens"] == 512

    @pytest.mark.asyncio
    async def test_model_empty_inherits_provider_default(
        self,
        tool_registry: ToolRegistry,
        llm_registry: LLMProviderRegistry,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import litellm

        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="x", tool_calls=None))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )
        acompletion = AsyncMock(return_value=response)
        monkeypatch.setattr(litellm, "acompletion", acompletion)

        spec = AgentSpec(**_minimal_dict())  # model="" → inherit
        loaded = load_agent_from_spec(spec, tool_registry=tool_registry, llm_registry=llm_registry)
        state = {
            "messages": loaded.initial_messages("hi"),
            "next_action": None,
            "tool_results": [],
            "iteration": 0,
            "max_iter": spec.max_iterations,
            "sensory": None,
            "working": None,
            "usage": {},
        }
        await loaded.graph.ainvoke(state)
        assert acompletion.await_args.kwargs["model"] == "openai/deepseek-v4-flash"


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


class TestYamlLoader:
    def test_roundtrip(self, tmp_path: Path) -> None:
        yaml_text = (
            "name: researcher\n"
            "system_prompt: You are a careful researcher.\n"
            "llm_provider: higress\n"
            "model: openai/deepseek-v4-pro\n"
            "temperature: 0.3\n"
            "tools:\n"
            "  - echo\n"
            "skills:\n"
            "  - summarize\n"
            "max_iterations: 5\n"
        )
        path = tmp_path / "agent.yaml"
        path.write_text(yaml_text)

        spec = load_agent_from_yaml(path)

        assert spec == AgentSpec(
            name="researcher",
            system_prompt="You are a careful researcher.",
            llm_provider="higress",
            model="openai/deepseek-v4-pro",
            temperature=0.3,
            tools=["echo"],
            skills=["summarize"],
            max_iterations=5,
        )

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("name: x\n")  # missing required fields
        with pytest.raises(ValidationError):
            load_agent_from_yaml(path)


# ---------------------------------------------------------------------------
# load_agent_from_spec — tool_sets / providers selection (PR2)
# ---------------------------------------------------------------------------


def web_search(query: str) -> str:
    """Search the web (fake)."""
    return query


def image_gen(prompt: str) -> str:
    """Generate an image (fake)."""
    return prompt


class _Provider:
    """Minimal ToolProvider: yields a fixed list of specs."""

    def __init__(self, specs: list[ToolSpec]) -> None:
        self._specs = specs

    def provide(self) -> list[ToolSpec]:
        return self._specs


def _provider(*fns: Any) -> _Provider:
    return _Provider([spec_from_callable(fn) for fn in fns])


class TestLoaderToolSelection:
    def test_provider_candidate_not_auto_enabled(
        self, tool_registry: ToolRegistry, llm_registry: LLMProviderRegistry
    ) -> None:
        """A referenced provider whose tools aren't selected enables nothing."""
        providers = {"web": _provider(web_search)}
        spec = AgentSpec(**_minimal_dict(providers=["web"]))  # nothing selected

        # Builds without error …
        loaded = load_agent_from_spec(
            spec,
            tool_registry=tool_registry,
            llm_registry=llm_registry,
            tool_providers=providers,
        )
        assert loaded.graph is not None
        # … and resolves to zero enabled tools (no auto-enable).
        assert _resolve_tools(spec, tool_registry, None, providers) == []

    def test_provider_tool_selected_via_tools(
        self, tool_registry: ToolRegistry, llm_registry: LLMProviderRegistry
    ) -> None:
        providers = {"web": _provider(web_search)}
        spec = AgentSpec(**_minimal_dict(providers=["web"], tools=["web_search"]))

        resolved = _resolve_tools(spec, tool_registry, None, providers)
        assert [t.__name__ for t in resolved] == ["web_search"]

        loaded = load_agent_from_spec(
            spec,
            tool_registry=tool_registry,
            llm_registry=llm_registry,
            tool_providers=providers,
        )
        assert loaded.graph is not None

    def test_tool_set_expands_with_stable_dedup(self, tool_registry: ToolRegistry) -> None:
        providers = {"p": _provider(web_search, image_gen)}
        sets = {"bundle": ["web_search", "image_gen"]}
        spec = AgentSpec(
            **_minimal_dict(
                providers=["p"],
                tool_sets=["bundle"],
                # web_search duplicates the set; echo is a base-registry tool.
                tools=["web_search", "echo"],
            )
        )

        resolved = _resolve_tools(spec, tool_registry, sets, providers)
        # tool_sets (in order) then tools, first occurrence wins.
        assert [t.__name__ for t in resolved] == ["web_search", "image_gen", "echo"]

    def test_unknown_provider_raises(
        self, tool_registry: ToolRegistry, llm_registry: LLMProviderRegistry
    ) -> None:
        spec = AgentSpec(**_minimal_dict(providers=["ghost"]))
        with pytest.raises(ValueError, match="unknown provider"):
            load_agent_from_spec(
                spec,
                tool_registry=tool_registry,
                llm_registry=llm_registry,
                tool_providers={},
            )

    def test_unknown_tool_set_raises(
        self, tool_registry: ToolRegistry, llm_registry: LLMProviderRegistry
    ) -> None:
        spec = AgentSpec(**_minimal_dict(tool_sets=["ghost"]))
        with pytest.raises(ValueError, match="unknown tool set"):
            load_agent_from_spec(
                spec,
                tool_registry=tool_registry,
                llm_registry=llm_registry,
                tool_sets={},
            )

    def test_unknown_expanded_tool_raises(
        self, tool_registry: ToolRegistry, llm_registry: LLMProviderRegistry
    ) -> None:
        spec = AgentSpec(**_minimal_dict(tool_sets=["bundle"]))
        with pytest.raises(ValueError, match="ghost"):
            load_agent_from_spec(
                spec,
                tool_registry=tool_registry,
                llm_registry=llm_registry,
                tool_sets={"bundle": ["ghost"]},
            )

    def test_provider_base_conflict_raises(
        self, tool_registry: ToolRegistry, llm_registry: LLMProviderRegistry
    ) -> None:
        """A provider candidate may not shadow a base-registry tool name."""
        # base registry already has "echo"; provider exposes another "echo".
        clashing = _Provider([spec_from_callable(web_search, name="echo")])
        spec = AgentSpec(**_minimal_dict(providers=["p"]))
        with pytest.raises(ValueError, match="conflicts with a base registry tool"):
            load_agent_from_spec(
                spec,
                tool_registry=tool_registry,
                llm_registry=llm_registry,
                tool_providers={"p": clashing},
            )

    def test_provider_provider_conflict_raises(
        self, tool_registry: ToolRegistry, llm_registry: LLMProviderRegistry
    ) -> None:
        p1 = _Provider([spec_from_callable(web_search, name="dup")])
        p2 = _Provider([spec_from_callable(image_gen, name="dup")])
        spec = AgentSpec(**_minimal_dict(providers=["a", "b"]))
        with pytest.raises(ValueError, match="another provider candidate"):
            load_agent_from_spec(
                spec,
                tool_registry=tool_registry,
                llm_registry=llm_registry,
                tool_providers={"a": p1, "b": p2},
            )

    def test_does_not_mutate_tool_registry(
        self, tool_registry: ToolRegistry, llm_registry: LLMProviderRegistry
    ) -> None:
        before = set(tool_registry.list_tools())
        providers = {"web": _provider(web_search)}
        spec = AgentSpec(**_minimal_dict(providers=["web"], tools=["web_search"]))
        load_agent_from_spec(
            spec,
            tool_registry=tool_registry,
            llm_registry=llm_registry,
            tool_providers=providers,
        )
        assert set(tool_registry.list_tools()) == before  # candidate not registered
