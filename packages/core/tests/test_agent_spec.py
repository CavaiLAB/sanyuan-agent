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
from neuroagent.config import (
    AgentSpec,
    LoadedAgent,
    load_agent_from_spec,
    load_agent_from_yaml,
)
from neuroagent.models import LLMProvider, LLMProviderRegistry
from neuroagent.skills.registry import Skill, SkillRegistry
from neuroagent.tools.registry import ToolRegistry
from pydantic import ValidationError

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
            api_key_env="NEUROAGENT_LLM_API_KEY",
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
        assert spec.skills == []
        assert spec.max_iterations == 10

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
        loaded = load_agent_from_spec(
            spec, tool_registry=tool_registry, llm_registry=llm_registry
        )
        assert isinstance(loaded, LoadedAgent)
        assert loaded.spec is spec
        assert loaded.graph is not None

    def test_initial_messages_prepends_system_prompt(
        self, tool_registry: ToolRegistry, llm_registry: LLMProviderRegistry
    ) -> None:
        spec = AgentSpec(**_minimal_dict())
        loaded = load_agent_from_spec(
            spec, tool_registry=tool_registry, llm_registry=llm_registry
        )
        msgs = loaded.initial_messages("hello")
        assert msgs[0] == {"role": "system", "content": "You are a careful researcher."}
        assert msgs[-1] == {"role": "user", "content": "hello"}

    def test_empty_system_prompt_yields_no_system_message(
        self, tool_registry: ToolRegistry, llm_registry: LLMProviderRegistry
    ) -> None:
        spec = AgentSpec(name="a", system_prompt="", llm_provider="higress")
        loaded = load_agent_from_spec(
            spec, tool_registry=tool_registry, llm_registry=llm_registry
        )
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
        loaded = load_agent_from_spec(
            spec, tool_registry=tool_registry, llm_registry=llm_registry
        )

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
        loaded = load_agent_from_spec(
            spec, tool_registry=tool_registry, llm_registry=llm_registry
        )
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
