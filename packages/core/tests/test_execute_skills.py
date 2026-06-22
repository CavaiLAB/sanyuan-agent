"""execute_skills (PR6): load a skill's prompt by name, over SkillRegistry only.

Pins:

1. Behavior - returns the active skill's prompt via load_prompt (incl. the
   adjacent prompt.md path), plus name/version/description; read-only; refuses
   unknown and non-active (pending/unsigned) skills.
2. Provider path - ExecuteSkillsToolProvider yields a clean spec with skill/read
   metadata, registers + executes, and is selectable through load_agent_from_spec
   (providers + tools / tool_sets).

No shell, no app import, no secret reads, no signing changes - it only honors the
existing ``status`` field.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from neuroagent.config import AgentSpec, load_agent_from_spec
from neuroagent.config.loader import _resolve_tools
from neuroagent.models import LLMProvider, LLMProviderRegistry
from neuroagent.skills import Skill, SkillRegistry
from neuroagent.tools import (
    ExecuteSkillsToolProvider,
    build_execute_skills_tool,
)
from neuroagent.tools.provider import ToolProvider, register_provider
from neuroagent.tools.registry import ToolRegistry


def _skill(
    name: str = "summarize",
    *,
    prompt_template: str = "Summarize: {input}",
    source_path: Path = Path("/fake/summarize/SKILL.md"),
    status: str = "active",
) -> Skill:
    return Skill(
        name=name,
        version="1.0.0",
        description="summarize text",
        tools=[],
        prompt_template=prompt_template,
        source_path=source_path,
        trigger_keywords=["summarize"],
        status=status,
    )


def _registry_with(*skills: Skill) -> SkillRegistry:
    reg = SkillRegistry()
    for sk in skills:
        reg.register(sk)
    return reg


# ---------------------------------------------------------------------------
# Behavior
# ---------------------------------------------------------------------------


class TestBehavior:
    def test_returns_prompt_and_metadata(self) -> None:
        skills = _registry_with(_skill())
        tool = build_execute_skills_tool(ToolRegistry(), skills)

        result = tool("summarize")

        assert result == {
            "skill": "summarize",
            "version": "1.0.0",
            "description": "summarize text",
            "prompt": "Summarize: {input}",  # falls back to prompt_template
        }

    def test_loads_adjacent_prompt_md(self, tmp_path: Path) -> None:
        # load_prompt prefers prompt.md next to SKILL.md - prove we go through it.
        (tmp_path / "prompt.md").write_text("FROM PROMPT.MD", encoding="utf-8")
        skills = _registry_with(
            _skill(prompt_template="inline-fallback", source_path=tmp_path / "SKILL.md")
        )
        tool = build_execute_skills_tool(ToolRegistry(), skills)

        assert tool("summarize")["prompt"] == "FROM PROMPT.MD"

    def test_unknown_skill_raises(self) -> None:
        tool = build_execute_skills_tool(ToolRegistry(), SkillRegistry())
        with pytest.raises(ValueError, match="unknown skill 'ghost'"):
            tool("ghost")

    def test_non_active_skill_refused(self) -> None:
        # A pending (unsigned) skill must never be loaded into a prompt.
        skills = _registry_with(_skill(status="pending"))
        tool = build_execute_skills_tool(ToolRegistry(), skills)
        with pytest.raises(ValueError, match="not active"):
            tool("summarize")

    def test_is_read_only(self) -> None:
        skills = _registry_with(_skill())
        tool = build_execute_skills_tool(ToolRegistry(), skills)
        before = list(skills.list())
        tool("summarize")
        assert list(skills.list()) == before  # registry untouched


# ---------------------------------------------------------------------------
# Provider path
# ---------------------------------------------------------------------------


class TestProvider:
    def test_provider_is_runtime_checkable(self) -> None:
        assert isinstance(ExecuteSkillsToolProvider(SkillRegistry()), ToolProvider)

    def test_provide_yields_one_clean_spec_with_metadata(self) -> None:
        (spec,) = ExecuteSkillsToolProvider(_registry_with(_skill())).provide()
        assert spec.name == "execute_skills"
        assert spec.category == "skill"
        assert spec.capabilities == ("skill", "load")
        assert spec.side_effect == "read"
        assert spec.artifact_output is False
        assert spec.policy_hint is None
        assert spec.required_secrets == ()  # no secrets
        props = spec.schema["parameters"]["properties"]
        assert set(props) == {"skill"}
        assert "arguments" not in props
        assert spec.schema["parameters"]["required"] == ["skill"]

    async def test_register_provider_then_execute(self) -> None:
        skills = _registry_with(_skill())
        registry = ToolRegistry()
        names = register_provider(registry, ExecuteSkillsToolProvider(skills))

        assert names == ["execute_skills"]
        result = await registry.execute("execute_skills", '{"skill": "summarize"}')
        assert result["skill"] == "summarize"
        assert result["prompt"] == "Summarize: {input}"

    def test_builder_and_provider_schemas_are_equivalent(self) -> None:
        skills = _registry_with(_skill())

        legacy = ToolRegistry()
        build_execute_skills_tool(legacy, skills)

        provider_reg = ToolRegistry()
        register_provider(provider_reg, ExecuteSkillsToolProvider(skills))

        assert legacy.get_schema("execute_skills") == provider_reg.get_schema("execute_skills")
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
            api_key_env="NEUROAGENT_LLM_API_KEY",
            default_model="openai/deepseek-v4-flash",
        )
    )
    return reg


def _spec(**over: Any) -> AgentSpec:
    base = {
        "name": "skiller",
        "system_prompt": "You use skills.",
        "llm_provider": "higress",
    }
    base.update(over)
    return AgentSpec(**base)


class TestLoaderSelection:
    def test_candidate_not_auto_enabled(self) -> None:
        providers = {"skills": ExecuteSkillsToolProvider(_registry_with(_skill()))}
        spec = _spec(providers=["skills"])  # referenced, nothing selected
        assert _resolve_tools(spec, ToolRegistry(), None, providers) == []

    def test_selected_via_tools(self) -> None:
        providers = {"skills": ExecuteSkillsToolProvider(_registry_with(_skill()))}
        spec = _spec(providers=["skills"], tools=["execute_skills"])
        resolved = _resolve_tools(spec, ToolRegistry(), None, providers)
        assert [t.__name__ for t in resolved] == ["execute_skills"]

    def test_selected_via_tool_set(self) -> None:
        providers = {"skills": ExecuteSkillsToolProvider(_registry_with(_skill()))}
        sets = {"skilling": ["execute_skills"]}
        spec = _spec(providers=["skills"], tool_sets=["skilling"])
        resolved = _resolve_tools(spec, ToolRegistry(), sets, providers)
        assert [t.__name__ for t in resolved] == ["execute_skills"]

    def test_load_agent_builds_graph(self) -> None:
        providers = {"skills": ExecuteSkillsToolProvider(_registry_with(_skill()))}
        spec = _spec(providers=["skills"], tools=["execute_skills"])
        loaded = load_agent_from_spec(
            spec,
            tool_registry=ToolRegistry(),
            llm_registry=_llm_registry(),
            tool_providers=providers,
        )
        assert loaded.graph is not None
