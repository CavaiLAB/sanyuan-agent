"""Instantiate an agent graph from an :class:`AgentSpec`.

Resolves the spec's provider/tool/skill *references* against the supplied
registries (failing loudly on a dangling reference), then delegates to the
existing :func:`neuroagent.agents.simple.build_simple_agent`. No existing call
site changes — the hardcoded path and this declarative path coexist.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from neuroagent.agents.simple import build_simple_agent
from neuroagent.models import LLMProviderRegistry
from neuroagent.skills.registry import SkillRegistry
from neuroagent.tools.registry import ToolRegistry

from .spec import AgentSpec


@dataclass(frozen=True)
class LoadedAgent:
    """A compiled agent graph plus the spec it was built from."""

    graph: Any
    spec: AgentSpec

    def initial_messages(self, user_content: str) -> list[dict[str, str]]:
        """Build the opening message list: system prompt (if any) then the user turn."""
        messages: list[dict[str, str]] = []
        if self.spec.system_prompt:
            messages.append({"role": "system", "content": self.spec.system_prompt})
        messages.append({"role": "user", "content": user_content})
        return messages


def load_agent_from_spec(
    spec: AgentSpec,
    *,
    tool_registry: ToolRegistry,
    llm_registry: LLMProviderRegistry,
    skill_registry: SkillRegistry | None = None,
    **agent_kwargs: Any,
) -> LoadedAgent:
    """Build a :class:`LoadedAgent` from *spec*.

    *agent_kwargs* (tenant_id / conversation_id / user_id / run_id / memory
    backends) pass straight through to ``build_simple_agent``.

    Raises ``KeyError`` if the provider reference is unknown, ``ValueError`` if
    a tool or skill reference is unknown.
    """
    provider = llm_registry.get(spec.llm_provider)  # KeyError if missing
    overrides: dict[str, Any] = {}
    if spec.model:
        overrides["default_model"] = spec.model
    if spec.fallback_models:
        overrides["fallback_models"] = tuple(spec.fallback_models)
    if overrides:
        provider = dataclasses.replace(provider, **overrides)

    tools = []
    for tool_name in spec.tools:
        if not tool_registry.has_tool(tool_name):
            raise ValueError(
                f"AgentSpec {spec.name!r} references unknown tool {tool_name!r}"
            )
        tools.append(tool_registry.get(tool_name))

    scoped_skills: SkillRegistry | None = None
    if spec.skills:
        scoped_skills = SkillRegistry()
        source = skill_registry or SkillRegistry()
        for skill_name in spec.skills:
            try:
                skill = source.get(skill_name)
            except KeyError as exc:
                raise ValueError(
                    f"AgentSpec {spec.name!r} references unknown skill {skill_name!r}"
                ) from exc
            scoped_skills.register(skill)

    graph = build_simple_agent(
        llm_provider=provider,
        tools=tools,
        max_iter=spec.max_iterations,
        temperature=spec.temperature,
        max_tokens=spec.max_tokens,
        skill_registry=scoped_skills,
        **agent_kwargs,
    )
    return LoadedAgent(graph=graph, spec=spec)
