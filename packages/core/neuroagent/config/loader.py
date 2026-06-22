"""Instantiate an agent graph from an :class:`AgentSpec`.

Resolves the spec's provider/tool/skill *references* against the supplied
registries (failing loudly on a dangling reference), then delegates to the
existing :func:`neuroagent.agents.simple.build_simple_agent`. No existing call
site changes — the hardcoded path and this declarative path coexist.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from neuroagent.agents.simple import build_simple_agent
from neuroagent.models import LLMProviderRegistry
from neuroagent.security.tool_policy import PolicyRule, ToolPolicy
from neuroagent.skills.registry import SkillRegistry
from neuroagent.tools.provider import ToolProvider
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


def _provider_candidates(
    spec: AgentSpec,
    tool_registry: ToolRegistry,
    tool_providers: Mapping[str, ToolProvider] | None,
) -> dict[str, Callable[..., Any]]:
    """Expand ``spec.providers`` into a ``name -> callable`` candidate map.

    A referenced provider only *contributes candidates* to the tool space; it
    never enables a tool by itself (only ``tools`` / ``tool_sets`` do that).
    Conflicts are loud: a candidate may not shadow a base-registry tool, nor
    collide with another provider's candidate — this is what keeps MCP / builtin
    / provider tool names from silently overwriting each other.
    """
    available = tool_providers or {}
    candidates: dict[str, Callable[..., Any]] = {}
    for provider_name in spec.providers:
        provider = available.get(provider_name)
        if provider is None:
            raise ValueError(
                f"AgentSpec {spec.name!r} references unknown provider {provider_name!r}"
            )
        for tool_spec in provider.provide():
            if tool_registry.has_tool(tool_spec.name):
                raise ValueError(
                    f"AgentSpec {spec.name!r}: provider {provider_name!r} tool "
                    f"{tool_spec.name!r} conflicts with a base registry tool"
                )
            if tool_spec.name in candidates:
                raise ValueError(
                    f"AgentSpec {spec.name!r}: provider {provider_name!r} tool "
                    f"{tool_spec.name!r} conflicts with another provider candidate"
                )
            candidates[tool_spec.name] = tool_spec.callable
    return candidates


def _enabled_tool_names(
    spec: AgentSpec,
    tool_sets: Mapping[str, Sequence[str]] | None,
) -> list[str]:
    """Final enabled names: ``tool_sets`` (in order) then ``tools``, de-duped.

    De-duplication is stable (first occurrence wins); a name appearing in
    several sets or also in ``tools`` is not an error.
    """
    sets = tool_sets or {}
    ordered: list[str] = []
    for set_name in spec.tool_sets:
        members = sets.get(set_name)
        if members is None:
            raise ValueError(f"AgentSpec {spec.name!r} references unknown tool set {set_name!r}")
        ordered.extend(members)
    ordered.extend(spec.tools)

    seen: set[str] = set()
    deduped: list[str] = []
    for name in ordered:
        if name not in seen:
            seen.add(name)
            deduped.append(name)
    return deduped


def _resolve_tools(
    spec: AgentSpec,
    tool_registry: ToolRegistry,
    tool_sets: Mapping[str, Sequence[str]] | None,
    tool_providers: Mapping[str, ToolProvider] | None,
) -> list[Callable[..., Any]]:
    """Resolve enabled names to callables; base registry wins over candidates.

    Does not mutate *tool_registry* — provider candidates are resolved from a
    local map, never registered into the passed registry.
    """
    candidates = _provider_candidates(spec, tool_registry, tool_providers)
    resolved: list[Callable[..., Any]] = []
    for name in _enabled_tool_names(spec, tool_sets):
        if tool_registry.has_tool(name):
            resolved.append(tool_registry.get(name))
        elif name in candidates:
            resolved.append(candidates[name])
        else:
            raise ValueError(f"AgentSpec {spec.name!r} references unknown tool {name!r}")
    return resolved


def load_agent_from_spec(
    spec: AgentSpec,
    *,
    tool_registry: ToolRegistry,
    llm_registry: LLMProviderRegistry,
    skill_registry: SkillRegistry | None = None,
    tool_sets: Mapping[str, Sequence[str]] | None = None,
    tool_providers: Mapping[str, ToolProvider] | None = None,
    **agent_kwargs: Any,
) -> LoadedAgent:
    """Build a :class:`LoadedAgent` from *spec*.

    *agent_kwargs* (tenant_id / conversation_id / user_id / run_id / memory
    backends) pass straight through to ``build_simple_agent``.

    *tool_sets* maps a named set to its tool names; *tool_providers* maps a
    provider name to a :class:`~neuroagent.tools.provider.ToolProvider`. A
    referenced provider only contributes *candidate* tools — the final enabled
    set is ``spec.tool_sets`` expanded (in order) plus ``spec.tools``, stably
    de-duplicated. Neither argument mutates *tool_registry*.

    Raises ``KeyError`` if the provider reference is unknown, ``ValueError`` if
    a tool / tool-set / provider / skill reference is unknown, or if a provider
    candidate name conflicts with the base registry or another provider.
    """
    provider = llm_registry.get(spec.llm_provider)  # KeyError if missing
    overrides: dict[str, Any] = {}
    if spec.model:
        overrides["default_model"] = spec.model
    if spec.fallback_models:
        overrides["fallback_models"] = tuple(spec.fallback_models)
    if overrides:
        provider = dataclasses.replace(provider, **overrides)

    tools = _resolve_tools(spec, tool_registry, tool_sets, tool_providers)

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

    policy: ToolPolicy | None = None
    if spec.tool_policy or spec.tool_policy_default != "allow":
        policy = ToolPolicy(
            [PolicyRule(rule.pattern, rule.action) for rule in spec.tool_policy],
            default=spec.tool_policy_default,
        )

    graph = build_simple_agent(
        llm_provider=provider,
        tools=tools,
        max_iter=spec.max_iterations,
        temperature=spec.temperature,
        max_tokens=spec.max_tokens,
        skill_registry=scoped_skills,
        tool_policy=policy,
        context_token_budget=spec.context_token_budget,
        **agent_kwargs,
    )
    return LoadedAgent(graph=graph, spec=spec)
