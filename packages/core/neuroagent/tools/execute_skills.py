"""execute_skills tool: load a registered skill's prompt by name (no code exec).

VeADK-style ``execute_skills`` scoped to the safe subset: it returns a skill's
prompt text via :meth:`SkillRegistry.load_prompt` so the agent can pull a skill's
instructions into context. Deliberate boundaries:

- **No code execution.** It runs no shell and spawns no process; it only reads
  the prompt the registry already holds.
- **No app / secret / signing changes.** It imports nothing from neuroagent_app,
  reads no secret, and does not touch skill signing or trust. It only *honors*
  the existing ``status`` field: a skill that is not ``active`` is refused, and
  pending (unsigned) skills are never registered, so they cannot be loaded here.

Exposed two equivalent ways, mirroring the rest of the tool layer: the legacy
builder :func:`build_execute_skills_tool` and :class:`ExecuteSkillsToolProvider`
(provider path), so ``execute_skills`` is selectable via ``AgentSpec.providers``
plus ``AgentSpec.tools`` / ``tool_sets``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from neuroagent.skills.registry import SkillRegistry
from neuroagent.tools.provider import ToolSpec, spec_from_callable
from neuroagent.tools.registry import ToolRegistry, callable_to_schema

ExecuteSkillsTool = Callable[[str], dict[str, Any]]


def _make_execute_skills(skill_registry: SkillRegistry) -> ExecuteSkillsTool:
    """Build the bound execute_skills callable over a *skill_registry*."""

    def execute_skills(skill: str) -> dict[str, Any]:
        """Load a registered skill's prompt by name (no code execution).

        Returns the skill's prompt text (via load_prompt) plus its name, version
        and description. Raises ValueError if the skill is unknown or not active;
        pending (unsigned) skills are never registered, so they cannot be loaded.
        """
        try:
            sk = skill_registry.get(skill)
        except KeyError as exc:
            raise ValueError(f"unknown skill {skill!r}") from exc
        if sk.status != "active":
            raise ValueError(f"skill {skill!r} is not active (status={sk.status!r})")
        return {
            "skill": sk.name,
            "version": sk.version,
            "description": sk.description,
            "prompt": skill_registry.load_prompt(sk),
        }

    return execute_skills


def build_execute_skills_tool(
    registry: ToolRegistry,
    skill_registry: SkillRegistry,
) -> ExecuteSkillsTool:
    """Return a bound execute_skills tool, registered under ``"execute_skills"``.

    *skill_registry* is the source of loadable skills; it is read only (never
    mutated). The tool is registered in *registry* with a reflected schema.
    """
    tool = _make_execute_skills(skill_registry)
    registry.register("execute_skills", tool, callable_to_schema(tool))
    return tool


class ExecuteSkillsToolProvider:
    """Yields the execute_skills tool bound to a SkillRegistry (provider path).

    Selectable through the provider path: reference the provider in
    ``AgentSpec.providers`` and enable ``"execute_skills"`` via ``AgentSpec.tools``
    or a ``tool_set``. The yielded spec carries descriptive metadata only
    (category ``"skill"``, read-only side effect); it never gates execution and
    performs no shell / app / secret access.
    """

    def __init__(self, skill_registry: SkillRegistry) -> None:
        self._skill_registry = skill_registry

    def provide(self) -> list[ToolSpec]:
        tool = _make_execute_skills(self._skill_registry)
        return [
            spec_from_callable(
                tool,
                category="skill",
                capabilities=("skill", "load"),
                side_effect="read",
            )
        ]
