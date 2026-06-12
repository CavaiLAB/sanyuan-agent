"""``AgentSpec`` — the declarative shape of an agent definition."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ToolPolicyRule(BaseModel):
    """One ordered permission rule: glob *pattern* over tool names + action."""

    model_config = ConfigDict(extra="forbid")

    pattern: str = Field(min_length=1)
    action: Literal["allow", "deny", "confirm"]


class AgentSpec(BaseModel):
    """Declarative definition of a single agent.

    Names under ``llm_provider`` / ``tools`` / ``skills`` are *references* —
    they must resolve against the registries handed to
    :func:`neuroagent.config.load_agent_from_spec`.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    system_prompt: str
    llm_provider: str = Field(description="name registered in LLMProviderRegistry")
    # Overrides the provider's default_model (a litellm model id). Empty = inherit.
    model: str = ""
    # Ordered alternative model ids tried on a transient failure of the primary.
    fallback_models: list[str] = Field(default_factory=list)
    temperature: float = 0.7
    max_tokens: int = 4096
    tools: list[str] = Field(default_factory=list, description="ToolRegistry names")
    skills: list[str] = Field(default_factory=list, description="SkillRegistry names")
    max_iterations: int = 10
    # Rough token bound for the message window sent to the LLM each turn.
    context_token_budget: int = Field(default=24000, gt=0)
    # Ordered permission rules (first match wins); default applies when none match.
    tool_policy: list[ToolPolicyRule] = Field(default_factory=list)
    tool_policy_default: Literal["allow", "deny", "confirm"] = "allow"
