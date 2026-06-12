"""``AgentSpec`` — the declarative shape of an agent definition."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


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
    temperature: float = 0.7
    max_tokens: int = 4096
    tools: list[str] = Field(default_factory=list, description="ToolRegistry names")
    skills: list[str] = Field(default_factory=list, description="SkillRegistry names")
    max_iterations: int = 10
