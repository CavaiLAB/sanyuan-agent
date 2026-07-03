"""Run a worker agent (declared as an AgentSpec) and expose it as a tool."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from neuroagent.config import AgentSpec, load_agent_from_spec
from neuroagent.models import LLMProviderRegistry
from neuroagent.skills.registry import SkillRegistry
from neuroagent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SubagentOutcome:
    """Structured result of one worker run: the answer plus token accounting.

    ``usage`` follows the agent-state camelCase convention
    (``promptTokens`` / ``completionTokens``) with ``totalTokens`` derived at
    this boundary, so orchestration layers never re-derive it differently.
    """

    answer: str
    usage: dict[str, int] = field(default_factory=dict)


def _final_answer(state: dict[str, Any]) -> str:
    """Pull the worker's final answer: the last assistant message with no tool_calls."""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, dict):
            role = msg.get("role")
            tool_calls = msg.get("tool_calls")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "type", getattr(msg, "role", ""))
            tool_calls = getattr(msg, "tool_calls", None)
            content = getattr(msg, "content", "")
        if role == "assistant" and not tool_calls:
            return content if isinstance(content, str) else str(content)
    return ""


async def run_subagent_outcome(
    spec: AgentSpec,
    task: str,
    *,
    tool_registry: ToolRegistry,
    llm_registry: LLMProviderRegistry,
    skill_registry: SkillRegistry | None = None,
    **agent_kwargs: Any,
) -> SubagentOutcome:
    """Run the worker declared by *spec* on *task*; return answer + usage.

    The worker runs in a fresh, isolated state seeded only with its own system
    prompt and the delegated *task* — no parent conversation leaks in, only the
    final answer (and its token accounting) comes out. Usage is the run's
    cumulative state total (summed across reasoning turns by the graph's
    usage reducer).
    """
    loaded = load_agent_from_spec(
        spec,
        tool_registry=tool_registry,
        llm_registry=llm_registry,
        skill_registry=skill_registry,
        **agent_kwargs,
    )
    state: dict[str, Any] = {
        "messages": loaded.initial_messages(task),
        "next_action": None,
        "tool_results": [],
        "iteration": 0,
        "max_iter": spec.max_iterations,
        "sensory": None,
        "working": None,
        "usage": {},
    }
    result = await loaded.graph.ainvoke(state)
    raw_usage = result.get("usage") or {}
    prompt_tokens = int(raw_usage.get("promptTokens", 0) or 0)
    completion_tokens = int(raw_usage.get("completionTokens", 0) or 0)
    return SubagentOutcome(
        answer=_final_answer(result),
        usage={
            "promptTokens": prompt_tokens,
            "completionTokens": completion_tokens,
            "totalTokens": prompt_tokens + completion_tokens,
        },
    )


async def run_subagent(
    spec: AgentSpec,
    task: str,
    *,
    tool_registry: ToolRegistry,
    llm_registry: LLMProviderRegistry,
    skill_registry: SkillRegistry | None = None,
    **agent_kwargs: Any,
) -> str:
    """Run the worker declared by *spec* on *task*; return its final answer.

    Thin wrapper over :func:`run_subagent_outcome` for callers that only need
    the answer (e.g. the ``delegate_to_*`` tool contract, which must stay a
    plain string).
    """
    outcome = await run_subagent_outcome(
        spec,
        task,
        tool_registry=tool_registry,
        llm_registry=llm_registry,
        skill_registry=skill_registry,
        **agent_kwargs,
    )
    return outcome.answer


def _slug(name: str) -> str:
    """Turn an agent name into a tool-name-safe identifier suffix."""
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()
    return slug or "agent"


def make_subagent_tool(
    spec: AgentSpec,
    *,
    tool_registry: ToolRegistry,
    llm_registry: LLMProviderRegistry,
    skill_registry: SkillRegistry | None = None,
    **agent_kwargs: Any,
) -> Callable[[str], Any]:
    """Build an LLM-callable tool that delegates a task to the *spec* worker.

    The returned callable is named ``delegate_to_<slug>`` with a typed
    ``task: str`` parameter, so it slots into an orchestrator agent's tool list
    like any other tool. A worker failure is returned as an ``[subagent-error]``
    string rather than raised — a sub-agent must never crash the orchestrator.
    """

    async def delegate(task: str) -> str:
        try:
            return await run_subagent(
                spec,
                task,
                tool_registry=tool_registry,
                llm_registry=llm_registry,
                skill_registry=skill_registry,
                **agent_kwargs,
            )
        except Exception as exc:
            logger.warning("sub-agent %r failed: %s", spec.name, exc)
            return f"[subagent-error] {spec.name} failed: {exc}"

    delegate.__name__ = f"delegate_to_{_slug(spec.name)}"
    summary = spec.system_prompt.strip().replace("\n", " ")[:120]
    delegate.__doc__ = (
        f"Delegate a self-contained task to the {spec.name!r} sub-agent "
        f"and return its answer. {summary}"
    )
    return delegate
