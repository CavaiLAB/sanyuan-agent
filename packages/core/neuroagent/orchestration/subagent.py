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


# Structured run-progress callback: receives one dict per graph event —
# ``{"kind": "tool_call" | "tool_result", ...}`` with the same field shapes the
# chat SSE layer uses (id/name/arguments; tool_call_id/name/content/is_error/
# blocked/block_reason), plus ``{"kind": "usage", "usage": {promptTokens,
# completionTokens, totalTokens}}`` after each completed reasoning turn — the
# run's CUMULATIVE token accounting (never a delta, so a dropped frame cannot
# corrupt the figure). Synchronous by design so callers can relay onto an
# asyncio.Queue without awaiting inside the graph loop.
ToolEventObserver = Callable[[dict[str, Any]], None]


def _tool_events_from_update(node_name: str, update: Any) -> list[dict[str, Any]]:
    """Translate one graph node update into structured tool events.

    Mirrors the chat SSE translation (``gateway/sse.py``): a ``plan`` update
    whose last assistant message carries tool_calls yields one ``tool_call``
    event each; a ``tool_executor`` update yields one ``tool_result`` per tool
    message (dict or LangChain ToolMessage), carrying the honest outcome flags
    (``is_error`` / ``blocked`` / ``block_reason``).
    """
    if not isinstance(update, dict):
        return []
    events: list[dict[str, Any]] = []
    if node_name == "plan":
        messages = update.get("messages") or []
        if not messages:
            return []
        msg = messages[-1]
        tool_calls = (
            msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)
        )
        for tc in tool_calls or []:
            if isinstance(tc, dict):
                fn = tc.get("function", {}) or {}
                events.append(
                    {
                        "kind": "tool_call",
                        "tool_call_id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "arguments": fn.get("arguments", ""),
                    }
                )
            else:
                fn = getattr(tc, "function", None)
                events.append(
                    {
                        "kind": "tool_call",
                        "tool_call_id": getattr(tc, "id", ""),
                        "name": getattr(fn, "name", ""),
                        "arguments": getattr(fn, "arguments", ""),
                    }
                )
    elif node_name == "tool_executor":
        for msg in update.get("messages") or []:
            if isinstance(msg, dict):
                if msg.get("role") != "tool" and not msg.get("tool_call_id"):
                    continue
                tool_call_id = msg.get("tool_call_id", "")
                name = msg.get("name", "")
                content = msg.get("content", "")
                is_error = msg.get("status") == "error"
                extra: dict[str, Any] = msg
            else:
                tool_call_id = getattr(msg, "tool_call_id", None)
                if not tool_call_id:
                    continue
                name = getattr(msg, "name", "") or ""
                content = getattr(msg, "content", "")
                is_error = getattr(msg, "status", None) == "error"
                extra = getattr(msg, "additional_kwargs", None) or {}
            event: dict[str, Any] = {
                "kind": "tool_result",
                "tool_call_id": tool_call_id,
                "name": name,
                "content": str(content),
                "is_error": bool(is_error),
                "blocked": bool(extra.get("blocked")),
            }
            if extra.get("block_reason"):
                event["block_reason"] = str(extra["block_reason"])
            events.append(event)
    return events


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
    tool_event_observer: ToolEventObserver | None = None,
    **agent_kwargs: Any,
) -> SubagentOutcome:
    """Run the worker declared by *spec* on *task*; return answer + usage.

    The worker runs in a fresh, isolated state seeded only with its own system
    prompt and the delegated *task* — no parent conversation leaks in, only the
    final answer (and its token accounting) comes out. Usage is the run's
    cumulative state total (summed across reasoning turns by the graph's
    usage reducer).

    With a *tool_event_observer* the run streams the graph (``astream``
    updates) and relays each tool call/result as a structured event while it
    happens — the observability seam for orchestration layers. Without one,
    the original single-shot ``ainvoke`` path is used, byte-for-byte
    unchanged. Both paths return the same outcome for the same run.
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
    if tool_event_observer is None:
        result = await loaded.graph.ainvoke(state)
        raw_usage = result.get("usage") or {}
        messages = result.get("messages", [])
    else:
        # Streamed path: accumulate what ainvoke's reducers would have —
        # messages are appended per node, usage deltas come from the
        # plan/finalize turns (same accounting as the chat SSE layer).
        messages = []
        usage_total = {"promptTokens": 0, "completionTokens": 0}
        async for chunk in loaded.graph.astream(state, stream_mode="updates"):
            for node_name, update in chunk.items():
                if not isinstance(update, dict):
                    continue
                for event in _tool_events_from_update(node_name, update):
                    try:
                        tool_event_observer(event)
                    except Exception:
                        logger.warning("tool event observer failed (non-blocking)", exc_info=True)
                if node_name in ("plan", "finalize"):
                    turn_usage = update.get("usage")
                    if turn_usage:
                        usage_total["promptTokens"] += int(turn_usage.get("promptTokens", 0) or 0)
                        usage_total["completionTokens"] += int(
                            turn_usage.get("completionTokens", 0) or 0
                        )
                        # One reasoning turn (one LLM call) just finished:
                        # surface the run's cumulative accounting so the
                        # orchestration layer can stream live per-agent tokens.
                        try:
                            tool_event_observer(
                                {
                                    "kind": "usage",
                                    "usage": {
                                        "promptTokens": usage_total["promptTokens"],
                                        "completionTokens": usage_total["completionTokens"],
                                        "totalTokens": usage_total["promptTokens"]
                                        + usage_total["completionTokens"],
                                    },
                                }
                            )
                        except Exception:
                            logger.warning(
                                "usage event observer failed (non-blocking)", exc_info=True
                            )
                messages.extend(update.get("messages") or [])
        raw_usage = usage_total
    prompt_tokens = int(raw_usage.get("promptTokens", 0) or 0)
    completion_tokens = int(raw_usage.get("completionTokens", 0) or 0)
    return SubagentOutcome(
        answer=_final_answer({"messages": messages}),
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
