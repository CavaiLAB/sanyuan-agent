"""Simplified LangGraph agent (plan → act → tool_executor → observe → plan)."""

import inspect
import logging
import os
import time
from collections.abc import Callable
from typing import Annotated, Any, cast

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from neuroagent.agents.tool_executor import tool_executor_node
from neuroagent.memory.episodic import EpisodicMemory
from neuroagent.memory.sensory import SensoryEvent
from neuroagent.memory.working import WorkingMemory
from neuroagent.models import LLMProvider
from neuroagent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """Shared state across the agent graph."""

    messages: Annotated[list[dict[str, Any]], list.__add__]
    next_action: str | None
    tool_results: Annotated[list[dict[str, Any]], list.__add__]
    iteration: int
    max_iter: int
    sensory: Any
    working: Any


def _to_tool_def(tool: Callable[..., Any]) -> dict[str, Any]:
    """Convert a plain callable to an OpenAI-style tool definition."""
    sig = inspect.signature(tool)
    props: dict[str, dict[str, str]] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        ann = param.annotation
        if ann is str:
            props[name] = {"type": "string"}
        elif ann is int:
            props[name] = {"type": "integer"}
        else:
            props[name] = {"type": "string"}
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {
        "type": "function",
        "function": {
            "name": tool.__name__,
            "description": (tool.__doc__ or "").strip(),
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        },
    }


async def _call_llm(
    provider: LLMProvider,
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]],
) -> Any:
    """Unified async LLM call via LiteLLM."""
    kwargs: dict[str, Any] = {
        "model": provider.default_model,
        "messages": messages,
    }
    if provider.api_key_env:
        kwargs["api_key"] = os.environ.get(provider.api_key_env)
    if provider.api_base:
        kwargs["api_base"] = provider.api_base
    if tools:
        kwargs["tools"] = tools
    # Delay import so tests can patch *before* the module is imported.
    import litellm

    return await litellm.acompletion(**kwargs)


def _get_content(response: Any) -> str:
    try:
        return response.choices[0].message.content or ""
    except (AttributeError, IndexError):
        return ""


def _get_tool_calls(response: Any) -> list[dict[str, Any]]:
    try:
        tcs = response.choices[0].message.tool_calls
        if not tcs:
            return []
        return [
            {
                "id": tc.id,
                "type": getattr(tc, "type", "function"),
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tcs
        ]
    except (AttributeError, IndexError):
        return []


def _inject_working_summary(
    messages: list[dict[str, Any]],
    working: WorkingMemory | None,
) -> list[dict[str, Any]]:
    """Prepend a system message with the working-memory summary."""
    if working is None:
        return messages
    summary = working.get_summary()
    if summary == "(no previous steps)":
        return messages
    system_msg: dict[str, Any] = {
        "role": "system",
        "content": f"Recent context:\n{summary}",
    }
    return [system_msg, *list(messages)]


def _inject_episodic_summary(
    messages: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Prepend a system message with recent episodic summaries.

    TODO(P1-PY-021): Replace naive injection with LLM-based relevance ranking.
    """
    if not episodes:
        return messages
    lines: list[str] = []
    for ep in episodes:
        lines.append(
            f"- Query: {ep['user_query']}\n  Answer: {ep['final_answer']}"
        )
    system_msg: dict[str, Any] = {
        "role": "system",
        "content": (
            "Related past conversations:\n"
            + "\n".join(lines)
            + "\n\n"
            "Use the above as background context if relevant."
        ),
    }
    return [system_msg, *list(messages)]


def _extract_user_query(messages: list[dict[str, Any]]) -> str:
    """Find the first user message in the conversation."""
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            return str(content)
    return ""


async def _maybe_inject_episodes(
    messages: list[dict[str, Any]],
    episodic: EpisodicMemory | None,
    tenant_id: str,
    conversation_id: str,
) -> list[dict[str, Any]]:
    """Fetch recent episodes and inject them as a system message.

    Failures are silently swallowed — episodic recall is a best-effort
    optimisation, not a hard dependency.
    """
    if episodic is None or not tenant_id or not conversation_id:
        return messages
    try:
        episodes = await episodic.recent_episodes(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            limit=5,
        )
        return _inject_episodic_summary(messages, episodes)
    except Exception as exc:
        logger.warning("episodic recall failed: %s", exc)
        return messages


async def _maybe_save_episode(
    state: AgentState,
    episodic: EpisodicMemory | None,
    tenant_id: str,
    conversation_id: str,
    session_started_at: float,
) -> None:
    """Persist the completed turn to episodic memory.

    Failures are logged but do not block the agent flow.
    """
    if episodic is None or not tenant_id or not conversation_id:
        return

    messages: list[Any] = state.get("messages", [])
    user_query = _extract_user_query(messages)
    # Final answer is the last assistant message
    final_answer = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            final_answer = msg.get("content", "")
            break

    working = state.get("working")
    tool_call_count = 0
    if working is not None:
        tool_call_count = len(getattr(working, "_steps", []))

    duration = time.time() - session_started_at

    try:
        await episodic.add_episode(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            user_query=user_query,
            final_answer=final_answer,
            metadata={
                "tool_call_count": tool_call_count,
                "duration": round(duration, 2),
            },
        )
    except Exception as exc:
        logger.warning("episodic save failed: %s", exc)


def build_simple_agent(
    llm_provider: LLMProvider,
    tools: list[Callable[..., Any]],
    max_iter: int = 10,
    *,
    episodic_memory: EpisodicMemory | None = None,
    tenant_id: str = "",
    conversation_id: str = "",
) -> Any:
    """Build and compile a plan/act/tool_executor/observe agent graph."""
    registry = ToolRegistry()
    for t in tools:
        schema = _to_tool_def(t)["function"]
        registry.register(t.__name__, t, schema)

    session_started_at = time.time()

    async def plan(state: AgentState) -> dict[str, Any]:
        working = state.get("working")
        sensory = state.get("sensory")

        # Feed user input into sensory memory
        if sensory is not None and state["messages"]:
            last = state["messages"][-1]
            if isinstance(last, dict) and last.get("role") == "user":
                sensory.append(
                    SensoryEvent(kind="user", content=last.get("content", ""))
                )

        llm_messages = _inject_working_summary(state["messages"], working)
        # TODO(P1-PY-021): Replace naive recall with relevance-ranked retrieval.
        llm_messages = await _maybe_inject_episodes(
            llm_messages,
            episodic_memory,
            tenant_id,
            conversation_id,
        )
        response = await _call_llm(
            llm_provider, llm_messages, registry.tool_defs()
        )
        tcs = _get_tool_calls(response)
        if tcs:
            msg: dict[str, Any] = {
                "role": "assistant",
                "content": _get_content(response),
                "tool_calls": tcs,
            }
            return {"next_action": "tool_call", "messages": [msg]}
        msg = {"role": "assistant", "content": _get_content(response)}

        # Session ends here — save episode asynchronously (best-effort)
        await _maybe_save_episode(
            state, episodic_memory, tenant_id, conversation_id, session_started_at
        )
        return {"next_action": "stop", "messages": [msg]}

    async def act(state: AgentState) -> dict[str, Any]:
        """Routing node — decides whether tool execution is needed."""
        last_msg = state["messages"][-1] if state["messages"] else {}
        has_tool_calls = (
            last_msg.get("role") == "assistant" and last_msg.get("tool_calls")
        )
        return {"next_action": "tool_call" if has_tool_calls else "stop"}

    async def observe(state: AgentState) -> dict[str, Any]:
        current_iter = state.get("iteration", 0)
        sensory = state.get("sensory")
        working = state.get("working")

        # Write tool_call + result into both memory layers
        if sensory is not None and working is not None:
            messages: list[Any] = state.get("messages", [])
            if messages:
                # Find the last assistant message
                last_assistant_idx = -1
                for i in range(len(messages) - 1, -1, -1):
                    msg = messages[i]
                    if (
                        isinstance(msg, dict)
                        and msg.get("role") == "assistant"
                    ):
                        last_assistant_idx = i
                        break

                if last_assistant_idx >= 0:
                    assistant_msg = messages[last_assistant_idx]
                    tcs = assistant_msg.get("tool_calls", [])
                    if tcs:
                        # Only process if there are fresh ToolMessages after it
                        plan = assistant_msg.get("content", "")
                        for tc in tcs:
                            fn = tc.get("function", {})
                            tool_name = fn.get("name", "")
                            args = fn.get("arguments", "{}")
                            for j in range(
                                last_assistant_idx + 1, len(messages)
                            ):
                                tm = messages[j]
                                tc_id = getattr(tm, "tool_call_id", None)
                                if tc_id == tc.get("id"):
                                    result = getattr(tm, "content", "")
                                    sensory.append(
                                        SensoryEvent(
                                            kind="tool_result",
                                            content=(
                                                f"{tool_name} → {result}"
                                            ),
                                        )
                                    )
                                    working.add_step(
                                        plan=plan,
                                        action=f"{tool_name}({args})",
                                        result=result,
                                    )
                                    break

        # Backward-compat: inject legacy tool_results if present.
        if state.get("tool_results"):
            return {
                "messages": state["tool_results"],
                "iteration": current_iter + 1,
            }
        return {"iteration": current_iter + 1}

    def route_after_act(state: AgentState) -> str:
        last_msg = state["messages"][-1] if state["messages"] else {}
        if last_msg.get("role") == "assistant" and last_msg.get("tool_calls"):
            return "tool_executor"
        return "observe"

    def should_continue(state: AgentState) -> str:
        limit = state.get("max_iter", max_iter)
        if (
            state["next_action"] == "tool_call"
            and state.get("iteration", 0) < limit
        ):
            return "plan"
        return END

    async def tool_executor(state: AgentState) -> dict[str, Any]:
        return await tool_executor_node(
            cast(dict[str, Any], state),
            {"configurable": {"tool_registry": registry}},
        )

    builder = StateGraph(AgentState)
    builder.add_node("plan", plan)
    builder.add_node("act", act)
    builder.add_node("tool_executor", tool_executor)
    builder.add_node("observe", observe)
    builder.set_entry_point("plan")
    builder.add_edge("plan", "act")
    builder.add_conditional_edges("act", route_after_act)
    builder.add_edge("tool_executor", "observe")
    builder.add_conditional_edges("observe", should_continue)

    return builder.compile()
