"""Simplified LangGraph agent (plan → act → tool_executor → observe → plan)."""

import inspect
import os
from collections.abc import Callable
from typing import Annotated, Any, cast

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from neuroagent.agents.tool_executor import tool_executor_node
from neuroagent.models import LLMProvider
from neuroagent.tools.registry import ToolRegistry


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


class AgentState(TypedDict):
    """Shared state across the agent graph."""

    messages: Annotated[list[dict[str, Any]], list.__add__]
    next_action: str | None
    tool_results: Annotated[list[dict[str, Any]], list.__add__]
    iteration: int
    max_iter: int


def build_simple_agent(
    llm_provider: LLMProvider,
    tools: list[Callable[..., Any]],
    max_iter: int = 10,
) -> Any:
    """Build and compile a plan/act/tool_executor/observe agent graph."""
    registry = ToolRegistry()
    for t in tools:
        schema = _to_tool_def(t)["function"]
        registry.register(t.__name__, t, schema)

    async def plan(state: AgentState) -> dict[str, Any]:
        response = await _call_llm(
            llm_provider, state["messages"], registry.tool_defs()
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
