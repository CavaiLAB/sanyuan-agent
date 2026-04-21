"""Simplified 3-node LangGraph agent (plan → act → observe)."""

import inspect
import json
import os
from typing import Annotated, Any, Callable

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from neuroagent.models import LLMProvider


def _to_tool_def(tool: Callable) -> dict:
    """Convert a plain callable to an OpenAI-style tool definition."""
    sig = inspect.signature(tool)
    props: dict[str, dict] = {}
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
    messages: list[dict],
    tools: list[dict],
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


def _get_tool_calls(response: Any) -> list[dict]:
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
    """Shared state across the 3-node graph."""

    messages: Annotated[list[dict], list.__add__]
    next_action: str | None
    tool_results: Annotated[list[dict], list.__add__]


def build_simple_agent(
    llm_provider: LLMProvider,
    tools: list[Callable],
) -> Any:
    """Build and compile a 3-node plan/act/observe agent graph."""
    tool_defs = [_to_tool_def(t) for t in tools]
    tools_map = {t.__name__: t for t in tools}

    async def plan(state: AgentState):
        response = await _call_llm(llm_provider, state["messages"], tool_defs)
        tcs = _get_tool_calls(response)
        if tcs:
            msg = {
                "role": "assistant",
                "content": _get_content(response),
                "tool_calls": tcs,
            }
            return {"next_action": "tool_call", "messages": [msg]}
        msg = {"role": "assistant", "content": _get_content(response)}
        return {"next_action": "stop", "messages": [msg]}

    async def act(state: AgentState):
        if state["next_action"] != "tool_call":
            return {}
        last_msg = state["messages"][-1]
        tcs = last_msg.get("tool_calls", [])
        if not tcs:
            return {}
        results = []
        for tc in tcs:
            fn_info = tc["function"]
            tool_fn = tools_map.get(fn_info["name"])
            if tool_fn is None:
                result = f"Tool '{fn_info['name']}' not found"
            else:
                try:
                    args = json.loads(fn_info["arguments"])
                    if inspect.iscoroutinefunction(tool_fn):
                        result = await tool_fn(**args)
                    else:
                        result = tool_fn(**args)
                except Exception as exc:
                    result = f"Error: {exc}"
            results.append({
                "tool_call_id": tc["id"],
                "role": "tool",
                "content": str(result),
                "name": fn_info["name"],
            })
        return {"tool_results": results}

    async def observe(state: AgentState):
        if state["next_action"] == "tool_call" and state["tool_results"]:
            return {"messages": state["tool_results"]}
        return {}

    def should_continue(state: AgentState):
        if state["next_action"] == "tool_call":
            return "plan"
        return END

    builder = StateGraph(AgentState)
    builder.add_node("plan", plan)
    builder.add_node("act", act)
    builder.add_node("observe", observe)
    builder.set_entry_point("plan")
    builder.add_edge("plan", "act")
    builder.add_edge("act", "observe")
    builder.add_conditional_edges("observe", should_continue)

    return builder.compile()
