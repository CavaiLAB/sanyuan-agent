"""LangGraph tool-executor node backed by ToolRegistry."""

from typing import Any, cast

from langchain_core.messages import ToolMessage

from neuroagent.tools.registry import ToolRegistry


def _last_assistant_tool_calls(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract tool_calls from the last assistant message, if any."""
    messages = state.get("messages", [])
    if not messages:
        return []
    last = messages[-1]
    if last.get("role") != "assistant":
        return []
    tcs = last.get("tool_calls")
    if not tcs:
        return []
    return cast(list[dict[str, Any]], tcs)


async def tool_executor_node(
    state: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Execute tool_calls from the last assistant message via ToolRegistry.

    Reads the last message's ``tool_calls``, looks each up in the
    ``ToolRegistry`` passed via ``config["configurable"]["tool_registry"]``,
    executes them in parallel (coroutines are gathered), and appends
    ``ToolMessage`` instances to the conversation.
    """
    registry: ToolRegistry | None = config.get("configurable", {}).get(
        "tool_registry"
    )
    if registry is None:
        return {}

    tcs = _last_assistant_tool_calls(state)
    if not tcs:
        return {}

    tool_messages: list[ToolMessage] = []
    for tc in tcs:
        fn_info = tc.get("function", {})
        name = fn_info.get("name", "")
        args = fn_info.get("arguments", "{}")
        result = await registry.execute(name, args)
        tool_messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tc.get("id", ""),
                name=name,
            )
        )

    return {"messages": tool_messages}
