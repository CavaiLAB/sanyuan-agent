"""LangGraph tool-executor node backed by ToolRegistry."""

import json
import logging
from typing import Any, cast

from langchain_core.messages import ToolMessage

from neuroagent.agents.errors import (
    ToolArgsValidationError,
    ToolCircuitBreakerError,
    ToolExecutionError,
    ToolNotFoundError,
)
from neuroagent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def _error_tool_message(content: str, tool_call_id: str, name: str) -> ToolMessage:
    """A failed tool call, marked ``status="error"``.

    The status is the structured outcome signal: the SSE layer reads it
    (``is_error``) so a failed tool renders red in the UI instead of a green
    success. The *content* still carries a clean, model-readable reason so the
    LLM can self-correct on the next turn. (LangChain drops ``status`` when the
    message is converted back to the OpenAI tool role, so the model only sees the
    reason text — never the status flag.)
    """
    return ToolMessage(
        content=content, tool_call_id=tool_call_id, name=name, status="error"
    )


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


def _sanitize_tool_call(tc: dict[str, Any]) -> dict[str, Any] | None:
    """Clean a potentially malformed tool_call dict.

    Returns *None* if the structure is too broken to recover.
    """
    fn = tc.get("function")
    if not isinstance(fn, dict):
        return None
    name = fn.get("name", "")
    if not isinstance(name, str):
        name = str(name)
    args = fn.get("arguments", "{}")
    if not isinstance(args, str):
        args = json.dumps(args) if args is not None else "{}"
    tc_id = tc.get("id", "")
    if not isinstance(tc_id, str):
        tc_id = str(tc_id)
    return {
        "id": tc_id,
        "type": tc.get("type", "function"),
        "function": {"name": name.strip(), "arguments": args.strip() or "{}"},
    }


def _validate_args(
    name: str,
    args: dict[str, Any],
    registry: ToolRegistry,
) -> None:
    """Validate arguments against the tool's JSON schema.

    Raises *ToolArgsValidationError* on mismatch.
    """
    if not registry.has_tool(name):
        raise ToolNotFoundError(name, registry.list_tools())

    schema = registry.get_schema(name)
    parameters = schema.get("parameters", {})
    if not parameters:
        return

    from jsonschema import ValidationError, validate

    try:
        validate(instance=args, schema=parameters)
    except ValidationError as exc:
        errors = [str(exc.message)]
        raise ToolArgsValidationError(name, errors) from exc


async def _recover_tool_call(
    tc: dict[str, Any],
    registry: ToolRegistry,
) -> ToolMessage:
    """Validate, execute, and recover a single tool_call.

    On *any* failure a ``ToolMessage`` is returned whose *content*
    carries a structured error description so the LLM can self-correct.
    """
    tc_id = tc.get("id", "")
    fn_info = tc.get("function", {})
    name = fn_info.get("name", "")
    args_raw = fn_info.get("arguments", "{}")

    # 1. Sanitize / malformed guard
    if not name:
        return _error_tool_message(
            "[tool-error] Missing tool name. Please provide a valid function name.",
            tc_id,
            "unknown",
        )

    # 2. Tool not found → list available tools
    if not registry.has_tool(name):
        available = registry.list_tools()
        return _error_tool_message(
            f"[tool-error] Tool '{name}' not found. "
            f"Available tools: {', '.join(available)}",
            tc_id,
            name,
        )

    # 3. Parse JSON arguments
    try:
        args = json.loads(args_raw)
    except json.JSONDecodeError as exc:
        schema = registry.get_schema(name)
        params = schema.get("parameters", {})
        props = params.get("properties", {})
        expected = ", ".join(props.keys()) if props else "(none defined)"
        return _error_tool_message(
            f"[tool-error] Invalid JSON in arguments for '{name}': {exc}. "
            f"Expected parameters: {expected}",
            tc_id,
            name,
        )

    # 4. Schema validation
    try:
        _validate_args(name, args, registry)
    except ToolArgsValidationError as exc:
        return _error_tool_message(
            f"[tool-error] Argument validation failed for '{name}': "
            f"{'; '.join(exc.errors)}. Please correct and retry.",
            tc_id,
            name,
        )

    # 5. Execute with circuit-breaker / exception recovery
    try:
        result = await registry.execute(name, args_raw)
    except ToolCircuitBreakerError as exc:
        return _error_tool_message(
            f"[tool-error] Circuit breaker open for '{name}' "
            f"({exc.fail_count}/{exc.threshold} consecutive failures). "
            f"Please use an alternative tool or wait.",
            tc_id,
            name,
        )
    except ToolExecutionError as exc:
        # The Python traceback is for operators, never the chat surface — log it
        # server-side and hand the model a clean one-line reason. Streaming the
        # raw traceback used to dump internal paths + stack frames into the
        # conversation (错误信息不泄敏感数据 / 不向用户暴露程序栈).
        logger.warning(
            "tool %r execution failed: %s\n%s",
            name,
            exc.original,
            exc.traceback_summary,
        )
        return _error_tool_message(
            f"[tool-error] 工具 '{name}' 执行失败：{exc.original}",  # noqa: RUF001
            tc_id,
            name,
        )

    # Lift a tool's structured soft-block signal onto the message so the SSE layer
    # can flag it (a captcha/拦截 page returns HTTP 200, so it's not an error).
    additional: dict[str, Any] = {}
    if isinstance(result, dict) and result.get("blocked"):
        additional["blocked"] = True
        reason = result.get("block_reason")
        if reason:
            additional["block_reason"] = str(reason)

    return ToolMessage(
        content=str(result),
        tool_call_id=tc_id,
        name=name,
        additional_kwargs=additional,
    )


async def tool_executor_node(
    state: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Execute tool_calls from the last assistant message via ToolRegistry.

    Reads the last message's ``tool_calls``, looks each up in the
    ``ToolRegistry`` passed via ``config["configurable"]["tool_registry"]``,
    executes them with recovery (_recover_tool_call), and appends
    ``ToolMessage`` instances to the conversation.

    A high-risk action confirmation gate (PRD §9.1) is consulted when a
    ``risk_gate`` is supplied via config: a high-risk call that is not in
    ``confirmed_tool_calls`` is returned as ``[confirmation-required]`` and not
    executed. Absent a gate, behaviour is unchanged.

    A per-agent permission policy is consulted when a ``tool_policy`` is
    supplied via config: a ``deny`` decision is returned as
    ``[policy-denied]`` without executing; a ``confirm`` decision requires the
    call id in ``confirmed_tool_calls`` (same flow as the risk gate). Absent a
    policy, behaviour is unchanged.
    """
    configurable = config.get("configurable", {})
    registry: ToolRegistry | None = configurable.get("tool_registry")
    if registry is None:
        return {}

    tcs = _last_assistant_tool_calls(state)
    if not tcs:
        return {}

    risk_gate = configurable.get("risk_gate")
    tool_policy = configurable.get("tool_policy")
    confirmed = configurable.get("confirmed_tool_calls") or frozenset()

    tool_messages: list[ToolMessage] = []
    for tc in tcs:
        sanitized = _sanitize_tool_call(tc)
        if sanitized is None:
            tool_messages.append(
                _error_tool_message(
                    "[tool-error] Malformed tool_call structure.",
                    tc.get("id", ""),
                    "unknown",
                )
            )
            continue

        if tool_policy is not None:
            name = sanitized["function"]["name"]
            decision = tool_policy.decide(name)
            if decision == "deny":
                tool_messages.append(
                    _error_tool_message(
                        f"[policy-denied] Tool '{name}' is not permitted "
                        "for this agent. Use one of the allowed tools instead.",
                        sanitized["id"],
                        name,
                    )
                )
                continue
            if decision == "confirm" and sanitized["id"] not in confirmed:
                tool_messages.append(
                    ToolMessage(
                        content=(
                            f"[confirmation-required] '{name}' requires explicit "
                            "user confirmation by this agent's tool policy."
                        ),
                        tool_call_id=sanitized["id"],
                        name=name,
                    )
                )
                continue

        if risk_gate is not None:
            name = sanitized["function"]["name"]
            args_raw = sanitized["function"]["arguments"]
            if risk_gate.requires_confirmation(
                name, args_raw, tc_id=sanitized["id"], confirmed_ids=confirmed
            ):
                tool_messages.append(
                    ToolMessage(
                        content=(
                            f"[confirmation-required] '{name}' is a high-risk operation "
                            "and needs explicit user confirmation before it runs."
                        ),
                        tool_call_id=sanitized["id"],
                        name=name,
                    )
                )
                continue

        msg = await _recover_tool_call(sanitized, registry)
        tool_messages.append(msg)

    return {"messages": tool_messages}
