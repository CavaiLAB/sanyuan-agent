"""Context-window trimming — bound the message list sent to the LLM.

The agent loop appends messages on every tool round-trip, so a long
conversation's ``state["messages"]`` grows without bound and eventually
overflows the model's context. ``trim_messages`` keeps the leading system
messages and the first task turn, drops the middle, and keeps the most
recent messages within a token budget — replacing the elided span with a
single marker message and never starting the kept tail on an orphaned tool
result (which would break OpenAI tool-call pairing).
"""

from __future__ import annotations

from typing import Any

_CHARS_PER_TOKEN = 4
_PER_MESSAGE_OVERHEAD = 4  # rough allowance for role/formatting tokens


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) — good enough for budgeting."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _role(msg: Any) -> str:
    if isinstance(msg, dict):
        return str(msg.get("role", ""))
    # langchain ToolMessage carries type="tool"
    return str(getattr(msg, "type", getattr(msg, "role", "")))


def _content(msg: Any) -> str:
    content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
    return content if isinstance(content, str) else str(content)


def _msg_tokens(msg: Any) -> int:
    return estimate_tokens(_content(msg)) + _PER_MESSAGE_OVERHEAD


def trim_messages(
    messages: list[Any],
    *,
    token_budget: int = 24000,
    keep_recent: int = 4,
) -> list[Any]:
    """Return *messages* bounded to roughly *token_budget* tokens.

    Under budget the list is returned unchanged. Over budget the result is
    ``[system…, first task turn, marker, recent tail]`` where the tail is the
    longest suffix fitting the remaining budget (but at least *keep_recent*
    messages) and never begins with a tool-result message.
    """
    if sum(_msg_tokens(m) for m in messages) <= token_budget:
        return messages

    head: list[Any] = []
    i = 0
    while i < len(messages) and _role(messages[i]) == "system":
        head.append(messages[i])
        i += 1
    if i < len(messages):
        head.append(messages[i])  # the first non-system message is the task
        i += 1

    body = messages[i:]
    if len(body) <= keep_recent:
        return messages

    budget = token_budget - sum(_msg_tokens(m) for m in head)
    start = len(body)
    spent = 0
    while start > 0:
        cost = _msg_tokens(body[start - 1])
        if spent + cost > budget and (len(body) - start) >= keep_recent:
            break
        spent += cost
        start -= 1

    # Never start the kept tail on an orphaned tool result.
    while start < len(body) and _role(body[start]) == "tool":
        start += 1

    if start <= 0 or start >= len(body):
        return messages  # nothing sensible to elide

    marker = {
        "role": "system",
        "content": f"[context trimmed: {start} earlier messages elided]",
    }
    return [*head, marker, *body[start:]]
