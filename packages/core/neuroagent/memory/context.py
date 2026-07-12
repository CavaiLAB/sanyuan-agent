"""Memory context primitives plus context-window trimming.

The agent loop appends messages on every tool round-trip, so a long
conversation's ``state["messages"]`` grows without bound and eventually
overflows the model's context. ``trim_messages`` keeps the leading system
messages and the first task turn, drops the middle, and keeps the most
recent messages within a token budget — replacing the elided span with a
single marker message and never starting the kept tail on an orphaned tool
result (which would break OpenAI tool-call pairing).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

_CHARS_PER_TOKEN = 4
_PER_MESSAGE_OVERHEAD = 4  # rough allowance for role/formatting tokens
# Fixed token charge for one inlined image (or other structured) content part,
# so a base64 data: URI is never counted as ~len/4 text tokens (which would blow
# the trim budget). Deliberately modest — real vision cost is model-specific.
_IMAGE_PART_TOKENS = 512


class MemoryLevel(StrEnum):
    """Context level for a durable memory, orthogonal to access scope."""

    PERSONAL = "personal"
    PROJECT = "project"
    SESSION = "session"


@dataclass(frozen=True, slots=True)
class MemoryContext:
    """Project/session context bound to one agent run."""

    project_id: str | None = None
    session_id: str | None = None


def validate_memory_context(
    level: MemoryLevel | str,
    *,
    project_id: str | None = None,
    session_id: str | None = None,
) -> MemoryLevel:
    """Validate a level/context tuple and return the normalized enum."""
    normalized = MemoryLevel(level)
    if normalized is MemoryLevel.PERSONAL and (project_id or session_id):
        raise ValueError("personal memory cannot carry project_id or session_id")
    if normalized is MemoryLevel.PROJECT and (not project_id or session_id):
        raise ValueError("project memory requires project_id and no session_id")
    if normalized is MemoryLevel.SESSION and not session_id:
        raise ValueError("session memory requires session_id")
    return normalized


def memory_visible_in_context(
    level: MemoryLevel | str,
    *,
    memory_project_id: str | None = None,
    memory_session_id: str | None = None,
    context: MemoryContext | None = None,
) -> bool:
    """Return whether a record belongs to the run's ancestor context chain."""
    normalized = validate_memory_context(
        level,
        project_id=memory_project_id,
        session_id=memory_session_id,
    )
    bound = context or MemoryContext()
    if normalized is MemoryLevel.PERSONAL:
        return True
    if normalized is MemoryLevel.PROJECT:
        return bool(bound.project_id) and memory_project_id == bound.project_id
    if not bound.session_id or memory_session_id != bound.session_id:
        return False
    return memory_project_id is None or memory_project_id == bound.project_id


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) — good enough for budgeting."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _role(msg: Any) -> str:
    if isinstance(msg, dict):
        return str(msg.get("role", ""))
    # langchain ToolMessage carries type="tool"
    return str(getattr(msg, "type", getattr(msg, "role", "")))


def _content_tokens(content: Any) -> int:
    """Estimate tokens for a message ``content`` (string or multimodal part list).

    A multimodal user turn carries a list of content parts — text parts plus
    ``image_url`` parts whose ``url`` is a base64 ``data:`` URI. Counting that
    base64 as text (``str(content)`` → ~len/4 tokens) would estimate a single
    inlined image at tens of thousands of tokens and collapse ``trim_messages``
    to its ``keep_recent`` floor. Instead count text parts normally and charge
    each image/structured part a small fixed cost — the real vision token cost is
    model-specific, but it is nowhere near the base64 length.
    """
    if isinstance(content, str):
        return estimate_tokens(content)
    if isinstance(content, list):
        total = 0
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                total += estimate_tokens(str(part.get("text", "")))
            elif isinstance(part, dict):
                # image_url / other structured parts: fixed cost, never the base64.
                total += _IMAGE_PART_TOKENS
            else:
                total += estimate_tokens(str(part))
        return total
    return estimate_tokens(str(content))


def _msg_tokens(msg: Any) -> int:
    # tool_calls arguments (file contents, large payloads) can dwarf content.
    tool_calls = (
        msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)
    )
    extra = 0
    if tool_calls:
        extra = estimate_tokens(json.dumps(tool_calls, ensure_ascii=False, default=str))
    content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
    return _content_tokens(content) + extra + _PER_MESSAGE_OVERHEAD


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

    # Never start the kept tail on an orphaned tool result: walk back to the
    # assistant turn that issued it, so the pair stays intact. (Walking back
    # can only grow the tail — the plan node runs right after a tool round,
    # so the list typically *ends* in tool results; skipping forward instead
    # would run off the end and silently abandon trimming.)
    while 0 < start < len(body) and _role(body[start]) == "tool":
        start -= 1

    if start <= 0 or start >= len(body):
        return messages  # nothing sensible to elide

    marker = {
        "role": "system",
        "content": f"[context trimmed: {start} earlier messages elided]",
    }
    return [*head, marker, *body[start:]]
