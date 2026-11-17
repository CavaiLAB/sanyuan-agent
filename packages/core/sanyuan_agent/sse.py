"""Server-Sent Events streaming helper for the standalone agent runtime."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from typing import Any

from langchain_core.messages import ToolMessage

# ── Agent SSE event schema ───────────────────────────────────────────────────
# Single source of truth for the wire protocol the web/mobile clients consume.
# Mirrored on the frontend in apps/web/src/lib/agentEvents.ts — keep both in sync.
#
#   step          {content}                              short user-facing progress
#                                                         narration emitted BEFORE a
#                                                         tool call ("先搜索最新信息")
#   tool_call     {id, name, arguments}                  a structured tool invocation
#   tool_result   {tool_call_id, content, is_error,      the tool's observation; is_error
#                  blocked, block_reason?, artifacts?}    / blocked carry the outcome so the
#                                                         UI renders red / 已拦截 vs success
#   execution     {target, provider, sandboxId, workdir} where this run's tools run —
#                                                         honest sandbox evidence, emitted
#                                                         once up front when sandboxed
#   artifact      {id, filename, url, contentType,        a file the agent PRODUCED (e.g. a
#                  sizeBytes, sha256, origin, sandboxId}  bash output harvested from the
#                                                         sandbox), surfaced before `done`
#   message       {delta}                                incremental final-answer tokens
#                                                         (same frame ask-mode streams;
#                                                         clients accumulate then let the
#                                                         terminal final_answer win)
#   tool_progress {name, chars, phase, filename?,        a tool turn is generating a large
#                  contentChars?}
#                                                         argument payload (e.g. a whole file
#                                                         for artifact_write); chars is the
#                                                         CUMULATIVE argument length so far —
#                                                         liveness paint only, superseded by
#                                                         the eventual tool_call frame
#   final_answer  {content}                              the assistant's final answer
#                                                         (a turn with NO tool calls)
#   error         {message}                              a stream-level failure
#   done          {usage, contextUsage?}                 terminal aggregate billing usage
#                                                         plus the LAST model call's
#                                                         context-window snapshot
#
# Invariant: a single plan turn is EITHER a tool step (optional `step` + one or
# more `tool_call`) OR a `final_answer` — never both. This keeps progress
# narration and the final answer cleanly separated on the wire. `message`
# deltas are additive live rendering only: exactly one final_answer still
# terminates the run and its content equals the deltas' concatenation.
EVENT_STEP = "step"
EVENT_TOOL_CALL = "tool_call"
EVENT_TOOL_RESULT = "tool_result"
EVENT_EXECUTION = "execution"
EVENT_ARTIFACT = "artifact"
EVENT_FINAL_ANSWER = "final_answer"
EVENT_MESSAGE = "message"
EVENT_TOOL_PROGRESS = "tool_progress"
EVENT_ERROR = "error"
EVENT_DONE = "done"


def _build_tool_result_payload(
    tool_call_id: str,
    content: Any,
    *,
    is_error: bool = False,
    blocked: bool = False,
    block_reason: str | None = None,
) -> dict[str, Any]:
    """Build a tool_result event payload, extracting artifacts if present.

    ``is_error`` carries the tool's outcome so the UI renders a failed tool as
    failed (red) instead of a green success — see ToolCallCard / messageStore.

    ``blocked`` flags an anti-bot / captcha "soft block" page: the request
    succeeded (HTTP 200, ``is_error`` false) but the content is a 拦截/验证 wall,
    so the UI shows an honest "已拦截" instead of success. ``block_reason`` is an
    optional short human-readable explanation.
    """
    payload: dict[str, Any] = {
        "tool_call_id": tool_call_id,
        "content": str(content),
        "is_error": is_error,
        "blocked": blocked,
    }
    if block_reason:
        payload["block_reason"] = block_reason
    try:
        parsed = json.loads(content) if isinstance(content, str) else content
        if isinstance(parsed, dict) and "artifact_id" in parsed:
            payload["artifacts"] = [
                {
                    "id": parsed["artifact_id"],
                    "filename": parsed.get("filename", ""),
                    "url": parsed.get("download_url", ""),
                }
            ]
    except (json.JSONDecodeError, TypeError):
        pass
    return payload


def _extract_message_data(msg: Any) -> dict[str, Any]:
    """Normalise a message (dict or LangChain Message) into a plain dict."""
    if isinstance(msg, dict):
        return msg
    # LangChain message object
    extra = getattr(msg, "additional_kwargs", None) or {}
    return {
        "role": getattr(msg, "type", getattr(msg, "role", "")),
        "content": getattr(msg, "content", ""),
        "tool_calls": getattr(msg, "tool_calls", None),
        "tool_call_id": getattr(msg, "tool_call_id", None),
        "name": getattr(msg, "name", None),
        "status": getattr(msg, "status", None),
        "blocked": extra.get("blocked"),
        "block_reason": extra.get("block_reason"),
    }


def _format_sse_event(event_type: str, data: dict[str, Any]) -> str:
    """Format a single SSE event frame."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


async def sse_stream(
    agent: Any,
    state: dict[str, Any],
    *,
    heartbeat_interval: float = 15.0,
    post_agent: Callable[[], AsyncIterator[str]] | None = None,
    context_window_tokens: int | None = None,
) -> AsyncIterator[str]:
    """Stream agent execution as SSE events.

    Yields frames like:
        event: step\ndata: {"content": "..."}\n\n
        event: tool_call\ndata: {"id": "...", "name": "..."}\n\n
        event: tool_result\ndata: {"tool_call_id": "...", "content": "..."}\n\n
        event: final_answer\ndata: {"content": "..."}\n\n
        event: error\ndata: {"message": "..."}\n\n
    Heartbeat ``: ping\n\n`` is emitted every *heartbeat_interval* seconds.

    *post_agent* is an optional async-frame source drained AFTER the agent loop
    finishes but BEFORE the terminal ``done`` — the seam where the gateway emits
    artifacts harvested from the sandbox (so they arrive while the stream is
    still open, ahead of ``done``). Each yielded value must already be a
    formatted SSE frame. A failure inside it surfaces as a single ``error``
    frame and never blocks ``done``.
    """
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def _producer() -> None:
        # Accumulate real token usage across reasoning turns. astream(updates)
        # yields each node's return (a per-turn delta), so we sum them here and
        # report the total in the terminal `done` event.
        usage_total = {"promptTokens": 0, "completionTokens": 0}
        # Aggregate usage is a billing/cost figure. It cannot drive a context
        # meter because every tool iteration re-sends much of the same history.
        # Keep the latest model call separately: that is the actual context
        # footprint when the run settles.
        context_usage: dict[str, int] | None = None
        # Opt in to live final-answer deltas — only this SSE layer wants them;
        # workflow workers / ainvoke callers keep non-streaming LLM calls.
        streamed_state = {**state, "stream_final": True}
        try:
            # "custom" carries the plan/finalize nodes' live final-answer deltas
            # (get_stream_writer); "updates" keeps every existing frame source.
            async for item in agent.astream(streamed_state, stream_mode=["updates", "custom"]):
                # With a list stream_mode langgraph yields (mode, chunk) tuples;
                # bare dicts (test doubles / older fakes) are treated as updates.
                if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
                    mode, chunk = item
                else:
                    mode, chunk = "updates", item

                if mode == "custom":
                    if (
                        isinstance(chunk, dict)
                        and chunk.get("kind") == "final_delta"
                        and chunk.get("content")
                    ):
                        await queue.put(
                            _format_sse_event(EVENT_MESSAGE, {"delta": chunk["content"]})
                        )
                    elif (
                        isinstance(chunk, dict)
                        and chunk.get("kind") == "tool_progress"
                        and chunk.get("name")
                    ):
                        await queue.put(
                            _format_sse_event(
                                EVENT_TOOL_PROGRESS,
                                {
                                    "name": chunk["name"],
                                    "chars": int(chunk.get("chars", 0) or 0),
                                    "phase": chunk.get("phase", "generating_tool_input"),
                                    **(
                                        {"filename": chunk["filename"]}
                                        if chunk.get("filename")
                                        else {}
                                    ),
                                    **(
                                        {"contentChars": int(chunk["contentChars"])}
                                        if chunk.get("contentChars") is not None
                                        else {}
                                    ),
                                },
                            )
                        )
                    continue

                for node_name, update in chunk.items():
                    if node_name == "plan":
                        turn_usage = update.get("usage")
                        if turn_usage:
                            usage_total["promptTokens"] += turn_usage.get("promptTokens", 0)
                            usage_total["completionTokens"] += turn_usage.get("completionTokens", 0)
                            context_usage = {
                                "promptTokens": int(turn_usage.get("promptTokens", 0) or 0),
                                "completionTokens": int(turn_usage.get("completionTokens", 0) or 0),
                            }
                        messages = update.get("messages", [])
                        if messages:
                            msg = _extract_message_data(messages[-1])
                            content = msg.get("content", "")
                            tool_calls = msg.get("tool_calls")

                            if tool_calls:
                                # A tool step. `content` is the optional progress
                                # narration the agent wrote before acting; emit it
                                # as a `step` only when non-empty — and never also
                                # as a final_answer (that double-emit used to leak
                                # the model's text into the trace as a narration).
                                if isinstance(content, str) and content.strip():
                                    await queue.put(
                                        _format_sse_event(EVENT_STEP, {"content": content})
                                    )
                                for tc in tool_calls:
                                    if isinstance(tc, dict):
                                        fn = tc.get("function", {})
                                        tc_data = {
                                            "id": tc.get("id", ""),
                                            "name": fn.get("name", ""),
                                            "arguments": fn.get("arguments", ""),
                                        }
                                    else:
                                        fn = getattr(tc, "function", None)
                                        tc_data = {
                                            "id": getattr(tc, "id", ""),
                                            "name": getattr(fn, "name", ""),
                                            "arguments": getattr(fn, "arguments", ""),
                                        }
                                    await queue.put(_format_sse_event(EVENT_TOOL_CALL, tc_data))
                            else:
                                # No tool calls → this turn IS the final answer.
                                # Emit it once, as final_answer (not a step).
                                await queue.put(
                                    _format_sse_event(EVENT_FINAL_ANSWER, {"content": content})
                                )

                    elif node_name == "finalize":
                        # Forced wrap-up turn (iteration cap hit): always a plain
                        # assistant answer with no tool calls. Emit it as the
                        # final_answer so the body is never empty.
                        turn_usage = update.get("usage")
                        if turn_usage:
                            usage_total["promptTokens"] += turn_usage.get("promptTokens", 0)
                            usage_total["completionTokens"] += turn_usage.get("completionTokens", 0)
                            context_usage = {
                                "promptTokens": int(turn_usage.get("promptTokens", 0) or 0),
                                "completionTokens": int(turn_usage.get("completionTokens", 0) or 0),
                            }
                        messages = update.get("messages", [])
                        if messages:
                            msg = _extract_message_data(messages[-1])
                            await queue.put(
                                _format_sse_event(
                                    EVENT_FINAL_ANSWER,
                                    {"content": msg.get("content", "")},
                                )
                            )

                    elif node_name == "tool_executor":
                        messages = update.get("messages", [])
                        for msg in messages:
                            if isinstance(msg, ToolMessage):
                                extra = getattr(msg, "additional_kwargs", None) or {}
                                payload = _build_tool_result_payload(
                                    msg.tool_call_id,
                                    msg.content,
                                    is_error=getattr(msg, "status", None) == "error",
                                    blocked=bool(extra.get("blocked")),
                                    block_reason=extra.get("block_reason"),
                                )
                                await queue.put(_format_sse_event(EVENT_TOOL_RESULT, payload))
                            else:
                                data = _extract_message_data(msg)
                                if data.get("role") == "tool" or data.get("tool_call_id"):
                                    payload = _build_tool_result_payload(
                                        data.get("tool_call_id", ""),
                                        data.get("content", ""),
                                        is_error=data.get("status") == "error",
                                        blocked=bool(data.get("blocked")),
                                        block_reason=data.get("block_reason"),
                                    )
                                    await queue.put(_format_sse_event(EVENT_TOOL_RESULT, payload))
            # Post-agent frames (e.g. artifacts harvested from the sandbox):
            # emitted while the stream is still open, ahead of `done`, so the
            # client renders produced files as part of the same turn. Isolated so
            # a harvest failure can't lose the terminal `done`.
            if post_agent is not None:
                try:
                    async for extra in post_agent():
                        await queue.put(extra)
                except Exception as exc:
                    await queue.put(_format_sse_event(EVENT_ERROR, {"message": str(exc)}))
            await queue.put(
                _format_sse_event(
                    EVENT_DONE,
                    {
                        "usage": {
                            **usage_total,
                            "totalTokens": usage_total["promptTokens"]
                            + usage_total["completionTokens"],
                        },
                        **(
                            {
                                "contextUsage": {
                                    **context_usage,
                                    "totalTokens": context_usage["promptTokens"]
                                    + context_usage["completionTokens"],
                                    **(
                                        {"contextWindowTokens": context_window_tokens}
                                        if context_window_tokens and context_window_tokens > 0
                                        else {}
                                    ),
                                }
                            }
                            if context_usage is not None
                            else {}
                        ),
                    },
                )
            )
            await queue.put(None)  # sentinel
        except Exception as exc:
            await queue.put(_format_sse_event(EVENT_ERROR, {"message": str(exc)}))
            await queue.put(None)

    async def _heartbeat() -> None:
        while True:
            await asyncio.sleep(heartbeat_interval)
            await queue.put(": ping\n\n")

    producer_task = asyncio.create_task(_producer())
    heartbeat_task = asyncio.create_task(_heartbeat())

    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item
    finally:
        producer_task.cancel()
        heartbeat_task.cancel()
        try:
            await producer_task
        except asyncio.CancelledError:
            pass
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
