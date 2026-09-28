"""Simplified LangGraph agent (plan → act → tool_executor → observe → plan)."""

import json
import logging
import os
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Annotated, Any, cast

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from neuroagent.agents.tool_executor import tool_executor_node
from neuroagent.memory.context import MemoryContext, trim_messages
from neuroagent.memory.episodic import EpisodicMemory
from neuroagent.memory.longterm.store import LongtermMemoryStore, ScopeKind
from neuroagent.memory.recall import EpisodicRecall
from neuroagent.memory.sensory import SensoryEvent
from neuroagent.memory.vector import VectorRecall
from neuroagent.memory.working import WorkingMemory
from neuroagent.models import LLMProvider
from neuroagent.security.tool_policy import ToolPolicy
from neuroagent.skills.registry import SkillRegistry
from neuroagent.tools.registry import ToolRegistry, callable_to_schema

logger = logging.getLogger(__name__)


def _merge_usage(a: dict[str, int] | None, b: dict[str, int] | None) -> dict[str, int]:
    """Reducer: sum token-usage deltas across reasoning turns (None-safe)."""
    a = a or {}
    b = b or {}
    return {
        "promptTokens": a.get("promptTokens", 0) + b.get("promptTokens", 0),
        "completionTokens": a.get("completionTokens", 0) + b.get("completionTokens", 0),
    }


class AgentState(TypedDict):
    """Shared state across the agent graph."""

    messages: Annotated[list[dict[str, Any]], list.__add__]
    next_action: str | None
    tool_results: Annotated[list[dict[str, Any]], list.__add__]
    iteration: int
    max_iter: int
    sensory: Any
    working: Any
    # Cumulative LLM token usage across the run (summed by _merge_usage).
    usage: Annotated[dict[str, int], _merge_usage]
    # Opt-in live final-answer streaming (message deltas over the langgraph
    # custom channel). ONLY the chat SSE layer sets this; workflow workers and
    # ainvoke callers leave it unset so their LLM calls stay non-streaming
    # (mocks, cost-callback timing and batch semantics unchanged).
    stream_final: bool


def _to_tool_def(tool: Callable[..., Any]) -> dict[str, Any]:
    """Convert a callable to an OpenAI-style tool definition.

    A tool that carries an explicit ``__tool_schema__`` (the ``function`` object)
    — e.g. an MCP-adapted tool whose JSON schema can't be reflected from a
    generic ``**kwargs`` wrapper — has it used verbatim. Otherwise the schema is
    reflected from the callable's signature via the single shared reflector
    :func:`~neuroagent.tools.registry.callable_to_schema` (no parallel copy).
    """
    explicit = getattr(tool, "__tool_schema__", None)
    if isinstance(explicit, dict):
        return {"type": "function", "function": explicit}
    return {"type": "function", "function": callable_to_schema(tool)}


# Map a LangChain message ``.type`` to its OpenAI ``role``.
_ROLE_BY_TYPE = {"human": "user", "ai": "assistant", "tool": "tool", "system": "system"}


def _to_openai_message(msg: Any) -> dict[str, Any]:
    """Coerce one state message into an OpenAI-compatible dict for litellm.

    Plain dicts pass through unchanged. LangChain message objects (notably the
    ``ToolMessage`` the tool executor appends) carry extra fields - a string
    ``status``, plus ``artifact`` / ``additional_kwargs`` / ``response_metadata``
    / ``id`` / ``type`` - that litellm's ``GPT3Message`` model rejects (a string
    ``status`` raises ``1 validation error for GPT3Message / status``). Map them
    to the minimal OpenAI shape (role / content / tool_call_id / name), dropping
    everything litellm does not expect.
    """
    if isinstance(msg, dict):
        return msg
    msg_type = getattr(msg, "type", "") or getattr(msg, "role", "")
    out: dict[str, Any] = {
        "role": _ROLE_BY_TYPE.get(msg_type, msg_type or "user"),
        "content": getattr(msg, "content", ""),
    }
    tool_call_id = getattr(msg, "tool_call_id", None)
    if tool_call_id:
        out["tool_call_id"] = tool_call_id
    name = getattr(msg, "name", None)
    if name:
        out["name"] = name
    return out


async def _stream_completion(
    litellm_mod: Any,
    model: str,
    base_kwargs: dict[str, Any],
    on_content_delta: Callable[[str], None],
    hold_chars: int,
    on_tool_progress: Callable[[str, int, dict[str, Any]], None] | None = None,
) -> Any:
    """Streaming variant of one completion: forward content deltas, return a
    response shaped like the non-streaming one (choices[0].message + usage).

    Emission gate — the caller can't know upfront whether this turn is the
    final answer or a tool turn, so:
    - a tool_call delta permanently stops emission (a tool turn's content is
      narration; it surfaces as a `step` frame, and double-emitting it here
      would duplicate text in the live body);
    - the first *hold_chars* of content are buffered before emitting — tool
      narration is typically short, so a turn that opens with a brief blurb and
      then calls tools emits nothing at all; a real final answer blows past the
      hold window almost immediately.
    Everything still accumulates into the returned message regardless of the
    gate, so the terminal final_answer frame always carries the full text.

    *on_tool_progress* is the inverse of the gate: once tool_call deltas start
    (and the body goes silent by design), it receives (tool_name, cumulative
    argument chars, structured detail) so the UI can show that a large tool
    payload — e.g. a whole HTML file for artifact_write — is being generated
    instead of dead air.
    Throttled to every _TOOL_PROGRESS_STEP chars; cumulative, so a dropped
    frame never corrupts the figure.
    """
    stream = await litellm_mod.acompletion(
        model=model,
        stream=True,
        # Usage arrives only on the last chunk; without this flag the streamed
        # call would report zero tokens and llm_call_costs would under-record.
        stream_options={"include_usage": True},
        **base_kwargs,
    )
    if not hasattr(stream, "__aiter__"):
        # Provider/proxy ignored stream=True and returned a complete response —
        # degrade gracefully: emit the final answer once (tool turns emit
        # nothing, matching the gate) and hand the response back unchanged.
        message = getattr(stream.choices[0], "message", None) if getattr(stream, "choices", None) else None
        content = getattr(message, "content", None) if message is not None else None
        if content and not getattr(message, "tool_calls", None):
            on_content_delta(content)
        return stream

    content_parts: list[str] = []
    tool_calls_by_index: dict[int, dict[str, str]] = {}
    usage_obj: Any = None
    buffered = ""
    flushed = False
    saw_tool_call = False
    progress_last = -1

    async for chunk in stream:
        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage is not None:
            usage_obj = chunk_usage
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        if delta is None:
            continue

        for tc in getattr(delta, "tool_calls", None) or []:
            saw_tool_call = True
            idx = getattr(tc, "index", 0) or 0
            slot = tool_calls_by_index.setdefault(
                idx, {"id": "", "type": "function", "name": "", "arguments": ""}
            )
            if getattr(tc, "id", None):
                slot["id"] = tc.id
            fn = getattr(tc, "function", None)
            if fn is not None:
                if getattr(fn, "name", None):
                    slot["name"] = fn.name
                # Arguments stream as JSON fragments — concatenate.
                slot["arguments"] += getattr(fn, "arguments", None) or ""

        if saw_tool_call and on_tool_progress is not None:
            first = tool_calls_by_index.get(min(tool_calls_by_index), {})
            name = first.get("name", "")
            total = sum(len(slot["arguments"]) for slot in tool_calls_by_index.values())
            if name and (progress_last < 0 or total - progress_last >= _TOOL_PROGRESS_STEP):
                progress_last = total
                arguments = first.get("arguments", "")
                on_tool_progress(name, total, _tool_progress_detail(name, arguments))

        piece = getattr(delta, "content", None)
        if piece:
            content_parts.append(piece)
            if not saw_tool_call:
                if flushed:
                    on_content_delta(piece)
                else:
                    buffered += piece
                    if len(buffered) >= hold_chars:
                        on_content_delta(buffered)
                        buffered = ""
                        flushed = True

    # A final answer shorter than the hold window still streams (once, whole).
    if not saw_tool_call and not flushed and buffered:
        on_content_delta(buffered)

    tool_calls = [
        SimpleNamespace(
            id=slot["id"],
            type=slot["type"],
            function=SimpleNamespace(name=slot["name"], arguments=slot["arguments"]),
        )
        for _, slot in sorted(tool_calls_by_index.items())
    ] or None
    message = SimpleNamespace(
        role="assistant", content="".join(content_parts), tool_calls=tool_calls
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage_obj)


async def _call_llm(
    provider: LLMProvider,
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]],
    metadata: dict[str, str] | None = None,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    on_content_delta: Callable[[str], None] | None = None,
    hold_chars: int = 0,
    on_tool_progress: Callable[[str, int, dict[str, Any]], None] | None = None,
) -> Any:
    """Unified async LLM call via LiteLLM.

    *metadata* (run_id / tenant_id) rides along in litellm's metadata so the
    usage success-callback can attribute the call without global state.

    *temperature* / *max_tokens* are passed through only when set, so callers
    that omit them keep litellm's defaults (no behaviour change).

    *on_content_delta* switches the call to streaming: content deltas are
    forwarded to it (gated — see ``_stream_completion``) and the return value
    keeps the non-streaming response shape. ``None`` keeps today's plain call.
    """
    # Normalize any LangChain message objects (e.g. ToolMessage) into OpenAI
    # dicts so litellm's GPT3Message model never sees a leaked string ``status``.
    base_kwargs: dict[str, Any] = {
        "messages": [_to_openai_message(m) for m in messages]
    }
    if provider.api_key_env:
        base_kwargs["api_key"] = os.environ.get(provider.api_key_env)
    if provider.api_base:
        base_kwargs["api_base"] = provider.api_base
    if tools:
        base_kwargs["tools"] = tools
    if metadata:
        base_kwargs["metadata"] = metadata
    if temperature is not None:
        base_kwargs["temperature"] = temperature
    if max_tokens is not None:
        base_kwargs["max_tokens"] = max_tokens

    # Delay import so tests can patch *before* the module is imported.
    import litellm

    # Transient failures worth retrying on another model. Bad-request / auth
    # errors are excluded — a different model won't fix a malformed request.
    fallback_exceptions = (
        litellm.RateLimitError,
        litellm.APIConnectionError,
        litellm.Timeout,
        litellm.ServiceUnavailableError,
        litellm.InternalServerError,
    )

    # Streamed calls must not retry once tokens reached the client — a fallback
    # would re-emit the answer's head. Track first emission across attempts.
    emitted = False

    def _guarded_delta(piece: str) -> None:
        nonlocal emitted
        emitted = True
        assert on_content_delta is not None
        on_content_delta(piece)

    # Try the primary model, then each fallback in order on a *transient*
    # failure. Non-transient errors bubble immediately.
    model_chain = (provider.default_model, *provider.fallback_models)
    last_exc: Exception | None = None
    for i, model in enumerate(model_chain):
        try:
            if on_content_delta is not None:
                return await _stream_completion(
                    litellm,
                    model,
                    base_kwargs,
                    _guarded_delta,
                    hold_chars,
                    on_tool_progress=on_tool_progress,
                )
            return await litellm.acompletion(model=model, **base_kwargs)
        except fallback_exceptions as exc:
            if emitted:
                raise
            last_exc = exc
            remaining = model_chain[i + 1 :]
            if remaining:
                logger.warning(
                    "model %r failed (%s); falling back to %r",
                    model,
                    type(exc).__name__,
                    remaining[0],
                )
    assert last_exc is not None  # only reachable after a caught failure
    raise last_exc


# Hold window for streamed final-answer deltas: tool narration ("我先查一下…")
# is nearly always shorter than this, so a narration-then-tools turn emits
# nothing; a real final answer crosses it within its first sentence or two.
_FINAL_STREAM_HOLD_CHARS = 160

# Emit a tool-argument progress signal every this many streamed argument chars.
# Coarse on purpose: it's a liveness indicator, not a byte counter.
_TOOL_PROGRESS_STEP = 512


def _decode_partial_json_string(arguments: str, field: str) -> str | None:
    """Decode a complete JSON string field from an otherwise partial object."""
    marker = f'"{field}"'
    start = arguments.find(marker)
    if start < 0:
        return None
    colon = arguments.find(":", start + len(marker))
    if colon < 0:
        return None
    fragment = arguments[colon + 1 :].lstrip()
    try:
        value, _ = json.JSONDecoder().raw_decode(fragment)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, str) else None


def _partial_json_string_chars(arguments: str, field: str) -> int | None:
    """Approximate generated chars for a JSON string that may still be open."""
    decoded = _decode_partial_json_string(arguments, field)
    if decoded is not None:
        return len(decoded)
    marker = f'"{field}"'
    start = arguments.find(marker)
    if start < 0:
        return None
    colon = arguments.find(":", start + len(marker))
    if colon < 0:
        return None
    fragment = arguments[colon + 1 :].lstrip()
    if not fragment.startswith('"'):
        return None
    # This is intentionally approximate: escaped JSON characters count as more
    # than one byte on the wire.  The UI labels it as generated payload size,
    # not the final artifact byte size.
    return max(0, len(fragment) - 1)


def _tool_progress_detail(name: str, arguments: str) -> dict[str, Any]:
    """Return user-facing phase metadata without exposing argument contents."""
    if name != "artifact_write":
        return {"phase": "generating_tool_input"}
    detail: dict[str, Any] = {"phase": "generating_artifact"}
    filename = _decode_partial_json_string(arguments, "filename")
    if filename:
        detail["filename"] = filename
    content_chars = _partial_json_string_chars(arguments, "content")
    if content_chars is not None:
        detail["contentChars"] = content_chars
    return detail


def _maybe_stream_writer() -> Callable[[dict[str, Any]], None] | None:
    """LangGraph custom-stream writer, or None outside a streaming context
    (ainvoke callers, older langgraph) — callers then take the non-stream path."""
    try:
        from langgraph.config import get_stream_writer

        return get_stream_writer()
    except Exception:  # pragma: no cover - depends on runtime context
        return None


def _final_delta_sink(
    writer: Callable[[dict[str, Any]], None] | None,
) -> Callable[[str], None] | None:
    """Adapt the graph writer into the ``on_content_delta`` callback shape."""
    if writer is None:
        return None

    def _sink(piece: str) -> None:
        writer({"kind": "final_delta", "content": piece})

    return _sink


def _tool_progress_sink(
    writer: Callable[[dict[str, Any]], None] | None,
) -> Callable[[str, int, dict[str, Any] | None], None] | None:
    """Adapt the graph writer into the ``on_tool_progress`` callback shape."""
    if writer is None:
        return None

    def _sink(name: str, chars: int, detail: dict[str, Any] | None = None) -> None:
        writer({"kind": "tool_progress", "name": name, "chars": chars, **(detail or {})})

    return _sink


def _get_content(response: Any) -> str:
    try:
        return response.choices[0].message.content or ""
    except (AttributeError, IndexError):
        return ""


def _get_usage(response: Any) -> dict[str, int]:
    """Extract this call's token usage from a litellm response (0s if absent)."""
    usage = getattr(response, "usage", None)
    return {
        "promptTokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completionTokens": int(getattr(usage, "completion_tokens", 0) or 0),
    }


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


# A per-turn reminder to narrate before acting. The capability prompt carries the
# same rule, but a single instruction at the TOP loses salience as the context
# grows with tool results — GLM (and peers) then fire tools with empty content,
# so the trace becomes a bare stack of tool rows with no "why". Re-asserting it as
# the LAST message before every tool-deciding call keeps it salient each turn
# (verified against the live model: recovers turns that otherwise drop narration).
# It is NOT added to persistent state — only to the per-call message list — so it
# never accumulates across turns.
_NARRATION_REMINDER = (
    "Reminder: if you call a tool now, first write ONE short sentence in your "
    "message content (in the user's language) saying what you're about to do next "
    "and why — then make the call. Do not leave content empty when calling a tool. "
    "Keep it to a sentence; do not restate the whole plan or expose your full "
    "chain-of-thought."
)


def _inject_narration_reminder(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Append the per-turn narrate-before-acting reminder as the final message."""
    reminder: dict[str, Any] = {"role": "system", "content": _NARRATION_REMINDER}
    return [*list(messages), reminder]


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
    """Read the current user turn, excluding non-text multimodal payloads."""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "\n".join(
                    part["text"]
                    for part in content
                    if isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text"), str)
                )
            return ""
    return ""


async def _maybe_inject_recall(
    messages: list[dict[str, Any]],
    recall: EpisodicRecall | None,
    tenant_id: str,
    conversation_id: str,
    *,
    user_query: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch relevant episodes via recall and inject them as a system message.

    Failures are silently swallowed — episodic recall is a best-effort
    optimisation, not a hard dependency.
    """
    if recall is None or not tenant_id:
        return messages

    if user_query is None:
        user_query = _extract_user_query(messages)
    if not user_query.strip():
        return messages

    try:
        episodes = await recall.recall(
            tenant_id=tenant_id,
            query=user_query,
            conversation_id=conversation_id or None,
        )
        if not episodes:
            return messages
        block = recall.format_prompt_block(episodes)
        if not block:
            return messages
        system_msg: dict[str, Any] = {
            "role": "system",
            "content": block + "\n\nUse the above as background context if relevant.",
        }
        return [system_msg, *list(messages)]
    except Exception as exc:
        logger.warning("episodic recall failed: %s", exc)
        return messages


async def _maybe_inject_longterm(
    messages: list[dict[str, Any]],
    longterm: LongtermMemoryStore | None,
    tenant_id: str,
    user_id: str,
    project_id: str | None = None,
    session_id: str | None = None,
    *,
    user_query: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch relevant long-term memories and inject as a system message."""
    if longterm is None or not tenant_id:
        return messages

    if user_query is None:
        user_query = _extract_user_query(messages)
    if not user_query.strip():
        return messages

    try:
        entries = await longterm.retrieve_semantic(
            tenant_id=tenant_id,
            query=user_query,
            scope=[ScopeKind.PRIVATE, ScopeKind.TEAM, ScopeKind.TENANT],
            scope_owner_id=user_id or None,
            limit=5,
            importance_threshold=0.3,
            context=MemoryContext(project_id=project_id, session_id=session_id),
        )
        if not entries:
            return messages
        lines: list[str] = []
        for e in entries:
            lines.append(
                f"- [{e.scope.value}] {e.content} (sim={e.similarity_score:.2f})"
            )
        system_msg: dict[str, Any] = {
            "role": "system",
            "content": "Relevant memories:\n"
            + "\n".join(lines)
            + "\n\nUse the above as background context if relevant.",
        }
        return [system_msg, *list(messages)]
    except Exception as exc:
        logger.warning("longterm recall failed: %s", exc)
        return messages


def _maybe_inject_skill(
    messages: list[dict[str, Any]],
    skill_registry: SkillRegistry | None,
    *,
    user_query: str | None = None,
) -> list[dict[str, Any]]:
    """Check user message for skill trigger keywords and inject skill prompt."""
    if skill_registry is None:
        return messages

    if user_query is None:
        user_query = _extract_user_query(messages)
    if not user_query.strip():
        return messages

    skill = skill_registry.match_trigger(user_query)
    if skill is None:
        return messages

    prompt = skill_registry.load_prompt(skill)
    if not prompt:
        return messages

    system_msg: dict[str, Any] = {
        "role": "system",
        "content": f"[Skill: {skill.name}] {prompt}",
    }
    logger.info("Skill triggered: %s (keyword match in: %s)", skill.name, user_query[:60])
    return [system_msg, *list(messages)]


async def _maybe_save_episode(
    state: AgentState,
    episodic: EpisodicMemory | None,
    tenant_id: str,
    conversation_id: str,
    session_started_at: float,
    *,
    final_answer: str,
) -> None:
    """Persist the completed turn to episodic memory.

    The terminal node supplies its new answer explicitly because its state
    still ends at the preceding turn. Failures do not block the agent flow.
    """
    if episodic is None or not tenant_id or not conversation_id:
        return

    messages: list[Any] = state.get("messages", [])
    user_query = _extract_user_query(messages)

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
    temperature: float | None = None,
    max_tokens: int | None = None,
    episodic_memory: EpisodicMemory | None = None,
    episodic_recall: EpisodicRecall | None = None,
    longterm_memory: LongtermMemoryStore | None = None,
    skill_registry: SkillRegistry | None = None,
    tool_policy: ToolPolicy | None = None,
    context_token_budget: int = 24000,
    tenant_id: str = "",
    conversation_id: str = "",
    project_id: str = "",
    user_id: str = "",
    run_id: str = "",
) -> Any:
    """Build and compile a plan/act/tool_executor/observe agent graph.

    The recall backend is controlled via the ``NEUROAGENT_MEMORY_BACKEND``
    environment variable:

    - ``"vector"``  — use :class:`VectorRecall` (sentence-transformers).
    - ``"keyword"`` — use :class:`EpisodicRecall` with TF-IDF.
    - ``"none"``    — disable recall entirely.

    If *episodic_recall* is already provided it takes precedence.

    *longterm_memory* enables semantic retrieval from the pgvector-backed
    long-term store. Results are restricted to the personal → matching project
    → matching conversation context chain and injected into the plan prompt.
    """
    registry = ToolRegistry()

    # Build recall from env if not explicitly provided
    effective_recall = episodic_recall
    if effective_recall is None and episodic_memory is not None:
        backend = os.environ.get("NEUROAGENT_MEMORY_BACKEND", "keyword").lower()
        if backend == "vector":
            effective_recall = VectorRecall(store=episodic_memory)
        elif backend == "keyword":
            effective_recall = EpisodicRecall(store=episodic_memory, strategy="keyword")
        # "none" or unknown → leave as None
    for t in tools:
        schema = _to_tool_def(t)["function"]
        registry.register(t.__name__, t, schema)

    session_started_at = time.time()

    # Per-call usage attribution for the litellm success-callback.
    llm_metadata = {
        k: v for k, v in (("run_id", run_id), ("tenant_id", tenant_id)) if v
    } or None

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

        # Choose retrieval context independently of the context-window budget.
        user_query = _extract_user_query(state["messages"])
        bounded = trim_messages(state["messages"], token_budget=context_token_budget)
        llm_messages = _inject_working_summary(bounded, working)
        llm_messages = _maybe_inject_skill(
            llm_messages, skill_registry, user_query=user_query
        )
        llm_messages = await _maybe_inject_recall(
            llm_messages,
            effective_recall,
            tenant_id,
            conversation_id,
            user_query=user_query,
        )
        llm_messages = await _maybe_inject_longterm(
            llm_messages,
            longterm_memory,
            tenant_id,
            user_id,
            project_id or None,
            conversation_id or None,
            user_query=user_query,
        )
        # Last, most-salient instruction: narrate before acting, this turn.
        llm_messages = _inject_narration_reminder(llm_messages)

        response = await _call_llm(
            llm_provider,
            llm_messages,
            registry.tool_defs(),
            metadata=llm_metadata,
            temperature=temperature,
            max_tokens=max_tokens,
            # Stream this turn's content as it generates: if it turns out to be
            # the final answer the client renders it live; a tool turn's
            # narration is held back by the gate (see _stream_completion).
            on_content_delta=(
                _final_delta_sink(_maybe_stream_writer())
                if state.get("stream_final")
                else None
            ),
            hold_chars=_FINAL_STREAM_HOLD_CHARS,
            # The gate's counterpart: while a tool turn silently generates a
            # large argument payload, surface (tool name, cumulative chars).
            on_tool_progress=(
                _tool_progress_sink(_maybe_stream_writer())
                if state.get("stream_final")
                else None
            ),
        )
        usage = _get_usage(response)
        tcs = _get_tool_calls(response)
        if tcs:
            msg: dict[str, Any] = {
                "role": "assistant",
                "content": _get_content(response),
                "tool_calls": tcs,
            }
            return {"next_action": "tool_call", "messages": [msg], "usage": usage}
        msg = {"role": "assistant", "content": _get_content(response)}

        # Session ends here — save episode asynchronously (best-effort)
        await _maybe_save_episode(
            state, episodic_memory, tenant_id, conversation_id, session_started_at,
            final_answer=msg["content"],
        )
        return {"next_action": "stop", "messages": [msg], "usage": usage}

    async def finalize(state: AgentState) -> dict[str, Any]:
        """Forced wrap-up when the loop is cut off at the iteration cap.

        ``should_continue`` routes here (instead of ``END``) when the agent still
        wanted to call a tool but hit ``max_iter``. Left alone the run would end
        on a tool-call turn with no assistant prose — an empty body the client
        drops on reload (the user sees "no final answer", and the conversation
        looks lost). Make one more LLM call *without tools* so the model must
        synthesize a textual answer from what it already gathered.
        """
        working = state.get("working")
        bounded = trim_messages(
            state["messages"], token_budget=context_token_budget
        )
        llm_messages = _inject_working_summary(bounded, working)
        llm_messages = [
            *llm_messages,
            {
                "role": "user",
                "content": (
                    "已达到本次任务的工具调用步数上限，请不要再请求任何工具。"  # noqa: RUF001
                    "请直接基于以上已经获取到的信息，给出尽可能完整的最终答复；"  # noqa: RUF001
                    "如信息不足，请说明已知部分与缺口。"  # noqa: RUF001
                ),
            },
        ]
        response = await _call_llm(
            llm_provider,
            llm_messages,
            [],  # no tools — force a textual final answer
            metadata=llm_metadata,
            temperature=temperature,
            max_tokens=max_tokens,
            # Toolless by construction → guaranteed terminal: stream every
            # delta immediately, no hold window needed.
            on_content_delta=(
                _final_delta_sink(_maybe_stream_writer())
                if state.get("stream_final")
                else None
            ),
            hold_chars=0,
        )
        usage = _get_usage(response)
        content = _get_content(response) or (
            "已达到本次任务的步数上限，未能生成完整答复。"  # noqa: RUF001
            "以上为已检索到的中间过程，可继续追问以完成。"  # noqa: RUF001
        )
        msg = {"role": "assistant", "content": content}
        await _maybe_save_episode(
            state, episodic_memory, tenant_id, conversation_id, session_started_at,
            final_answer=content,
        )
        return {"next_action": "stop", "messages": [msg], "usage": usage}

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
        if state["next_action"] == "tool_call":
            if state.get("iteration", 0) < limit:
                return "plan"
            # Cap reached while the agent still wanted tools: synthesize a final
            # answer instead of ending silently on a tool-call turn (empty body).
            return "finalize"
        return END

    async def tool_executor(state: AgentState) -> dict[str, Any]:
        return await tool_executor_node(
            cast(dict[str, Any], state),
            {"configurable": {"tool_registry": registry, "tool_policy": tool_policy}},
        )

    builder = StateGraph(AgentState)
    builder.add_node("plan", plan)
    builder.add_node("finalize", finalize)
    builder.add_node("act", act)
    builder.add_node("tool_executor", tool_executor)
    builder.add_node("observe", observe)
    builder.set_entry_point("plan")
    builder.add_edge("plan", "act")
    builder.add_conditional_edges("act", route_after_act)
    builder.add_edge("tool_executor", "observe")
    builder.add_conditional_edges(
        "observe",
        should_continue,
        {"plan": "plan", "finalize": "finalize", END: END},
    )
    builder.add_edge("finalize", END)

    return builder.compile()
