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
from neuroagent.memory.context import trim_messages
from neuroagent.memory.episodic import EpisodicMemory
from neuroagent.memory.longterm.store import LongtermMemoryStore, ScopeKind
from neuroagent.memory.recall import EpisodicRecall
from neuroagent.memory.sensory import SensoryEvent
from neuroagent.memory.vector import VectorRecall
from neuroagent.memory.working import WorkingMemory
from neuroagent.models import LLMProvider
from neuroagent.security.tool_policy import ToolPolicy
from neuroagent.skills.registry import SkillRegistry
from neuroagent.tools.registry import ToolRegistry

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


def _to_tool_def(tool: Callable[..., Any]) -> dict[str, Any]:
    """Convert a callable to an OpenAI-style tool definition.

    A tool that carries an explicit ``__tool_schema__`` (the ``function`` object)
    — e.g. an MCP-adapted tool whose JSON schema can't be reflected from a
    generic ``**kwargs`` wrapper — has it used verbatim. Otherwise the schema is
    reflected from the callable's signature.
    """
    explicit = getattr(tool, "__tool_schema__", None)
    if isinstance(explicit, dict):
        return {"type": "function", "function": explicit}

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


async def _call_llm(
    provider: LLMProvider,
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]],
    metadata: dict[str, str] | None = None,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Any:
    """Unified async LLM call via LiteLLM.

    *metadata* (run_id / tenant_id) rides along in litellm's metadata so the
    usage success-callback can attribute the call without global state.

    *temperature* / *max_tokens* are passed through only when set, so callers
    that omit them keep litellm's defaults (no behaviour change).
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

    # Try the primary model, then each fallback in order on a *transient*
    # failure. Non-transient errors bubble immediately.
    model_chain = (provider.default_model, *provider.fallback_models)
    last_exc: Exception | None = None
    for i, model in enumerate(model_chain):
        try:
            return await litellm.acompletion(model=model, **base_kwargs)
        except fallback_exceptions as exc:
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


async def _maybe_inject_recall(
    messages: list[dict[str, Any]],
    recall: EpisodicRecall | None,
    tenant_id: str,
    conversation_id: str,
) -> list[dict[str, Any]]:
    """Fetch relevant episodes via recall and inject them as a system message.

    Failures are silently swallowed — episodic recall is a best-effort
    optimisation, not a hard dependency.
    """
    if recall is None or not tenant_id:
        return messages

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
) -> list[dict[str, Any]]:
    """Fetch relevant long-term memories and inject as a system message."""
    if longterm is None or not tenant_id:
        return messages

    user_query = _extract_user_query(messages)
    if not user_query.strip():
        return messages

    try:
        entries = await longterm.retrieve_semantic(
            tenant_id=tenant_id,
            query=user_query,
            scope=[ScopeKind.PRIVATE, ScopeKind.TEAM, ScopeKind.TENANT],
            limit=5,
            importance_threshold=0.3,
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
) -> list[dict[str, Any]]:
    """Check user message for skill trigger keywords and inject skill prompt."""
    if skill_registry is None:
        return messages

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
    long-term store.  Results are injected into the plan node's system prompt.
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

        bounded = trim_messages(state["messages"], token_budget=context_token_budget)
        llm_messages = _inject_working_summary(bounded, working)
        llm_messages = _maybe_inject_skill(llm_messages, skill_registry)
        llm_messages = await _maybe_inject_recall(
            llm_messages,
            effective_recall,
            tenant_id,
            conversation_id,
        )
        llm_messages = await _maybe_inject_longterm(
            llm_messages,
            longterm_memory,
            tenant_id,
            user_id,
        )

        response = await _call_llm(
            llm_provider,
            llm_messages,
            registry.tool_defs(),
            metadata=llm_metadata,
            temperature=temperature,
            max_tokens=max_tokens,
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
            state, episodic_memory, tenant_id, conversation_id, session_started_at
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
        if (
            state["next_action"] == "tool_call"
            and state.get("iteration", 0) < limit
        ):
            return "plan"
        return END

    async def tool_executor(state: AgentState) -> dict[str, Any]:
        return await tool_executor_node(
            cast(dict[str, Any], state),
            {"configurable": {"tool_registry": registry, "tool_policy": tool_policy}},
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
