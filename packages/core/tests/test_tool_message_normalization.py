"""Regression: LangChain ToolMessage must not leak `status` into litellm.

Bug surfaced by the live eval smoke (smoke-003-web-research): the tool executor
appends LangChain ``ToolMessage`` objects to the agent state, and the next plan
turn fed them straight to litellm. ``ToolMessage.model_dump()`` carries a string
``status='success'`` (plus artifact / additional_kwargs / ...), which litellm's
``GPT3Message`` model rejects with::

    litellm.InternalServerError: OpenAIException -
    1 validation error for GPT3Message / status / dict_type

The fix normalizes every message to an OpenAI-compatible dict inside ``_call_llm``
before the litellm call. These tests pin both the unit normalizer and the full
tool-call -> ToolMessage -> next litellm call path.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import ToolMessage
from sanyuan_agent.agents import build_simple_agent
from sanyuan_agent.agents.simple import _to_openai_message
from sanyuan_agent.models import LLMProvider

# Fields litellm's GPT3Message rejects when they leak off a LangChain message.
_LEAKY_KEYS = {"status", "artifact", "additional_kwargs", "response_metadata", "type", "id"}


# ---------------------------------------------------------------------------
# Unit: _to_openai_message
# ---------------------------------------------------------------------------


class TestToOpenAIMessage:
    def test_dict_passes_through_unchanged(self) -> None:
        msg = {"role": "user", "content": "hi"}
        assert _to_openai_message(msg) is msg

    def test_toolmessage_maps_to_minimal_openai_dict(self) -> None:
        tm = ToolMessage(content="echo hi", tool_call_id="tc1", name="web_search")
        out = _to_openai_message(tm)
        assert out == {
            "role": "tool",
            "content": "echo hi",
            "tool_call_id": "tc1",
            "name": "web_search",
        }
        # crucially, none of the litellm-rejected fields survive
        assert _LEAKY_KEYS.isdisjoint(out)
        assert "status" not in out

    def test_toolmessage_model_dump_would_have_leaked_status(self) -> None:
        # Guards the premise: the raw model_dump DOES carry a string status.
        raw = ToolMessage(content="x", tool_call_id="tc1", name="web_search").model_dump()
        assert raw.get("status") == "success"  # this is what broke litellm
        # ...and our normalizer strips it.
        assert "status" not in _to_openai_message(
            ToolMessage(content="x", tool_call_id="tc1", name="web_search")
        )


# ---------------------------------------------------------------------------
# Integration: tool-call -> ToolMessage -> next litellm call
# ---------------------------------------------------------------------------


def _echo_tool(msg: str) -> str:
    """Echo back the input message."""
    return f"echo {msg}"


def _llm_response(content: str = "", tool_calls: list[Any] | None = None) -> Any:
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


def _tool_call(tc_id: str, name: str, arguments: str) -> Any:
    return SimpleNamespace(
        id=tc_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


@pytest.mark.asyncio
async def test_toolmessage_normalized_before_second_litellm_call() -> None:
    provider = LLMProvider(name="test", provider_kind="openai", default_model="gpt-4")
    agent = build_simple_agent(provider, [_echo_tool])

    # Plan 1 -> request the tool; Plan 2 -> stop with a final answer.
    responses = [
        _llm_response(tool_calls=[_tool_call("tc1", "_echo_tool", '{"msg": "hi"}')]),
        _llm_response(content="all done"),
    ]
    captured: list[list[Any]] = []

    async def fake_acompletion(**kwargs: Any) -> Any:
        captured.append(kwargs["messages"])
        return responses.pop(0)

    state: dict[str, Any] = {
        "messages": [{"role": "user", "content": "echo hi please"}],
        "next_action": None,
        "tool_results": [],
        "iteration": 0,
        "max_iter": 10,
        "sensory": None,
        "working": None,
        "usage": {},
    }

    with patch("litellm.acompletion", new=AsyncMock(side_effect=fake_acompletion)):
        await agent.ainvoke(state)

    # The agent looped back to plan after the tool ran -> two litellm calls.
    assert len(captured) == 2
    second_call = captured[1]

    # Every message handed to litellm is a plain dict (no ToolMessage objects),
    # and the whole payload is JSON-serializable (the strongest leak guard).
    assert all(isinstance(m, dict) for m in second_call)
    json.dumps(second_call)

    tool_msgs = [m for m in second_call if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    tool_msg = tool_msgs[0]
    assert tool_msg["content"] == "echo hi"
    assert tool_msg["tool_call_id"] == "tc1"
    # the bug: a leaked string ``status`` litellm's GPT3Message rejects
    assert "status" not in tool_msg
    assert _LEAKY_KEYS.isdisjoint(tool_msg)
