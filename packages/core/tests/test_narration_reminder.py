"""The plan node re-asserts the narrate-before-acting rule every turn.

A single instruction at the top of the prompt loses salience as the context
grows with tool results, so the model starts firing tools with empty content and
the trace becomes a bare stack of tool rows. The plan node appends a per-turn
reminder as the LAST message before each tool-deciding LLM call to keep it
salient; it is never written into persistent state (so it cannot accumulate).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from sanyuan_agent.agents import build_simple_agent
from sanyuan_agent.agents.simple import _NARRATION_REMINDER, _inject_narration_reminder
from sanyuan_agent.models import LLMProvider


def test_inject_narration_reminder_appends_last() -> None:
    msgs = [{"role": "system", "content": "caps"}, {"role": "user", "content": "hi"}]
    out = _inject_narration_reminder(msgs)
    assert out[-1]["role"] == "system"
    assert out[-1]["content"] == _NARRATION_REMINDER
    # The original list is not mutated (immutability discipline).
    assert len(msgs) == 2


def _resp(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )


@pytest.mark.asyncio
async def test_plan_node_sends_reminder_as_last_message() -> None:
    provider = LLMProvider(name="test", provider_kind="openai", default_model="gpt-4")

    def echo(msg: str) -> str:
        """Echo."""
        return msg

    agent = build_simple_agent(provider, [echo])

    captured: dict[str, list] = {}

    async def fake_call_llm(_provider, messages, _tools, **_kw):
        captured["messages"] = messages
        return _resp("final answer, no tools")

    with patch("sanyuan_agent.agents.simple._call_llm", new=AsyncMock(side_effect=fake_call_llm)):
        await agent.ainvoke(
            {
                "messages": [
                    {"role": "system", "content": "You are iClaw."},
                    {"role": "user", "content": "do a thing"},
                ],
                "next_action": None,
                "tool_results": [],
                "iteration": 0,
                "max_iter": 2,
                "sensory": None,
                "working": None,
                "usage": {},
            }
        )

    sent = captured["messages"]
    assert sent[-1]["role"] == "system"
    assert sent[-1]["content"] == _NARRATION_REMINDER
    # The reminder mentions content + the user's language, and forbids empty content.
    assert "content" in _NARRATION_REMINDER
    assert "empty" in _NARRATION_REMINDER
