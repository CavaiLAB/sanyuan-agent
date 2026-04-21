"""Tests for the simplified 3-node LangGraph agent."""

import pytest
from unittest.mock import AsyncMock, patch

from neuroagent.agents import build_simple_agent
from neuroagent.models import LLMProvider


# -- helpers --

class _MockToolCall:
    def __init__(self, tc_id, name, arguments):
        self.id = tc_id
        self.type = "function"
        self.function = _MockFunction(name, arguments)


class _MockFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _MockMessage:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.role = "assistant"
        self.tool_calls = tool_calls


class _MockChoice:
    def __init__(self, message):
        self.message = message


class _MockResponse:
    def __init__(self, choices):
        self.choices = choices


def _echo_tool(msg: str) -> str:
    """Echo back the input message."""
    return f"echo: {msg}"


@pytest.mark.asyncio
class TestSimpleAgent:
    async def test_tool_call_then_stop(self):
        """LLM requests a tool call, then stops with final answer."""
        provider = LLMProvider(
            name="test",
            provider_kind="openai",
            default_model="gpt-4",
        )
        agent = build_simple_agent(provider, [_echo_tool])

        # First LLM call → tool_call; Second → stop
        responses = [
            _MockResponse([
                _MockChoice(_MockMessage(
                    content="",
                    tool_calls=[_MockToolCall("tc-1", "_echo_tool", '{"msg": "hi"}')],
                ))
            ]),
            _MockResponse([
                _MockChoice(_MockMessage(content="All done")),
            ]),
        ]

        mock_llm = AsyncMock(side_effect=responses)
        with patch("neuroagent.agents.simple._call_llm", new=mock_llm):
            result = await agent.ainvoke({
                "messages": [{"role": "user", "content": "Say hi"}],
                "next_action": None,
                "tool_results": [],
            })

        # 2 plan calls (tool_call + stop)
        assert mock_llm.await_count == 2

        # Final assistant message
        assistant_msgs = [m for m in result["messages"] if m.get("role") == "assistant"]
        assert assistant_msgs[-1]["content"] == "All done"

        # Tool result was injected
        tool_msgs = [m for m in result["messages"] if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["content"] == "echo: hi"

    async def test_stop_immediately(self):
        """LLM decides to stop on first plan — no tool calls."""
        provider = LLMProvider(
            name="test",
            provider_kind="openai",
            default_model="gpt-4",
        )
        agent = build_simple_agent(provider, [_echo_tool])

        mock_llm = AsyncMock(return_value=_MockResponse([
            _MockChoice(_MockMessage(content="No tools needed")),
        ]))

        with patch("neuroagent.agents.simple._call_llm", new=mock_llm):
            result = await agent.ainvoke({
                "messages": [{"role": "user", "content": "Hello"}],
                "next_action": None,
                "tool_results": [],
            })

        assert mock_llm.await_count == 1
        assistant_msgs = [m for m in result["messages"] if m.get("role") == "assistant"]
        assert assistant_msgs[-1]["content"] == "No tools needed"

    async def test_async_tool(self):
        """Agent can invoke async tools."""
        async def async_echo(msg: str) -> str:
            return f"async-echo: {msg}"

        provider = LLMProvider(
            name="test",
            provider_kind="openai",
            default_model="gpt-4",
        )
        agent = build_simple_agent(provider, [async_echo])

        responses = [
            _MockResponse([
                _MockChoice(_MockMessage(
                    content="",
                    tool_calls=[_MockToolCall("tc-1", "async_echo", '{"msg": "hello"}')],
                ))
            ]),
            _MockResponse([
                _MockChoice(_MockMessage(content="Finished")),
            ]),
        ]

        mock_llm = AsyncMock(side_effect=responses)
        with patch("neuroagent.agents.simple._call_llm", new=mock_llm):
            result = await agent.ainvoke({
                "messages": [{"role": "user", "content": "Test async"}],
                "next_action": None,
                "tool_results": [],
            })

        tool_msgs = [m for m in result["messages"] if m.get("role") == "tool"]
        assert tool_msgs[0]["content"] == "async-echo: hello"
