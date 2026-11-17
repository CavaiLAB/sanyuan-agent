"""Tests for the simplified LangGraph agent (plan → act → tool_executor → observe)."""

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import ToolMessage
from sanyuan_agent.agents import build_simple_agent
from sanyuan_agent.models import LLMProvider

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


def _role(msg):
    """Extract role/type from a message (dict or langchain Message)."""
    if hasattr(msg, "type"):
        return msg.type
    return msg.get("role")


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
        with patch("sanyuan_agent.agents.simple._call_llm", new=mock_llm):
            result = await agent.ainvoke({
                "messages": [{"role": "user", "content": "Say hi"}],
                "next_action": None,
                "tool_results": [],
                "iteration": 0,
                "max_iter": 10,
            })

        # 2 plan calls (tool_call + stop)
        assert mock_llm.await_count == 2

        # Final assistant message
        assistant_msgs = [m for m in result["messages"] if _role(m) == "assistant"]
        assert assistant_msgs[-1]["content"] == "All done"

        # Tool result was injected as ToolMessage
        tool_msgs = [m for m in result["messages"] if _role(m) == "tool"]
        assert len(tool_msgs) == 1
        assert isinstance(tool_msgs[0], ToolMessage)
        assert tool_msgs[0].content == "echo: hi"
        assert tool_msgs[0].tool_call_id == "tc-1"

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

        with patch("sanyuan_agent.agents.simple._call_llm", new=mock_llm):
            result = await agent.ainvoke({
                "messages": [{"role": "user", "content": "Hello"}],
                "next_action": None,
                "tool_results": [],
                "iteration": 0,
                "max_iter": 10,
            })

        assert mock_llm.await_count == 1
        assistant_msgs = [m for m in result["messages"] if _role(m) == "assistant"]
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
        with patch("sanyuan_agent.agents.simple._call_llm", new=mock_llm):
            result = await agent.ainvoke({
                "messages": [{"role": "user", "content": "Test async"}],
                "next_action": None,
                "tool_results": [],
                "iteration": 0,
                "max_iter": 10,
            })

        tool_msgs = [m for m in result["messages"] if _role(m) == "tool"]
        assert tool_msgs[0].content == "async-echo: hello"

    async def test_max_iter_forces_final_answer(self):
        """At the iteration cap the agent still produces a final answer.

        Regression: the loop used to END on a tool-call turn with no assistant
        prose — an empty body the client dropped on reload (user sees "no final
        answer", conversation looks lost). Now a `finalize` turn makes one
        tool-less LLM call so the body is never empty.
        """
        provider = LLMProvider(
            name="test",
            provider_kind="openai",
            default_model="gpt-4",
        )
        agent = build_simple_agent(provider, [_echo_tool], max_iter=2)

        # LLM keeps requesting tools; the cap is hit and `finalize` synthesizes.
        responses = [
            _MockResponse([
                _MockChoice(_MockMessage(
                    content="",
                    tool_calls=[_MockToolCall("tc-1", "_echo_tool", '{"msg": "hi"}')],
                ))
            ]),
            _MockResponse([
                _MockChoice(_MockMessage(
                    content="",
                    tool_calls=[_MockToolCall("tc-2", "_echo_tool", '{"msg": "again"}')],
                ))
            ]),
            _MockResponse([
                _MockChoice(_MockMessage(content="最终汇总")),
            ]),
        ]

        mock_llm = AsyncMock(side_effect=responses)
        with patch("sanyuan_agent.agents.simple._call_llm", new=mock_llm):
            result = await agent.ainvoke({
                "messages": [{"role": "user", "content": "Loop test"}],
                "next_action": None,
                "tool_results": [],
                "iteration": 0,
                "max_iter": 2,
            })

        # 2 plan calls (capped) + 1 forced finalize call.
        assert mock_llm.await_count == 3
        # The run ends with a real assistant answer, not a silent tool-call turn.
        assistant_msgs = [m for m in result["messages"] if _role(m) == "assistant"]
        assert assistant_msgs[-1]["content"] == "最终汇总"
        # The finalize call is made WITHOUT tools (3rd positional arg).
        assert mock_llm.await_args_list[-1].args[2] == []

    async def test_max_iter_finalize_empty_content_falls_back(self):
        """If the forced finalize call returns no content, use a safe fallback
        so the body is still non-empty (never a blank turn)."""
        provider = LLMProvider(
            name="test",
            provider_kind="openai",
            default_model="gpt-4",
        )
        agent = build_simple_agent(provider, [_echo_tool], max_iter=1)

        responses = [
            _MockResponse([
                _MockChoice(_MockMessage(
                    content="",
                    tool_calls=[_MockToolCall("tc-1", "_echo_tool", '{"msg": "hi"}')],
                ))
            ]),
            _MockResponse([_MockChoice(_MockMessage(content=""))]),
        ]

        mock_llm = AsyncMock(side_effect=responses)
        with patch("sanyuan_agent.agents.simple._call_llm", new=mock_llm):
            result = await agent.ainvoke({
                "messages": [{"role": "user", "content": "Loop test"}],
                "next_action": None,
                "tool_results": [],
                "iteration": 0,
                "max_iter": 1,
            })

        assert mock_llm.await_count == 2
        assistant_msgs = [m for m in result["messages"] if _role(m) == "assistant"]
        assert assistant_msgs[-1]["content"].strip() != ""
        assert "步数上限" in assistant_msgs[-1]["content"]
