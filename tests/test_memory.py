"""Tests for SensoryMemory, WorkingMemory, and InMemoryStore."""

import time
from unittest.mock import AsyncMock, patch

import pytest
from sanyuan_agent.agents import build_simple_agent
from sanyuan_agent.memory.abc import InMemoryStore
from sanyuan_agent.memory.sensory import SensoryEvent, SensoryMemory
from sanyuan_agent.memory.working import WorkingMemory
from sanyuan_agent.models import LLMProvider

# -- InMemoryStore --


class TestInMemoryStore:
    def test_save_and_load(self):
        store = InMemoryStore()
        store.save("key", {"a": 1})
        assert store.load("key") == {"a": 1}

    def test_load_default(self):
        store = InMemoryStore()
        assert store.load("missing", "default") == "default"

    def test_clear(self):
        store = InMemoryStore()
        store.save("key", "val")
        store.clear()
        assert store.load("key") is None

    def test_overwrite(self):
        store = InMemoryStore()
        store.save("key", "first")
        store.save("key", "second")
        assert store.load("key") == "second"


# -- SensoryMemory --


class TestSensoryMemory:
    def test_append_and_get_recent(self):
        mem = SensoryMemory()
        mem.append(SensoryEvent(kind="user", content="hello"))
        mem.append(SensoryEvent(kind="tool_result", content="result"))
        recent = mem.get_recent(2)
        assert len(recent) == 2
        assert recent[0].content == "hello"
        assert recent[1].content == "result"

    def test_get_recent_newest_first(self):
        mem = SensoryMemory()
        for i in range(5):
            mem.append(SensoryEvent(kind="user", content=str(i)))
        recent = mem.get_recent(3)
        assert [e.content for e in recent] == ["2", "3", "4"]

    def test_ring_buffer_overflow(self):
        mem = SensoryMemory(maxlen=3)
        for i in range(10):
            mem.append(SensoryEvent(kind="user", content=str(i)))
        all_events = mem.get_recent(100)
        assert len(all_events) == 3
        assert [e.content for e in all_events] == ["7", "8", "9"]

    def test_persist_to_store(self):
        store = InMemoryStore()
        mem = SensoryMemory(store=store)
        mem.append(SensoryEvent(kind="user", content="persisted"))

        # Reconstruct from same store
        mem2 = SensoryMemory(store=store)
        recent = mem2.get_recent(1)
        assert len(recent) == 1
        assert recent[0].content == "persisted"

    def test_clear(self):
        mem = SensoryMemory()
        mem.append(SensoryEvent(kind="user", content="x"))
        mem.clear()
        assert mem.get_recent(1) == []

    def test_ts_default(self):
        before = time.time()
        evt = SensoryEvent(kind="user", content="test")
        after = time.time()
        assert before <= evt.ts <= after


# -- WorkingMemory --


class TestWorkingMemory:
    def test_add_step_and_get_summary(self):
        mem = WorkingMemory()
        mem.add_step("plan A", "action A", "result A")
        mem.add_step("plan B", "action B", "result B")
        summary = mem.get_summary(max_steps=2)
        assert "plan A" in summary
        assert "plan B" in summary
        assert "action A" in summary
        assert "result B" in summary

    def test_get_summary_limits_steps(self):
        mem = WorkingMemory()
        for i in range(10):
            mem.add_step(f"plan {i}", f"action {i}", f"result {i}")
        summary = mem.get_summary(max_steps=3)
        assert "plan 7" in summary
        assert "plan 9" in summary
        assert "plan 0" not in summary

    def test_get_summary_empty(self):
        mem = WorkingMemory()
        assert mem.get_summary() == "(no previous steps)"

    def test_persist_to_store(self):
        store = InMemoryStore()
        mem = WorkingMemory(store=store)
        mem.add_step("p1", "a1", "r1")

        mem2 = WorkingMemory(store=store)
        summary = mem2.get_summary()
        assert "p1" in summary
        assert "a1" in summary

    def test_clear(self):
        mem = WorkingMemory()
        mem.add_step("p", "a", "r")
        mem.clear()
        assert mem.get_summary() == "(no previous steps)"

    def test_summary_format_stable(self):
        mem = WorkingMemory()
        mem.add_step("search web", "web_search(query='py')", "found 5 results")
        summary = mem.get_summary(max_steps=1)
        lines = summary.splitlines()
        assert lines[0] == "Step 1:"
        assert lines[1] == "  Plan: search web"
        assert lines[2] == "  Action: web_search(query='py')"
        assert lines[3] == "  Result: found 5 results"


# -- Agent end-to-end with memory --


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
class TestAgentMemory:
    async def test_memory_populated_after_tool_call(self):
        provider = LLMProvider(
            name="test",
            provider_kind="openai",
            default_model="gpt-4",
        )
        agent = build_simple_agent(provider, [_echo_tool])

        responses = [
            _MockResponse([
                _MockChoice(_MockMessage(
                    content="call echo",
                    tool_calls=[_MockToolCall("tc-1", "_echo_tool", '{"msg": "hi"}')],
                ))
            ]),
            _MockResponse([
                _MockChoice(_MockMessage(content="Done")),
            ]),
        ]

        sensory = SensoryMemory()
        working = WorkingMemory()

        mock_llm = AsyncMock(side_effect=responses)
        with patch("sanyuan_agent.agents.simple._call_llm", new=mock_llm):
            await agent.ainvoke({
                "messages": [{"role": "user", "content": "Say hi"}],
                "next_action": None,
                "tool_results": [],
                "iteration": 0,
                "max_iter": 10,
                "sensory": sensory,
                "working": working,
            })

        # Sensory should have user + tool_result
        recent = sensory.get_recent(10)
        assert len(recent) == 2
        assert recent[0].kind == "user"
        assert recent[0].content == "Say hi"
        assert recent[1].kind == "tool_result"
        assert "echo" in recent[1].content

        # Working should have one step
        summary = working.get_summary()
        assert "call echo" in summary
        assert "_echo_tool" in summary
        assert "echo: hi" in summary

    async def test_working_summary_injected_into_plan(self):
        provider = LLMProvider(
            name="test",
            provider_kind="openai",
            default_model="gpt-4",
        )
        agent = build_simple_agent(provider, [_echo_tool])

        responses = [
            _MockResponse([
                _MockChoice(_MockMessage(content="first")),
            ]),
            _MockResponse([
                _MockChoice(_MockMessage(
                    content="second",
                    tool_calls=[_MockToolCall("tc-1", "_echo_tool", '{"msg": "a"}')],
                ))
            ]),
            _MockResponse([
                _MockChoice(_MockMessage(content="third")),
            ]),
        ]

        sensory = SensoryMemory()
        working = WorkingMemory()
        # Pre-seed working memory
        working.add_step("plan-0", "action-0", "result-0")

        mock_llm = AsyncMock(side_effect=responses)
        with patch("sanyuan_agent.agents.simple._call_llm", new=mock_llm):
            await agent.ainvoke({
                "messages": [{"role": "user", "content": "Test"}],
                "next_action": None,
                "tool_results": [],
                "iteration": 0,
                "max_iter": 10,
                "sensory": sensory,
                "working": working,
            })

        # Verify _call_llm was called with summary in messages
        calls = mock_llm.await_args_list
        # At least one call should have system message with "Recent context"
        system_calls = [
            c for c in calls
            if c.args[1] and c.args[1][0].get("role") == "system"
        ]
        assert len(system_calls) >= 1
        assert "plan-0" in system_calls[0].args[1][0]["content"]

    async def test_no_memory_fields_agent_runs(self):
        """Agent works when sensory/working are omitted (backward compat)."""
        provider = LLMProvider(
            name="test",
            provider_kind="openai",
            default_model="gpt-4",
        )
        agent = build_simple_agent(provider, [_echo_tool])

        responses = [
            _MockResponse([
                _MockChoice(_MockMessage(content="no tools")),
            ]),
        ]

        mock_llm = AsyncMock(side_effect=responses)
        with patch("sanyuan_agent.agents.simple._call_llm", new=mock_llm):
            result = await agent.ainvoke({
                "messages": [{"role": "user", "content": "Hello"}],
                "next_action": None,
                "tool_results": [],
                "iteration": 0,
                "max_iter": 10,
            })

        assert result is not None
