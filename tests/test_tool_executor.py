"""Tests for ToolRegistry and tool_executor_node."""

import pytest
from neuroagent.agents.tool_executor import _last_assistant_tool_calls, tool_executor_node
from neuroagent.tools.registry import ToolRegistry, callable_to_schema

# -- helpers --

def _echo_tool(msg: str) -> str:
    """Echo back the input message."""
    return f"echo: {msg}"


async def _async_add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class TestToolRegistry:
    def test_register_and_get(self):
        reg = ToolRegistry()
        schema = callable_to_schema(_echo_tool)
        reg.register("echo", _echo_tool, schema)
        assert reg.has_tool("echo")
        assert reg.get("echo") is _echo_tool
        assert reg.get_schema("echo")["name"] == "_echo_tool"

    def test_list_tools_sorted(self):
        reg = ToolRegistry()
        reg.register("z", _echo_tool, {})
        reg.register("a", _echo_tool, {})
        assert reg.list_tools() == ["a", "z"]

    def test_tool_defs_format(self):
        reg = ToolRegistry()
        reg.register("echo", _echo_tool, callable_to_schema(_echo_tool))
        defs = reg.tool_defs()
        assert len(defs) == 1
        assert defs[0]["type"] == "function"
        assert defs[0]["function"]["name"] == "_echo_tool"

    @pytest.mark.asyncio
    async def test_execute_sync_tool(self):
        reg = ToolRegistry()
        reg.register("echo", _echo_tool, callable_to_schema(_echo_tool))
        result = await reg.execute("echo", '{"msg": "hi"}')
        assert result == "echo: hi"

    @pytest.mark.asyncio
    async def test_execute_async_tool(self):
        reg = ToolRegistry()
        reg.register("add", _async_add, callable_to_schema(_async_add))
        result = await reg.execute("add", '{"a": 2, "b": 3}')
        assert result == 5

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        reg = ToolRegistry()
        result = await reg.execute("missing", "{}")
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_execute_tool_raises(self):
        def _boom():
            raise RuntimeError("kapow")

        reg = ToolRegistry()
        reg.register("boom", _boom, callable_to_schema(_boom))
        result = await reg.execute("boom", "{}")
        assert "Error: kapow" in result


class TestLastAssistantToolCalls:
    def test_extracts_tool_calls(self):
        state = {
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "tc1", "function": {"name": "echo", "arguments": "{}"}}
                    ],
                },
            ]
        }
        tcs = _last_assistant_tool_calls(state)
        assert len(tcs) == 1
        assert tcs[0]["id"] == "tc1"

    def test_empty_when_no_messages(self):
        assert _last_assistant_tool_calls({}) == []

    def test_empty_when_last_is_user(self):
        state = {"messages": [{"role": "user", "content": "hi"}]}
        assert _last_assistant_tool_calls(state) == []

    def test_empty_when_no_tool_calls(self):
        state = {"messages": [{"role": "assistant", "content": "hello"}]}
        assert _last_assistant_tool_calls(state) == []


class TestToolExecutorNode:
    @pytest.mark.asyncio
    async def test_single_tool_call(self):
        reg = ToolRegistry()
        reg.register("echo", _echo_tool, callable_to_schema(_echo_tool))

        state = {
            "messages": [
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "tc-1", "function": {"name": "echo", "arguments": '{"msg": "hi"}'}}
                ]}
            ]
        }
        config = {"configurable": {"tool_registry": reg}}

        result = await tool_executor_node(state, config)

        msgs = result["messages"]
        assert len(msgs) == 1
        assert msgs[0].content == "echo: hi"
        assert msgs[0].tool_call_id == "tc-1"
        assert msgs[0].name == "echo"

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self):
        reg = ToolRegistry()
        reg.register("echo", _echo_tool, callable_to_schema(_echo_tool))
        reg.register("add", _async_add, callable_to_schema(_async_add))

        state = {
            "messages": [
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "tc-1", "function": {"name": "echo", "arguments": '{"msg": "a"}'}},
                    {"id": "tc-2", "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'}},
                ]}
            ]
        }
        config = {"configurable": {"tool_registry": reg}}

        result = await tool_executor_node(state, config)

        msgs = result["messages"]
        assert len(msgs) == 2
        assert msgs[0].content == "echo: a"
        assert msgs[1].content == "3"

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error_message(self):
        reg = ToolRegistry()
        state = {
            "messages": [
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "tc-1", "function": {"name": "missing", "arguments": "{}"}}
                ]}
            ]
        }
        config = {"configurable": {"tool_registry": reg}}

        result = await tool_executor_node(state, config)

        assert "not found" in result["messages"][0].content

    @pytest.mark.asyncio
    async def test_empty_tool_calls_short_circuit(self):
        reg = ToolRegistry()
        state = {
            "messages": [
                {"role": "assistant", "content": "hello"}
            ]
        }
        config = {"configurable": {"tool_registry": reg}}

        result = await tool_executor_node(state, config)
        assert result == {}

    @pytest.mark.asyncio
    async def test_no_registry_short_circuit(self):
        state = {
            "messages": [
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "tc-1", "function": {"name": "echo", "arguments": "{}"}}
                ]}
            ]
        }
        result = await tool_executor_node(state, {})
        assert result == {}

    @pytest.mark.asyncio
    async def test_sandbox_isolation_via_bound_callable(self):
        """Different registry instances with different bound tools are isolated."""
        reg1 = ToolRegistry()
        reg2 = ToolRegistry()

        def _make_bound(prefix: str):
            def _tool(msg: str) -> str:
                return f"{prefix}: {msg}"
            return _tool

        reg1.register("echo", _make_bound("A"), callable_to_schema(_make_bound("A")))
        reg2.register("echo", _make_bound("B"), callable_to_schema(_make_bound("B")))

        state = {
            "messages": [
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "tc-1", "function": {"name": "echo", "arguments": '{"msg": "x"}'}}
                ]}
            ]
        }

        result1 = await tool_executor_node(state, {"configurable": {"tool_registry": reg1}})
        result2 = await tool_executor_node(state, {"configurable": {"tool_registry": reg2}})

        assert result1["messages"][0].content == "A: x"
        assert result2["messages"][0].content == "B: x"
