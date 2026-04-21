"""Tests for tool call recovery — validation, circuit breaker, sanitization."""

import asyncio

import pytest
from langchain_core.messages import ToolMessage
from neuroagent.agents.errors import (
    ToolArgsValidationError,
    ToolCircuitBreakerError,
    ToolExecutionError,
)
from neuroagent.agents.tool_executor import (
    _recover_tool_call,
    _sanitize_tool_call,
    _validate_args,
    tool_executor_node,
)
from neuroagent.tools.registry import ToolRegistry, callable_to_schema


def _echo_tool(msg: str) -> str:
    """Echo back the input message."""
    return f"echo: {msg}"


def _add_tool(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def _boom_tool():
    """Always raises."""
    raise RuntimeError("kapow")


# -- _sanitize_tool_call --


class TestSanitizeToolCall:
    def test_well_formed_passthrough(self):
        tc = {
            "id": "tc-1",
            "type": "function",
            "function": {"name": "echo", "arguments": '{"msg": "hi"}'},
        }
        result = _sanitize_tool_call(tc)
        assert result == tc

    def test_fixes_non_string_arguments(self):
        tc = {
            "id": "tc-1",
            "function": {"name": "echo", "arguments": {"msg": "hi"}},
        }
        result = _sanitize_tool_call(tc)
        assert result is not None
        assert result["function"]["arguments"] == '{"msg": "hi"}'

    def test_fixes_none_arguments(self):
        tc = {
            "id": "tc-1",
            "function": {"name": "echo", "arguments": None},
        }
        result = _sanitize_tool_call(tc)
        assert result is not None
        assert result["function"]["arguments"] == "{}"

    def test_strips_empty_name(self):
        tc = {
            "id": "tc-1",
            "function": {"name": "  echo  ", "arguments": "{}"},
        }
        result = _sanitize_tool_call(tc)
        assert result["function"]["name"] == "echo"

    def test_returns_none_on_broken_structure(self):
        tc = {"id": "tc-1", "function": "not-a-dict"}
        assert _sanitize_tool_call(tc) is None


# -- _validate_args --


class TestValidateArgs:
    def test_valid_args_pass(self):
        reg = ToolRegistry()
        reg.register("echo", _echo_tool, callable_to_schema(_echo_tool))
        _validate_args("echo", {"msg": "hi"}, reg)

    def test_missing_required_raises(self):
        reg = ToolRegistry()
        reg.register("echo", _echo_tool, callable_to_schema(_echo_tool))
        with pytest.raises(ToolArgsValidationError) as exc_info:
            _validate_args("echo", {}, reg)
        assert "msg" in str(exc_info.value)

    def test_wrong_type_raises(self):
        reg = ToolRegistry()
        reg.register("add", _add_tool, callable_to_schema(_add_tool))
        with pytest.raises(ToolArgsValidationError) as exc_info:
            _validate_args("add", {"a": "not-an-int", "b": 2}, reg)
        assert "a" in str(exc_info.value)


# -- _recover_tool_call --


@pytest.mark.asyncio
class TestRecoverToolCall:
    async def test_unknown_tool_lists_available(self):
        reg = ToolRegistry()
        reg.register("echo", _echo_tool, callable_to_schema(_echo_tool))

        tc = {
            "id": "tc-1",
            "function": {"name": "missing", "arguments": "{}"},
        }
        msg = await _recover_tool_call(tc, reg)

        assert isinstance(msg, ToolMessage)
        assert "not found" in msg.content
        assert "echo" in msg.content
        assert msg.name == "missing"

    async def test_empty_name_returns_error(self):
        reg = ToolRegistry()
        tc = {
            "id": "tc-1",
            "function": {"name": "", "arguments": "{}"},
        }
        msg = await _recover_tool_call(tc, reg)

        assert isinstance(msg, ToolMessage)
        assert "Missing tool name" in msg.content
        assert msg.name == "unknown"

    async def test_invalid_json_args_returns_error(self):
        reg = ToolRegistry()
        reg.register("echo", _echo_tool, callable_to_schema(_echo_tool))

        tc = {
            "id": "tc-1",
            "function": {"name": "echo", "arguments": "not-json"},
        }
        msg = await _recover_tool_call(tc, reg)

        assert isinstance(msg, ToolMessage)
        assert "Invalid JSON" in msg.content
        assert "msg" in msg.content  # expected param hint
        assert msg.name == "echo"

    async def test_schema_validation_error_returns_error(self):
        reg = ToolRegistry()
        reg.register("echo", _echo_tool, callable_to_schema(_echo_tool))

        tc = {
            "id": "tc-1",
            "function": {"name": "echo", "arguments": "{}"},
        }
        msg = await _recover_tool_call(tc, reg)

        assert isinstance(msg, ToolMessage)
        assert "Argument validation failed" in msg.content
        assert "echo" in msg.content

    async def test_happy_path(self):
        reg = ToolRegistry()
        reg.register("echo", _echo_tool, callable_to_schema(_echo_tool))

        tc = {
            "id": "tc-1",
            "function": {"name": "echo", "arguments": '{"msg": "hi"}'},
        }
        msg = await _recover_tool_call(tc, reg)

        assert isinstance(msg, ToolMessage)
        assert msg.content == "echo: hi"
        assert msg.tool_call_id == "tc-1"
        assert msg.name == "echo"

    async def test_execution_error_includes_traceback(self):
        reg = ToolRegistry()
        reg.register("boom", _boom_tool, callable_to_schema(_boom_tool))

        tc = {
            "id": "tc-1",
            "function": {"name": "boom", "arguments": "{}"},
        }
        msg = await _recover_tool_call(tc, reg)

        assert isinstance(msg, ToolMessage)
        assert "[tool-error]" in msg.content
        assert "kapow" in msg.content
        assert "Traceback" in msg.content
        assert msg.name == "boom"

    async def test_circuit_breaker_error_message(self):
        reg = ToolRegistry(breaker_threshold=2)
        reg.register("boom", _boom_tool, callable_to_schema(_boom_tool))

        # First failure
        tc = {
            "id": "tc-1",
            "function": {"name": "boom", "arguments": "{}"},
        }
        msg1 = await _recover_tool_call(tc, reg)
        assert "[tool-error]" in msg1.content

        # Second failure
        tc["id"] = "tc-2"
        msg2 = await _recover_tool_call(tc, reg)
        assert "[tool-error]" in msg2.content

        # Third call → circuit breaker open
        tc["id"] = "tc-3"
        msg3 = await _recover_tool_call(tc, reg)

        assert isinstance(msg3, ToolMessage)
        assert "Circuit breaker open" in msg3.content
        assert "2/2" in msg3.content or "3/2" in msg3.content
        assert msg3.name == "boom"


# -- ToolRegistry circuit breaker --


class TestCircuitBreaker:
    @pytest.mark.asyncio
    async def test_consecutive_failures_trip_breaker(self):
        reg = ToolRegistry(breaker_threshold=3)
        reg.register("boom", _boom_tool, callable_to_schema(_boom_tool))

        for _ in range(3):
            with pytest.raises(ToolExecutionError):
                await reg.execute("boom", "{}")

        with pytest.raises(ToolCircuitBreakerError) as exc_info:
            await reg.execute("boom", "{}")
        assert exc_info.value.fail_count >= 3
        assert exc_info.value.threshold == 3

    @pytest.mark.asyncio
    async def test_success_resets_fail_count(self):
        reg = ToolRegistry(breaker_threshold=3)
        reg.register("echo", _echo_tool, callable_to_schema(_echo_tool))

        # Simulate a failure by wrapping in try/except (echo won't fail naturally)
        # Instead, register a failing tool, fail twice, then swap to echo
        reg.register("flip", _boom_tool, callable_to_schema(_boom_tool))

        for _ in range(2):
            with pytest.raises(ToolExecutionError):
                await reg.execute("flip", "{}")

        # Replace with success tool
        reg._tools["flip"] = _echo_tool

        result = await reg.execute("flip", '{"msg": "ok"}')
        assert result == "echo: ok"

        stats = reg.get_stats("flip")
        assert stats is not None
        assert stats.fail_count == 0

    @pytest.mark.asyncio
    async def test_reset_stats_clears_breaker(self):
        reg = ToolRegistry(breaker_threshold=2)
        reg.register("boom", _boom_tool, callable_to_schema(_boom_tool))

        for _ in range(2):
            with pytest.raises(ToolExecutionError):
                await reg.execute("boom", "{}")

        with pytest.raises(ToolCircuitBreakerError):
            await reg.execute("boom", "{}")

        reg.reset_stats("boom")

        # Should work again (but tool still raises)
        with pytest.raises(ToolExecutionError):
            await reg.execute("boom", "{}")

    @pytest.mark.asyncio
    async def test_stats_tracked_correctly(self):
        reg = ToolRegistry(breaker_threshold=5)
        reg.register("echo", _echo_tool, callable_to_schema(_echo_tool))
        reg.register("boom", _boom_tool, callable_to_schema(_boom_tool))

        await reg.execute("echo", '{"msg": "a"}')
        await reg.execute("echo", '{"msg": "b"}')

        with pytest.raises(ToolExecutionError):
            await reg.execute("boom", "{}")

        echo_stats = reg.get_stats("echo")
        assert echo_stats is not None
        assert echo_stats.call_count == 2
        assert echo_stats.fail_count == 0

        boom_stats = reg.get_stats("boom")
        assert boom_stats is not None
        assert boom_stats.call_count == 1
        assert boom_stats.fail_count == 1
        assert "kapow" in boom_stats.last_error

    @pytest.mark.asyncio
    async def test_concurrent_calls_stats_safe(self):
        reg = ToolRegistry(breaker_threshold=10)
        reg.register("echo", _echo_tool, callable_to_schema(_echo_tool))

        async def call() -> str:
            return await reg.execute("echo", '{"msg": "x"}')

        await asyncio.gather(*(call() for _ in range(50)))

        stats = reg.get_stats("echo")
        assert stats is not None
        assert stats.call_count == 50
        assert stats.fail_count == 0


# -- Malformed tool_calls via tool_executor_node --


@pytest.mark.asyncio
class TestToolExecutorNodeRecovery:
    async def test_malformed_tool_call_returns_error(self):
        reg = ToolRegistry()
        reg.register("echo", _echo_tool, callable_to_schema(_echo_tool))

        state = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "tc-1", "function": "not-a-dict"}
                    ],
                }
            ]
        }
        config = {"configurable": {"tool_registry": reg}}

        result = await tool_executor_node(state, config)
        msgs = result["messages"]
        assert len(msgs) == 1
        assert "Malformed" in msgs[0].content

    async def test_mixed_valid_and_invalid_calls(self):
        reg = ToolRegistry()
        reg.register("echo", _echo_tool, callable_to_schema(_echo_tool))

        state = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "tc-1", "function": {"name": "echo", "arguments": '{"msg": "hi"}'}},
                        {"id": "tc-2", "function": {"name": "missing", "arguments": "{}"}},
                    ],
                }
            ]
        }
        config = {"configurable": {"tool_registry": reg}}

        result = await tool_executor_node(state, config)
        msgs = result["messages"]
        assert len(msgs) == 2
        assert msgs[0].content == "echo: hi"
        assert "not found" in msgs[1].content
