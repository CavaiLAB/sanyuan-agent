"""MCP client: handshake, tool discovery/invocation, ToolRegistry bridging.

``McpClient`` (013) speaks JSON-RPC to one MCP server over an injected
transport; ``register_mcp_tools`` exposes the discovered tools through the
normal ``ToolRegistry`` so an ``AgentSpec.tools`` reference picks them up.
A scripted in-memory transport drives the protocol tests; ``StdioTransport``
is exercised against a real subprocess speaking the wire format.
"""

from __future__ import annotations

import json
import sys
from collections import deque
from collections.abc import Callable
from typing import Any

import pytest
from sanyuan_agent.mcp import (
    McpClient,
    McpError,
    McpToolDef,
    McpToolError,
    McpTransportError,
    StdioTransport,
    mcp_tool_schema,
    qualified_tool_name,
    register_mcp_tools,
)
from sanyuan_agent.tools.registry import ToolRegistry

Handler = Callable[[dict[str, Any]], dict[str, Any]]


class FakeTransport:
    """Scripted transport: each request method maps to a response factory."""

    def __init__(self, handlers: dict[str, Handler] | None = None) -> None:
        self.handlers = handlers or {}
        self.sent: list[dict[str, Any]] = []
        self._inbox: deque[dict[str, Any]] = deque()
        self.closed = False

    def push(self, message: dict[str, Any]) -> None:
        """Queue a server-initiated message ahead of the next response."""
        self._inbox.append(message)

    async def send(self, message: dict[str, Any]) -> None:
        self.sent.append(message)
        method = message.get("method")
        if "id" in message and method in self.handlers:
            self._inbox.append(self.handlers[method](message))

    async def receive(self) -> dict[str, Any]:
        assert self._inbox, "client waited for a message the server never sent"
        return self._inbox.popleft()

    async def close(self) -> None:
        self.closed = True


def _result(request: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request["id"], "result": result}


def _initialize_handler(request: dict[str, Any]) -> dict[str, Any]:
    return _result(
        request,
        {
            "protocolVersion": request["params"]["protocolVersion"],
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "docs", "version": "1.0.0"},
        },
    )


ECHO_TOOL = {
    "name": "echo",
    "description": "Echo text back",
    "inputSchema": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
}


class TestInitialize:
    async def test_handshake_request_and_initialized_notification(self) -> None:
        transport = FakeTransport({"initialize": _initialize_handler})
        client = McpClient(transport)

        result = await client.initialize()

        request = transport.sent[0]
        assert request["method"] == "initialize"
        assert request["params"]["protocolVersion"]
        assert request["params"]["clientInfo"]["name"] == "neuroagent"
        notification = transport.sent[1]
        assert notification == {"jsonrpc": "2.0", "method": "notifications/initialized"}
        assert result["serverInfo"]["name"] == "docs"
        assert client.server_name == "docs"
        assert client.initialized

    async def test_initialize_error_raises_mcp_error(self) -> None:
        def fail(request: dict[str, Any]) -> dict[str, Any]:
            return {
                "jsonrpc": "2.0",
                "id": request["id"],
                "error": {"code": -32600, "message": "unsupported version"},
            }

        client = McpClient(FakeTransport({"initialize": fail}))

        with pytest.raises(McpError) as excinfo:
            await client.initialize()
        assert excinfo.value.code == -32600
        assert not client.initialized


class TestListTools:
    async def test_parses_tool_definitions(self) -> None:
        def list_handler(request: dict[str, Any]) -> dict[str, Any]:
            return _result(request, {"tools": [ECHO_TOOL, {"name": "bare"}]})

        client = McpClient(FakeTransport({"tools/list": list_handler}))

        tools = await client.list_tools()

        assert tools[0] == McpToolDef(
            name="echo",
            description="Echo text back",
            input_schema=ECHO_TOOL["inputSchema"],
        )
        # Missing description/inputSchema fall back to empty values.
        assert tools[1] == McpToolDef(name="bare", description="", input_schema={})

    async def test_follows_pagination_cursor(self) -> None:
        def list_handler(request: dict[str, Any]) -> dict[str, Any]:
            if request["params"].get("cursor") == "page2":
                return _result(request, {"tools": [{"name": "second"}]})
            return _result(request, {"tools": [{"name": "first"}], "nextCursor": "page2"})

        transport = FakeTransport({"tools/list": list_handler})
        client = McpClient(transport)

        tools = await client.list_tools()

        assert [t.name for t in tools] == ["first", "second"]
        assert transport.sent[1]["params"] == {"cursor": "page2"}


class TestCallTool:
    async def test_returns_joined_text_content(self) -> None:
        def call_handler(request: dict[str, Any]) -> dict[str, Any]:
            return _result(
                request,
                {
                    "content": [
                        {"type": "text", "text": "part one"},
                        {"type": "image", "data": "..."},
                        {"type": "text", "text": "part two"},
                    ]
                },
            )

        client = McpClient(FakeTransport({"tools/call": call_handler}))

        assert await client.call_tool("echo", {"text": "hi"}) == "part one\npart two"

    async def test_is_error_result_raises_tool_error(self) -> None:
        def call_handler(request: dict[str, Any]) -> dict[str, Any]:
            return _result(
                request,
                {"content": [{"type": "text", "text": "boom"}], "isError": True},
            )

        client = McpClient(FakeTransport({"tools/call": call_handler}))

        with pytest.raises(McpToolError, match="boom"):
            await client.call_tool("echo", {"text": "hi"})

    async def test_jsonrpc_error_raises_mcp_error(self) -> None:
        def call_handler(request: dict[str, Any]) -> dict[str, Any]:
            return {
                "jsonrpc": "2.0",
                "id": request["id"],
                "error": {"code": -32602, "message": "unknown tool", "data": {"x": 1}},
            }

        client = McpClient(FakeTransport({"tools/call": call_handler}))

        with pytest.raises(McpError) as excinfo:
            await client.call_tool("nope", {})
        assert excinfo.value.code == -32602
        assert excinfo.value.data == {"x": 1}

    async def test_skips_notifications_and_answers_ping_while_waiting(self) -> None:
        def call_handler(request: dict[str, Any]) -> dict[str, Any]:
            return _result(request, {"content": [{"type": "text", "text": "done"}]})

        transport = FakeTransport({"tools/call": call_handler})
        # Server traffic queued ahead of the response: both must be consumed.
        transport.push({"jsonrpc": "2.0", "method": "notifications/progress", "params": {}})
        transport.push({"jsonrpc": "2.0", "id": "srv-1", "method": "ping"})
        client = McpClient(transport)

        assert await client.call_tool("echo", {"text": "hi"}) == "done"
        pong = next(m for m in transport.sent if m.get("id") == "srv-1")
        assert pong["result"] == {}


class TestBridge:
    @staticmethod
    def _client() -> tuple[McpClient, FakeTransport]:
        def list_handler(request: dict[str, Any]) -> dict[str, Any]:
            return _result(request, {"tools": [ECHO_TOOL, {"name": "search-docs"}]})

        def call_handler(request: dict[str, Any]) -> dict[str, Any]:
            name = request["params"]["name"]
            args = json.dumps(request["params"]["arguments"], sort_keys=True)
            return _result(
                request,
                {"content": [{"type": "text", "text": f"{name}:{args}"}]},
            )

        transport = FakeTransport(
            {
                "initialize": _initialize_handler,
                "tools/list": list_handler,
                "tools/call": call_handler,
            }
        )
        return McpClient(transport), transport

    async def test_registers_qualified_tools_with_schema(self) -> None:
        client, _ = self._client()
        await client.initialize()
        registry = ToolRegistry()

        names = await register_mcp_tools(registry, client)

        assert names == ["mcp__docs__echo", "mcp__docs__search-docs"]
        assert registry.list_tools() == names
        schema = registry.get_schema("mcp__docs__echo")
        assert schema["name"] == "mcp__docs__echo"
        assert schema["description"] == "Echo text back"
        assert schema["parameters"] == ECHO_TOOL["inputSchema"]
        # No inputSchema → minimal object schema; no description → fallback.
        bare = registry.get_schema("mcp__docs__search-docs")
        assert bare["parameters"] == {"type": "object", "properties": {}}
        assert "search-docs" in bare["description"]

    async def test_registry_execute_round_trips_to_server(self) -> None:
        client, _ = self._client()
        await client.initialize()
        registry = ToolRegistry()
        await register_mcp_tools(registry, client)

        result = await registry.execute("mcp__docs__echo", json.dumps({"text": "hi"}))

        assert result == 'echo:{"text": "hi"}'

    async def test_explicit_server_name_and_sanitization(self) -> None:
        client, _ = self._client()
        await client.initialize()

        assert qualified_tool_name("my docs!", "fetch page") == "mcp__my_docs__fetch_page"
        names = await register_mcp_tools(ToolRegistry(), client, server_name="alias")
        assert names[0] == "mcp__alias__echo"

    def test_schema_fallbacks(self) -> None:
        tool = McpToolDef(name="t", description="", input_schema={})
        schema = mcp_tool_schema("mcp__s__t", tool)
        assert schema == {
            "name": "mcp__s__t",
            "description": "MCP tool 't'",
            "parameters": {"type": "object", "properties": {}},
        }


MINI_SERVER = """
import json, sys

def reply(obj):
    sys.stdout.write(json.dumps(obj) + "\\n")
    sys.stdout.flush()

for line in sys.stdin:
    msg = json.loads(line)
    method = msg.get("method")
    if method == "initialize":
        reply({"jsonrpc": "2.0", "id": msg["id"], "result": {
            "protocolVersion": msg["params"]["protocolVersion"],
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "mini", "version": "0.0.1"}}})
    elif method == "notifications/initialized":
        pass
    elif method == "tools/list":
        reply({"jsonrpc": "2.0", "id": msg["id"], "result": {"tools": [
            {"name": "echo", "description": "Echo text back",
             "inputSchema": {"type": "object",
                             "properties": {"text": {"type": "string"}},
                             "required": ["text"]}}]}})
    elif method == "tools/call":
        text = msg["params"]["arguments"]["text"]
        reply({"jsonrpc": "2.0", "id": msg["id"], "result": {
            "content": [{"type": "text", "text": "echo: " + text}]}})
    else:
        reply({"jsonrpc": "2.0", "id": msg.get("id"),
               "error": {"code": -32601, "message": "method not supported"}})
"""


class TestStdioTransport:
    async def test_full_client_flow_over_subprocess(self) -> None:
        transport = StdioTransport(sys.executable, ["-c", MINI_SERVER])
        await transport.start()
        client = McpClient(transport)
        try:
            await client.initialize()
            assert client.server_name == "mini"

            registry = ToolRegistry()
            names = await register_mcp_tools(registry, client)
            assert names == ["mcp__mini__echo"]

            result = await registry.execute(
                "mcp__mini__echo", json.dumps({"text": "hello"})
            )
            assert result == "echo: hello"
        finally:
            await client.close()

    async def test_spawn_failure_raises_transport_error(self) -> None:
        transport = StdioTransport("/nonexistent/mcp-server-binary")
        with pytest.raises(McpTransportError, match="failed to spawn"):
            await transport.start()

    async def test_server_eof_raises_transport_error(self) -> None:
        transport = StdioTransport(sys.executable, ["-c", "pass"])
        await transport.start()
        try:
            with pytest.raises(McpTransportError, match="closed the connection"):
                await transport.receive()
        finally:
            await transport.close()

    async def test_not_started_raises(self) -> None:
        transport = StdioTransport(sys.executable)
        with pytest.raises(McpTransportError, match="not started"):
            await transport.send({"jsonrpc": "2.0"})
