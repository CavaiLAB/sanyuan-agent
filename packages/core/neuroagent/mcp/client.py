"""Minimal MCP client: initialize / tools/list / tools/call over a transport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .transport import McpTransport

PROTOCOL_VERSION = "2025-03-26"


class McpError(Exception):
    """JSON-RPC error returned by the MCP server."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(f"MCP error {code}: {message}")
        self.code = code
        self.message = message
        self.data = data


class McpToolError(Exception):
    """The MCP tool ran and reported a failure (``isError: true``)."""


@dataclass(frozen=True)
class McpToolDef:
    """A tool advertised by an MCP server via ``tools/list``."""

    name: str
    description: str
    input_schema: dict[str, Any]


class McpClient:
    """Sequential JSON-RPC client for a single MCP server connection.

    Requests are issued one at a time; server-initiated traffic that arrives
    while waiting for a response is handled inline (``ping`` answered,
    notifications skipped, other server requests rejected with -32601).
    """

    def __init__(
        self,
        transport: McpTransport,
        *,
        client_name: str = "neuroagent",
        client_version: str = "0.1.0",
        protocol_version: str = PROTOCOL_VERSION,
    ) -> None:
        self._transport = transport
        self._client_name = client_name
        self._client_version = client_version
        self._protocol_version = protocol_version
        self._next_id = 0
        self.server_name = ""
        self.initialized = False

    async def initialize(self) -> dict[str, Any]:
        """Run the MCP handshake; returns the server's initialize result."""
        result = await self._request(
            "initialize",
            {
                "protocolVersion": self._protocol_version,
                "capabilities": {},
                "clientInfo": {
                    "name": self._client_name,
                    "version": self._client_version,
                },
            },
        )
        await self._transport.send(
            {"jsonrpc": "2.0", "method": "notifications/initialized"}
        )
        server_info = result.get("serverInfo", {})
        self.server_name = str(server_info.get("name", ""))
        self.initialized = True
        return result

    async def list_tools(self) -> list[McpToolDef]:
        """Fetch every tool the server advertises, following pagination."""
        tools: list[McpToolDef] = []
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {"cursor": cursor} if cursor else {}
            result = await self._request("tools/list", params)
            for raw in result.get("tools", []):
                tools.append(
                    McpToolDef(
                        name=str(raw["name"]),
                        description=str(raw.get("description", "")),
                        input_schema=dict(raw.get("inputSchema", {})),
                    )
                )
            cursor = result.get("nextCursor")
            if not cursor:
                return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Invoke a server tool; returns the concatenated text content.

        Raises :class:`McpToolError` when the tool reports ``isError: true``
        (so the registry's normal tool-failure path applies).
        """
        result = await self._request(
            "tools/call", {"name": name, "arguments": arguments}
        )
        text = _content_text(result.get("content", []))
        if result.get("isError"):
            raise McpToolError(text or f"MCP tool {name!r} failed")
        return text

    async def close(self) -> None:
        await self._transport.close()

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._next_id += 1
        request_id = self._next_id
        await self._transport.send(
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        )
        while True:
            message = await self._transport.receive()
            if message.get("id") == request_id and "method" not in message:
                error = message.get("error")
                if error is not None:
                    raise McpError(
                        int(error.get("code", -32603)),
                        str(error.get("message", "unknown error")),
                        error.get("data"),
                    )
                result = message.get("result", {})
                return result if isinstance(result, dict) else {}
            if "method" in message and "id" in message:
                await self._answer_server_request(message)
            # Notifications and stale responses are skipped.

    async def _answer_server_request(self, message: dict[str, Any]) -> None:
        if message.get("method") == "ping":
            await self._transport.send(
                {"jsonrpc": "2.0", "id": message["id"], "result": {}}
            )
            return
        await self._transport.send(
            {
                "jsonrpc": "2.0",
                "id": message["id"],
                "error": {"code": -32601, "message": "method not supported"},
            }
        )


def _content_text(content: list[Any]) -> str:
    """Join the text blocks of a tool result; non-text blocks are dropped."""
    parts = [
        str(block.get("text", ""))
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n".join(parts)
