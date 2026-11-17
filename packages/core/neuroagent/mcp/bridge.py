"""Bridge MCP server tools into the agent's :class:`ToolRegistry`."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from neuroagent.tools.registry import ToolRegistry

from .client import McpClient, McpToolDef


def _slug(name: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z_-]+", "_", name).strip("_")
    return slug or "unnamed"


def qualified_tool_name(server_name: str, tool_name: str) -> str:
    """Registry name for an MCP tool: ``mcp__<server>__<tool>``."""
    return f"mcp__{_slug(server_name)}__{_slug(tool_name)}"


def mcp_tool_schema(qualified_name: str, tool: McpToolDef) -> dict[str, Any]:
    """OpenAI-style function schema for an MCP tool definition.

    MCP's ``inputSchema`` is already JSON Schema, so it maps straight onto the
    ``parameters`` field.
    """
    parameters = tool.input_schema or {"type": "object", "properties": {}}
    return {
        "name": qualified_name,
        "description": tool.description or f"MCP tool {tool.name!r}",
        "parameters": parameters,
    }


async def register_mcp_tools(
    registry: ToolRegistry,
    client: McpClient,
    *,
    server_name: str = "",
) -> list[str]:
    """Discover the server's tools and register each into *registry*.

    Returns the qualified names (``mcp__<server>__<tool>``) so callers can
    reference them from an ``AgentSpec.tools`` list like any built-in tool.
    """
    server = server_name or client.server_name or "server"
    registered: list[str] = []
    for tool in await client.list_tools():
        qualified = qualified_tool_name(server, tool.name)
        registry.register(
            qualified,
            _make_proxy(client, tool, qualified),
            mcp_tool_schema(qualified, tool),
        )
        registered.append(qualified)
    return registered


async def make_mcp_tools(
    client: McpClient,
    *,
    server_name: str = "",
) -> list[Callable[..., Any]]:
    """Adapt the server's tools into plain callables (no registry needed).

    Each callable carries its JSON schema via ``__tool_schema__`` — honoured
    by the agent builder's tool-definition reflection — so MCP tools slot
    directly into ``build_simple_agent(tools=...)`` like any builtin tool.
    """
    server = server_name or client.server_name or "server"
    return [
        _make_proxy(client, tool, qualified_tool_name(server, tool.name))
        for tool in await client.list_tools()
    ]


def _make_proxy(
    client: McpClient, tool: McpToolDef, qualified: str
) -> Callable[..., Any]:
    """An LLM-callable that forwards keyword arguments to the MCP server.

    The proxy's ``**arguments`` signature can't be reflected into the tool's
    real JSON schema, so the schema rides along verbatim in
    ``__tool_schema__`` for tool-definition builders to use.
    """

    async def proxy(**arguments: Any) -> str:
        return await client.call_tool(tool.name, arguments)

    proxy.__name__ = qualified
    proxy.__doc__ = tool.description or f"MCP tool {tool.name!r}"
    proxy.__tool_schema__ = mcp_tool_schema(qualified, tool)  # type: ignore[attr-defined]
    return proxy
