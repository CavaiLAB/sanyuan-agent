"""MCP (Model Context Protocol) client — plug external tool servers in.

``StdioTransport`` spawns an MCP server subprocess (newline-delimited JSON-RPC
over stdio); ``McpClient`` speaks the protocol (initialize / tools/list /
tools/call); ``register_mcp_tools`` bridges the discovered tools into a
``ToolRegistry`` under ``mcp__<server>__<tool>`` names, where an
``AgentSpec.tools`` reference picks them up like any built-in tool.
"""

from .bridge import mcp_tool_schema, qualified_tool_name, register_mcp_tools
from .client import McpClient, McpError, McpToolDef, McpToolError
from .transport import McpTransport, McpTransportError, StdioTransport

__all__ = [
    "McpClient",
    "McpError",
    "McpToolDef",
    "McpToolError",
    "McpTransport",
    "McpTransportError",
    "StdioTransport",
    "mcp_tool_schema",
    "qualified_tool_name",
    "register_mcp_tools",
]
