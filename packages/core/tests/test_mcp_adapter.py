"""MCP tools through the agent-build path: schema carried, not reflected (016).

The bridge (013) registers MCP tools with correct schemas in the
``ToolRegistry``, but ``build_simple_agent(tools=...)`` — the
``AgentSpec.tools`` path — derives each tool's LLM definition by *reflecting
the callable's signature*, and an MCP proxy's ``**arguments`` signature
reflects to a wrong schema. This slice closes the gap: proxies carry their
JSON schema verbatim via ``__tool_schema__``, ``_to_tool_def`` honours it,
and ``make_mcp_tools`` adapts a server's tools into plain callables for
direct ``tools=[...]`` use.
"""

from __future__ import annotations

from typing import Any

from neuroagent.agents.simple import _to_tool_def
from neuroagent.mcp import McpToolDef, make_mcp_tools, register_mcp_tools
from neuroagent.tools.registry import ToolRegistry

WEATHER = McpToolDef(
    name="get-weather",
    description="Get the weather for a city.",
    input_schema={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "enum": ["c", "f"]},
        },
        "required": ["city"],
    },
)


class FakeClient:
    """Duck-typed stand-in for McpClient: list_tools / call_tool / server_name."""

    def __init__(self, tools: list[McpToolDef]) -> None:
        self._tools = tools
        self.server_name = "wx"
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> list[McpToolDef]:
        return self._tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        return f"called {name}"


class TestSchemaCarriedVerbatim:
    async def test_to_tool_def_uses_carried_schema_not_reflection(self) -> None:
        """The critical gap: reflecting a ``**arguments`` proxy loses the schema."""
        (tool,) = await make_mcp_tools(FakeClient([WEATHER]))

        tool_def = _to_tool_def(tool)

        assert tool_def == {
            "type": "function",
            "function": {
                "name": "mcp__wx__get-weather",
                "description": "Get the weather for a city.",
                "parameters": WEATHER.input_schema,
            },
        }

    def test_reflection_fallback_unchanged_without_carried_schema(self) -> None:
        def lookup(key: str, count: int = 3) -> str:
            """Look something up."""
            return key * count

        tool_def = _to_tool_def(lookup)

        assert tool_def["function"]["name"] == "lookup"
        assert set(tool_def["function"]["parameters"]["properties"]) == {"key", "count"}
        assert tool_def["function"]["parameters"]["required"] == ["key"]

    async def test_registered_proxies_also_carry_schema(self) -> None:
        registry = ToolRegistry()
        client = FakeClient([WEATHER])

        (name,) = await register_mcp_tools(registry, client)

        proxy = registry.get(name)
        assert getattr(proxy, "__tool_schema__", None) == registry.get_schema(name)


class TestMakeMcpTools:
    async def test_invokes_server_with_original_tool_name(self) -> None:
        client = FakeClient([WEATHER])
        (tool,) = await make_mcp_tools(client)

        result = await tool(city="Qingdao", unit="c")

        assert result == "called get-weather"
        # The LLM-facing name is qualified, but the server gets the original.
        assert client.calls == [("get-weather", {"city": "Qingdao", "unit": "c"})]

    async def test_explicit_server_name_overrides(self) -> None:
        (tool,) = await make_mcp_tools(FakeClient([WEATHER]), server_name="alias!")
        assert tool.__name__ == "mcp__alias__get-weather"

    async def test_empty_server_yields_no_tools(self) -> None:
        assert await make_mcp_tools(FakeClient([])) == []
