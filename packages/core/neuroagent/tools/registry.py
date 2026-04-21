"""Tool registry for agent tool discovery and execution."""

import inspect
import json
from collections.abc import Callable
from typing import Any


def callable_to_schema(tool: Callable[..., Any]) -> dict[str, Any]:
    """Convert a plain callable to an OpenAI-style function schema."""
    sig = inspect.signature(tool)
    props: dict[str, dict[str, str]] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        ann = param.annotation
        if ann is str:
            props[name] = {"type": "string"}
        elif ann is int:
            props[name] = {"type": "integer"}
        else:
            props[name] = {"type": "string"}
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {
        "name": tool.__name__,
        "description": (tool.__doc__ or "").strip(),
        "parameters": {
            "type": "object",
            "properties": props,
            "required": required,
        },
    }


class ToolRegistry:
    """In-memory registry for agent-callable tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., Any]] = {}
        self._schemas: dict[str, dict[str, Any]] = {}

    def register(
        self,
        name: str,
        callable_: Callable[..., Any],
        schema: dict[str, Any],
    ) -> None:
        """Register a tool with its OpenAI-style function schema."""
        self._tools[name] = callable_
        self._schemas[name] = schema

    def get(self, name: str) -> Callable[..., Any]:
        """Look up a registered tool by name."""
        return self._tools[name]

    def get_schema(self, name: str) -> dict[str, Any]:
        """Look up a tool's schema by name."""
        return self._schemas[name]

    def list_tools(self) -> list[str]:
        """Return all registered tool names."""
        return sorted(self._tools.keys())

    def has_tool(self, name: str) -> bool:
        """Check whether a tool is registered."""
        return name in self._tools

    async def execute(self, name: str, arguments: str) -> Any:
        """Execute a registered tool by name with JSON-encoded arguments."""
        tool = self._tools.get(name)
        if tool is None:
            return f"Tool '{name}' not found"
        try:
            args = json.loads(arguments)
            if inspect.iscoroutinefunction(tool):
                return await tool(**args)
            return tool(**args)
        except Exception as exc:
            return f"Error: {exc}"

    def tool_defs(self) -> list[dict[str, Any]]:
        """Return all registered tools as OpenAI-style tool definitions."""
        return [
            {"type": "function", "function": self._schemas[name]}
            for name in sorted(self._schemas)
        ]
