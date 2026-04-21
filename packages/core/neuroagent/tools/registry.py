"""Tool registry for agent tool discovery and execution."""

import inspect
import json
import traceback
from collections.abc import Callable
from typing import Any


class ToolStats:
    """Per-tool execution statistics."""

    def __init__(self) -> None:
        self.call_count: int = 0
        self.fail_count: int = 0
        self.last_error: str = ""


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
    """In-memory registry for agent-callable tools with circuit breaker."""

    def __init__(self, breaker_threshold: int = 3) -> None:
        self._tools: dict[str, Callable[..., Any]] = {}
        self._schemas: dict[str, dict[str, Any]] = {}
        self._stats: dict[str, ToolStats] = {}
        self.breaker_threshold = breaker_threshold

    def register(
        self,
        name: str,
        callable_: Callable[..., Any],
        schema: dict[str, Any],
    ) -> None:
        """Register a tool with its OpenAI-style function schema."""
        self._tools[name] = callable_
        self._schemas[name] = schema
        if name not in self._stats:
            self._stats[name] = ToolStats()

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

    def get_stats(self, name: str) -> ToolStats | None:
        """Return execution stats for a tool, or None if not registered."""
        return self._stats.get(name)

    def reset_stats(self, name: str) -> None:
        """Reset execution stats for a tool."""
        if name in self._stats:
            self._stats[name] = ToolStats()

    def _check_circuit_breaker(self, name: str) -> None:
        """Raise if circuit breaker is open for this tool."""
        from neuroagent.agents.errors import ToolCircuitBreakerError

        stats = self._stats.get(name)
        if stats is None:
            return
        if stats.fail_count >= self.breaker_threshold:
            raise ToolCircuitBreakerError(
                name,
                stats.fail_count,
                self.breaker_threshold,
            )

    async def execute(self, name: str, arguments: str) -> Any:
        """Execute a registered tool by name with JSON-encoded arguments."""
        from neuroagent.agents.errors import ToolExecutionError

        tool = self._tools.get(name)
        if tool is None:
            return f"Tool '{name}' not found"

        self._check_circuit_breaker(name)
        stats = self._stats[name]
        stats.call_count += 1

        try:
            args = json.loads(arguments)
            if inspect.iscoroutinefunction(tool):
                result = await tool(**args)
            else:
                result = tool(**args)
            # On success, reset fail count
            stats.fail_count = 0
            stats.last_error = ""
            return result
        except ToolExecutionError:
            raise
        except Exception as exc:
            stats.fail_count += 1
            stats.last_error = str(exc)
            tb = traceback.format_exc(limit=3)
            raise ToolExecutionError(
                name=name,
                original=exc,
                traceback_summary=tb,
            ) from exc

    def tool_defs(self) -> list[dict[str, Any]]:
        """Return all registered tools as OpenAI-style tool definitions."""
        return [
            {"type": "function", "function": self._schemas[name]}
            for name in sorted(self._schemas)
        ]
