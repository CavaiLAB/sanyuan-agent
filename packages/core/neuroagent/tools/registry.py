"""Tool registry for agent tool discovery and execution."""

import inspect
import json
import re
import traceback
from collections.abc import Callable
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints


class ToolStats:
    """Per-tool execution statistics."""

    def __init__(self) -> None:
        self.call_count: int = 0
        self.fail_count: int = 0
        self.last_error: str = ""


# JSON-schema scalar name per Python scalar type. ``bool`` precedes ``int``
# only matters for value inference (handled in ``_json_type_of_value``); the
# ``is`` lookups below are exact so order is irrelevant here.
_JSON_BY_TYPE: dict[type, str] = {
    str: "string",
    bool: "boolean",
    int: "integer",
    float: "number",
}
# Recover common builtins from an unresolved stringized annotation (future
# annotations + a ``get_type_hints`` failure), so a closure whose hints can't be
# resolved still types its plain params instead of collapsing to "string".
_JSON_BY_NAME: dict[str, str] = {
    "str": "string",
    "bool": "boolean",
    "int": "integer",
    "float": "number",
}
# Section headings that terminate a Google-style ``Args:`` block.
_DOC_SECTION_END: frozenset[str] = frozenset(
    {"returns", "return", "raises", "yields", "examples", "example", "note", "notes"}
)
_ARG_LINE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:\([^)]*\))?\s*:\s*(.*)$")


def _json_type_of_value(value: Any) -> str:
    """Best-effort JSON-schema type name for a literal / default *value*."""
    # bool is a subclass of int — check it first.
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, (list, tuple, set, frozenset)):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "string"


def _schema_for_annotation(ann: Any) -> dict[str, Any]:
    """Map a (resolved) type annotation to a JSON-schema property dict.

    Falls back to ``{"type": "string"}`` for anything it cannot model — the
    conservative shape the original reflector used for every parameter.
    """
    if ann is inspect.Parameter.empty or ann is None or ann is Any:
        return {"type": "string"}

    # Unresolved stringized annotation (future annotations + get_type_hints miss).
    if isinstance(ann, str):
        return {"type": _JSON_BY_NAME.get(ann, "string")}

    origin = get_origin(ann)

    # Literal[...] -> enum (type inferred from the first member).
    if origin is Literal:
        members = list(get_args(ann))
        prop: dict[str, Any] = {"enum": members}
        if members:
            prop["type"] = _json_type_of_value(members[0])
        return prop

    # Optional[X] / X | None / Union[...] -> unwrap a lone non-None arm.
    if origin is Union or origin is UnionType:
        arms = [a for a in get_args(ann) if a is not type(None)]
        if len(arms) == 1:
            return _schema_for_annotation(arms[0])
        return {"type": "string"}  # heterogeneous union: don't over-constrain

    # Containers.
    if origin in (list, tuple, set, frozenset) or ann in (list, tuple, set, frozenset):
        item_args = get_args(ann)
        items = _schema_for_annotation(item_args[0]) if item_args else {"type": "string"}
        return {"type": "array", "items": items}
    if origin is dict or ann is dict:
        return {"type": "object"}

    # Scalars.
    js = _JSON_BY_TYPE.get(ann)
    if js is not None:
        return {"type": js}

    # Unknown / complex (dataclass, pydantic, …): free-form string.
    return {"type": "string"}


def _parse_arg_docs(doc: str) -> dict[str, str]:
    """Extract per-arg descriptions from a Google-style ``Args:`` block.

    Best-effort: recognizes ``name: description`` lines (with an optional
    ``(type)`` annotation) under an ``Args:`` / ``Arguments:`` / ``Parameters:``
    heading, folding continuation lines into the preceding arg and stopping at
    the next section heading. Any parse difficulty yields ``{}``.
    """
    if not doc:
        return {}
    lines = doc.splitlines()
    start: int | None = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.endswith(":") and stripped[:-1].strip().lower() in (
            "args",
            "arguments",
            "parameters",
        ):
            start = i + 1
            break
    if start is None:
        return {}

    out: dict[str, str] = {}
    current: str | None = None
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            # A blank line ends the Args block (Google style); leading blanks
            # before the first entry are tolerated.
            if out:
                break
            continue
        if stripped.rstrip(":").lower() in _DOC_SECTION_END:
            break
        match = _ARG_LINE_RE.match(line)
        if match:
            current = match.group(1)
            out[current] = match.group(2).strip()
        elif current is not None:
            out[current] = f"{out[current]} {stripped}".strip()
    return out


def callable_to_schema(tool: Callable[..., Any]) -> dict[str, Any]:
    """Convert a plain callable to an OpenAI-style function schema.

    Reflects the callable's signature into a JSON-schema ``parameters`` object:

    - Resolves string annotations from ``from __future__ import annotations`` via
      :func:`typing.get_type_hints`, so future-annotated tools get precise types
      instead of collapsing to ``"string"``.
    - Models ``str`` / ``int`` / ``float`` / ``bool``, ``list`` / ``dict``
      containers, unwraps ``Optional[X]`` / ``X | None``, and renders
      ``Literal[...]`` as an ``enum``.
    - Carries each parameter's default into its schema and marks a parameter
      ``required`` only when it has no default.
    - Pulls per-parameter descriptions from a Google-style ``Args:`` docstring.

    ``self`` / ``cls`` and ``*args`` / ``**kwargs`` are skipped. Anything that
    cannot be resolved falls back to ``{"type": "string"}``.
    """
    sig = inspect.signature(tool)
    try:
        hints = get_type_hints(tool)
    except Exception:
        # Unresolvable forward refs (e.g. a closure over local-only names):
        # fall back to raw (possibly stringized) annotations per parameter.
        hints = {}
    arg_docs = _parse_arg_docs(inspect.getdoc(tool) or "")

    props: dict[str, dict[str, Any]] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        prop = _schema_for_annotation(hints.get(name, param.annotation))
        if arg_docs.get(name):
            prop["description"] = arg_docs[name]
        if param.default is inspect.Parameter.empty:
            required.append(name)
        elif isinstance(param.default, (str, int, float, bool)) or param.default is None:
            # Carry a JSON-serializable default so the model sees the fallback.
            prop["default"] = param.default
        props[name] = prop
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
