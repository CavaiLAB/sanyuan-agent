"""Long-term memory tools as a :class:`ToolProvider` (``remember_this`` /
``recall_memories``).

The builder path already exists (:func:`neuroagent.memory.longterm.tools.build_memory_tools`,
which registers the two tools directly). What was missing is a provider/spec
wrapper so the same tools are selectable through ``AgentSpec.providers`` and
carry descriptive metadata, exactly like the rest of the tool layer. This adds
**no behavior and no schema**: each callable and its reflected schema come
straight from the builder (built into a scratch registry, re-emitted verbatim).

``remember_this`` mutates long-term state → ``side_effect="write"`` +
``policy_hint="confirm"``; ``recall_memories`` is read-only. Per the provider
contract that metadata is descriptive and never gates execution on its own.
"""

from __future__ import annotations

from typing import Any

from neuroagent.memory.longterm.store import LongtermMemoryStore
from neuroagent.memory.longterm.tools import build_memory_tools
from neuroagent.tools.provider import ToolSpec, spec_from_callable
from neuroagent.tools.registry import ToolRegistry

# Per-tool descriptive metadata; unknown names fall back to read-only.
_MEMORY_TOOL_META: dict[str, dict[str, Any]] = {
    "remember_this": {
        "capabilities": ("memory", "write"),
        "side_effect": "write",
        "policy_hint": "confirm",
    },
    "recall_memories": {
        "capabilities": ("memory", "recall"),
        "side_effect": "read",
    },
}


class MemoryToolProvider:
    """Thin ToolProvider over :func:`build_memory_tools`.

    Binds the long-term :class:`LongtermMemoryStore` (and the request-scoped
    *tenant_id* / *user_id* the builder captures), then re-emits the registered
    callables + schemas as specs with ``category="memory"`` metadata.
    """

    def __init__(
        self,
        store: LongtermMemoryStore,
        *,
        tenant_id: str = "",
        user_id: str = "",
        project_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self._store = store
        self._tenant_id = tenant_id
        self._user_id = user_id
        self._project_id = project_id
        self._session_id = session_id

    def provide(self) -> list[ToolSpec]:
        scratch = ToolRegistry()
        build_memory_tools(
            scratch,
            self._store,
            tenant_id=self._tenant_id,
            user_id=self._user_id,
            project_id=self._project_id,
            session_id=self._session_id,
        )
        specs: list[ToolSpec] = []
        for name in scratch.list_tools():
            meta = _MEMORY_TOOL_META.get(name, {"capabilities": ("memory",), "side_effect": "read"})
            specs.append(
                spec_from_callable(
                    scratch.get(name),
                    name=name,
                    schema=scratch.get_schema(name),
                    category="memory",
                    **meta,
                )
            )
        return specs
