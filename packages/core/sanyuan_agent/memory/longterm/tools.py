"""Memory tools — remember_this (store) and recall_memories (retrieve)."""

from __future__ import annotations

import logging
from typing import Any

from sanyuan_agent.memory.context import MemoryContext, validate_memory_context
from sanyuan_agent.memory.longterm.store import LongtermMemoryStore, ScopeKind
from sanyuan_agent.tools.registry import ToolRegistry, callable_to_schema

logger = logging.getLogger(__name__)


def build_memory_tools(
    registry: ToolRegistry,
    store: LongtermMemoryStore,
    *,
    tenant_id: str = "",
    user_id: str = "",
    project_id: str | None = None,
    session_id: str | None = None,
) -> None:
    """Register ``remember_this`` and ``recall_memories`` tools.

    Both tools capture *tenant_id* and *user_id* from the enclosing scope
    so agents don't need to pass them explicitly.
    """

    async def remember_this(
        content: str,
        scope: str = "private",
        level: str = "personal",
        importance: str = "0.5",
    ) -> dict[str, Any]:
        """Persist a piece of information to long-term memory.

        Args:
            content: The text to remember.
            scope: Access scope — "private" (only you), "team", or "tenant".
            level: Context level — "personal", "project", or "session".
            importance: Float 0.0-1.0 indicating how important this memory is.
        """
        try:
            scope_enum = ScopeKind(scope)
        except ValueError:
            return {
                "success": False,
                "error": f"Invalid scope '{scope}'. Use private/team/tenant.",
            }

        owner_id = user_id if scope_enum == ScopeKind.PRIVATE else None
        memory_project_id = project_id if level in {"project", "session"} else None
        memory_session_id = session_id if level == "session" else None
        try:
            level_enum = validate_memory_context(
                level,
                project_id=memory_project_id,
                session_id=memory_session_id,
            )
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        try:
            entry = await store.store(
                tenant_id=tenant_id,
                content=content,
                scope=scope_enum,
                scope_owner_id=owner_id,
                importance=float(importance),
                level=level_enum,
                project_id=memory_project_id,
                session_id=memory_session_id,
            )
            return {
                "success": True,
                "memory_id": entry.id,
                "scope": entry.scope.value,
                "level": entry.level.value,
            }
        except Exception as exc:
            logger.warning("remember_this failed: %s", exc)
            return {"success": False, "error": str(exc)}

    async def recall_memories(
        query: str,
        scope: str = "private,team,tenant",
        limit: str = "5",
        importance_threshold: str = "0.3",
    ) -> dict[str, Any]:
        """Retrieve relevant memories by semantic similarity.

        Args:
            query: Natural language query to search memories.
            scope: Comma-separated scopes to search (private,team,tenant).
            limit: Maximum number of results.
            importance_threshold: Minimum importance score (0.0-1.0).
        """
        scopes = []
        for s in scope.split(","):
            s = s.strip()
            if s:
                try:
                    scopes.append(ScopeKind(s))
                except ValueError:
                    pass
        if not scopes:
            scopes = [ScopeKind.PRIVATE, ScopeKind.TEAM, ScopeKind.TENANT]

        try:
            context = MemoryContext(project_id=project_id, session_id=session_id)
            entries = await store.retrieve_semantic(
                tenant_id=tenant_id,
                query=query,
                scope=scopes,
                scope_owner_id=user_id or None,
                limit=int(limit),
                importance_threshold=float(importance_threshold),
                context=context,
            )
            return {
                "success": True,
                "count": len(entries),
                "memories": [
                    {
                        "id": e.id,
                        "content": e.content,
                        "scope": e.scope.value,
                        "level": e.level.value,
                        "project_id": e.project_id,
                        "session_id": e.session_id,
                        "importance": e.importance,
                        "similarity": e.similarity_score,
                        "created_at": e.created_at,
                    }
                    for e in entries
                ],
            }
        except Exception as exc:
            logger.warning("recall_memories failed: %s", exc)
            return {"success": False, "error": str(exc)}

    _tools = [
        ("remember_this", remember_this),
        ("recall_memories", recall_memories),
    ]
    for name, fn in _tools:
        registry.register(name, fn, callable_to_schema(fn))  # type: ignore[arg-type]
