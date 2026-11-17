"""Portable, read-only ``MEMORY.md`` projection helpers.

The persistence backend remains the source of truth.  These helpers only render
and atomically materialize a deterministic view, and intentionally depend on
the Python standard library alone.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from sanyuan_agent.memory.context import MemoryLevel, validate_memory_context


class ProjectableMemory(Protocol):
    """Minimal record shape accepted by :func:`render_memory_markdown`."""

    id: str
    content: str
    scope: str
    level: MemoryLevel | str
    project_id: str | None
    session_id: str | None
    created_at: datetime | str | None


@dataclass(frozen=True, slots=True)
class MemoryProjectionEntry:
    """Stable pointer from a memory record to its 1-based projection line."""

    memory_id: str
    line: int
    level: MemoryLevel
    project_id: str | None = None
    session_id: str | None = None


@dataclass(frozen=True, slots=True)
class MemoryProjection:
    """Rendered ``MEMORY.md`` content plus record-to-line pointers."""

    content: str
    entries: tuple[MemoryProjectionEntry, ...]

    @property
    def line_by_id(self) -> dict[str, int]:
        return {entry.memory_id: entry.line for entry in self.entries}


def stable_memory_token(value: str | None, *, fallback: str) -> str:
    """Return a path-safe, non-reversible token for an untrusted context id."""
    raw = (value or fallback).strip() or fallback
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _one_line(content: str) -> str:
    return " ".join(content.replace("\x00", "").split())


def _sort_key(memory: ProjectableMemory) -> tuple[str, str]:
    created = memory.created_at
    if isinstance(created, datetime):
        created_key = created.isoformat()
    else:
        created_key = created or ""
    return created_key, memory.id


def render_memory_markdown(memories: list[ProjectableMemory]) -> MemoryProjection:
    """Render deterministic personal/project/session sections."""
    lines = [
        "# MEMORY.md",
        "",
        "> iClaw 记忆的只读投影; 真相源在记忆服务, 请勿直接编辑此文件。",
        "",
    ]
    entries: list[MemoryProjectionEntry] = []

    def append_memory(memory: ProjectableMemory) -> None:
        level = validate_memory_context(
            memory.level,
            project_id=memory.project_id,
            session_id=memory.session_id,
        )
        lines.append(f"- [{memory.scope}] {_one_line(memory.content)} <!-- memory:{memory.id} -->")
        entries.append(
            MemoryProjectionEntry(
                memory_id=memory.id,
                line=len(lines),
                level=level,
                project_id=memory.project_id,
                session_id=memory.session_id,
            )
        )

    ordered = sorted(memories, key=_sort_key)
    personal = [item for item in ordered if MemoryLevel(item.level) is MemoryLevel.PERSONAL]
    projects = [item for item in ordered if MemoryLevel(item.level) is MemoryLevel.PROJECT]
    sessions = [item for item in ordered if MemoryLevel(item.level) is MemoryLevel.SESSION]

    lines.append("## 个人记忆")
    if personal:
        for memory in personal:
            append_memory(memory)
    else:
        lines.append("- (无)")

    lines.extend(["", "## 项目记忆"])
    if projects:
        for project_id in sorted({item.project_id or "" for item in projects}):
            token = stable_memory_token(project_id, fallback="default-project")
            lines.extend(["", f"### project/{token}"])
            for memory in projects:
                if (memory.project_id or "") == project_id:
                    append_memory(memory)
    else:
        lines.append("- (无)")

    lines.extend(["", "## 会话记忆"])
    if sessions:
        for session_id in sorted({item.session_id or "" for item in sessions}):
            token = stable_memory_token(session_id, fallback="default-session")
            lines.extend(["", f"### session/{token}"])
            for memory in sessions:
                if (memory.session_id or "") == session_id:
                    append_memory(memory)
    else:
        lines.append("- (无)")

    return MemoryProjection(content="\n".join(lines) + "\n", entries=tuple(entries))


def materialize_memory_markdown(
    path: str | Path,
    projection: MemoryProjection | str,
) -> Path:
    """Atomically write a mode-0600 projection and return its resolved path."""
    target = Path(path).expanduser()
    content = projection.content if isinstance(projection, MemoryProjection) else projection
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(content, encoding="utf-8")
        os.chmod(tmp, 0o600)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    return target
