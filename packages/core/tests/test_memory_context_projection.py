"""The complete context/projection path works from the standalone SDK."""

from __future__ import annotations

import stat
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest
from neuroagent.memory import (
    MemoryContext,
    MemoryLevel,
    materialize_memory_markdown,
    memory_visible_in_context,
    render_memory_markdown,
    validate_memory_context,
)


@dataclass(frozen=True)
class _Record:
    id: str
    content: str
    scope: str = "private"
    level: MemoryLevel = MemoryLevel.PERSONAL
    project_id: str | None = None
    session_id: str | None = None
    created_at: datetime | str | None = datetime(2026, 1, 1, tzinfo=UTC)


def test_context_validation_and_ancestor_visibility() -> None:
    context = MemoryContext(project_id="p1", session_id="s1")
    assert memory_visible_in_context(MemoryLevel.PERSONAL, context=context)
    assert memory_visible_in_context(
        MemoryLevel.PROJECT,
        memory_project_id="p1",
        context=context,
    )
    assert memory_visible_in_context(
        MemoryLevel.SESSION,
        memory_project_id="p1",
        memory_session_id="s1",
        context=context,
    )
    assert not memory_visible_in_context(
        MemoryLevel.PROJECT,
        memory_project_id="other",
        context=context,
    )
    assert not memory_visible_in_context(
        MemoryLevel.SESSION,
        memory_session_id="other",
        context=context,
    )

    with pytest.raises(ValueError, match="requires project_id"):
        validate_memory_context(MemoryLevel.PROJECT)
    with pytest.raises(ValueError, match="cannot carry"):
        validate_memory_context(MemoryLevel.PERSONAL, project_id="p1")


def test_projection_render_and_materialize_without_app_dependency(tmp_path) -> None:
    projection = render_memory_markdown(
        [
            _Record(id="personal", content="prefers   concise\nanswers"),
            _Record(
                id="project",
                content="uses pytest",
                level=MemoryLevel.PROJECT,
                project_id="p1",
            ),
            _Record(
                id="session",
                content="current task is memory",
                level=MemoryLevel.SESSION,
                project_id="p1",
                session_id="s1",
            ),
        ]
    )

    assert "## 个人记忆" in projection.content
    assert "## 项目记忆" in projection.content
    assert "## 会话记忆" in projection.content
    assert "prefers concise answers" in projection.content
    assert set(projection.line_by_id) == {"personal", "project", "session"}
    for memory_id, line in projection.line_by_id.items():
        assert f"memory:{memory_id}" in projection.content.splitlines()[line - 1]

    path = materialize_memory_markdown(tmp_path / "system/memory/MEMORY.md", projection)
    assert path.read_text(encoding="utf-8") == projection.content
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
