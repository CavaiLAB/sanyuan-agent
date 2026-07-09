"""Tests for artifact_write tool."""

from typing import Any

import pytest
from neuroagent.artifacts.client import ArtifactRef
from neuroagent.tools.artifact_write import build_artifact_write_tool
from neuroagent.tools.registry import ToolRegistry


class FakeClient:
    """Fake StorageArtifactClient for unit tests."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create_artifact(
        self,
        tenant_id: str,
        content_bytes: bytes,
        content_type: str,
        filename: str,
        *,
        conversation_id: str | None = None,
        kind: str | None = None,
        origin: str | None = None,
        ownership: dict[str, str] | None = None,
    ) -> ArtifactRef:
        self.calls.append(
            {
                "tenant_id": tenant_id,
                "content_bytes": content_bytes,
                "content_type": content_type,
                "filename": filename,
                "conversation_id": conversation_id,
                "kind": kind,
                "origin": origin,
            }
        )
        return ArtifactRef(
            id="fake-id",
            url="https://example.com/fake",
            sha256="abc123",
            filename=filename,
            content_type=content_type,
        )


class TestArtifactWriteTool:
    async def test_tool_registers_in_registry(self) -> None:
        registry = ToolRegistry()
        fake = FakeClient()
        build_artifact_write_tool(registry, fake, tenant_id="t1")
        assert registry.has_tool("artifact_write")
        schema = registry.get_schema("artifact_write")
        assert schema["name"] == "artifact_write"
        assert "filename" in schema["parameters"]["properties"]
        assert "content" in schema["parameters"]["properties"]
        assert "content_type" in schema["parameters"]["properties"]

    async def test_tool_returns_artifact_ref(self) -> None:
        registry = ToolRegistry()
        fake = FakeClient()
        tool = build_artifact_write_tool(registry, fake, tenant_id="t1")

        result = await tool(filename="report.md", content="# Hello", content_type="text/markdown")

        assert result["artifact_id"] == "fake-id"
        assert result["download_url"] == "https://example.com/fake"
        assert result["filename"] == "report.md"
        assert result["content_type"] == "text/markdown"
        assert result["sha256"] == "abc123"

        assert len(fake.calls) == 1
        assert fake.calls[0]["tenant_id"] == "t1"
        assert fake.calls[0]["content_bytes"] == b"# Hello"
        assert fake.calls[0]["content_type"] == "text/markdown"
        assert fake.calls[0]["filename"] == "report.md"

    async def test_tool_scopes_artifact_to_conversation(self) -> None:
        # The saved artifact must carry conversation_id + produced/cloud tags so
        # the conversation file panel (which lists by conversation_id) surfaces it.
        registry = ToolRegistry()
        fake = FakeClient()
        tool = build_artifact_write_tool(
            registry, fake, tenant_id="t1", conversation_id="conv-42"
        )

        await tool(filename="out.md", content="x")

        assert fake.calls[0]["conversation_id"] == "conv-42"
        assert fake.calls[0]["kind"] == "produced"
        assert fake.calls[0]["origin"] == "cloud"

    async def test_tool_omits_conversation_when_absent(self) -> None:
        # No conversation id → pass None (client's GUID gate drops it from the wire).
        registry = ToolRegistry()
        fake = FakeClient()
        tool = build_artifact_write_tool(registry, fake, tenant_id="t1")

        await tool(filename="out.md", content="x")

        assert fake.calls[0]["conversation_id"] is None

    async def test_tool_uses_default_content_type(self) -> None:
        registry = ToolRegistry()
        fake = FakeClient()
        tool = build_artifact_write_tool(registry, fake, tenant_id="t1")

        await tool(filename="notes.txt", content="plain text")

        assert fake.calls[0]["content_type"] == "text/plain"

    async def test_tool_propagates_errors(self) -> None:
        registry = ToolRegistry()

        class FailingClient:
            async def create_artifact(self, **kwargs: Any) -> ArtifactRef:
                raise ValueError("storage down")

        tool = build_artifact_write_tool(registry, FailingClient(), tenant_id="t1")  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="storage down"):
            await tool(filename="x.txt", content="x")
