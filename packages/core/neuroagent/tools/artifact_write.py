"""Artifact-write tool builder — allows the agent to save files via Storage service."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from neuroagent.artifacts.client import ArtifactRef, StorageArtifactClient
from neuroagent.tools.registry import ToolRegistry, callable_to_schema

ArtifactWriteTool = Callable[[str, str, str], Awaitable[dict[str, str]]]


def build_artifact_write_tool(
    registry: ToolRegistry,
    client: StorageArtifactClient,
    *,
    tenant_id: str,
) -> ArtifactWriteTool:
    """Return a bound ``artifact_write`` tool for the given tenant.

    The tool is automatically registered in *registry* under
    ``"artifact_write"`` with its OpenAI-style function schema.
    """

    async def artifact_write(
        filename: str,
        content: str,
        content_type: str = "text/plain",
    ) -> dict[str, str]:
        ref: ArtifactRef = await client.create_artifact(
            tenant_id=tenant_id,
            content_bytes=content.encode("utf-8"),
            content_type=content_type,
            filename=filename,
        )
        return {
            "artifact_id": ref.id,
            "download_url": ref.url,
            "filename": ref.filename,
            "content_type": ref.content_type,
            "sha256": ref.sha256,
        }

    registry.register(
        "artifact_write",
        artifact_write,
        callable_to_schema(artifact_write),
    )
    return artifact_write
