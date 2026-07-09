"""Artifact-write tool builder — allows the agent to save files via Storage service."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from neuroagent.artifacts.client import ArtifactRef, StorageArtifactClient
from neuroagent.tools.provider import ToolSpec, spec_from_callable
from neuroagent.tools.registry import ToolRegistry, callable_to_schema

ArtifactWriteTool = Callable[[str, str, str], Awaitable[dict[str, str]]]


def build_artifact_write_tool(
    registry: ToolRegistry,
    client: StorageArtifactClient,
    *,
    tenant_id: str,
    conversation_id: str = "",
) -> ArtifactWriteTool:
    """Return a bound ``artifact_write`` tool for the given tenant.

    The tool is automatically registered in *registry* under
    ``"artifact_write"`` with its OpenAI-style function schema.

    ``conversation_id`` scopes the saved artifact to the conversation so the
    file panel (which lists by ``conversation_id``) surfaces it — parity with
    the sandbox-harvest path. Non-GUID session ids are safely omitted by the
    client's GUID gate.
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
            conversation_id=conversation_id or None,
            kind="produced",
            origin="cloud",
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


class ArtifactWriteToolProvider:
    """Thin ToolProvider wrapper over :func:`build_artifact_write_tool`.

    Yields the existing ``artifact_write`` tool as a spec with descriptive
    metadata (``category="artifact"``, ``side_effect="write"``,
    ``artifact_output=True``, ``policy_hint="confirm"``). It changes no behavior
    and no schema: the callable and its reflected schema come straight from the
    builder; per PR1 the metadata never gates execution.

    Selectable through the provider path: reference the provider in
    ``AgentSpec.providers`` and enable ``"artifact_write"`` via ``AgentSpec.tools``
    / ``tool_sets``.
    """

    def __init__(
        self, client: StorageArtifactClient, *, tenant_id: str, conversation_id: str = ""
    ) -> None:
        self._client = client
        self._tenant_id = tenant_id
        self._conversation_id = conversation_id

    def provide(self) -> list[ToolSpec]:
        scratch = ToolRegistry()
        build_artifact_write_tool(
            scratch,
            self._client,
            tenant_id=self._tenant_id,
            conversation_id=self._conversation_id,
        )
        return [
            spec_from_callable(
                scratch.get("artifact_write"),
                name="artifact_write",
                schema=scratch.get_schema("artifact_write"),
                category="artifact",
                capabilities=("artifact", "write"),
                side_effect="write",
                artifact_output=True,
                policy_hint="confirm",
            )
        ]
