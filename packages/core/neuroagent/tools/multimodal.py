"""Multimodal tool provider (image / video / speech) over an injected client.

VeADK-style builtin tools - ``image_generate`` / ``image_edit`` /
``video_generate`` / ``text_to_speech`` - exposed as a :class:`ToolProvider`.

Two boundaries are deliberate:

1. **No real external API in the harness.** The provider talks only to an
   injected :class:`MultimodalClient` (a Protocol). The commercial side supplies
   a real client (vendor + credentials) later; tests supply a fake. harness
   imports nothing from ``neuroagent_app`` / ``neuroagent_sandbox_supervisor``.
2. **Artifact/workspace output, never a bare local path.** Every tool returns a
   :class:`MultimodalArtifact` shape (``artifact_id`` / ``filename`` /
   ``content_type`` / ``download_url`` / ``sha256`` / optional ``workspace_ref``)
   so a generated asset is a first-class artifact, not a path on some box.

The tools carry provider metadata (``category="multimodal"``,
``side_effect="external"``, ``artifact_output=True``, ``policy_hint="confirm"``,
per-tool ``capabilities`` / ``required_secrets``). As in PR1, that metadata is
descriptive: it informs policy building / UI / audit but never gates execution
on its own. Credentials are *declared* via ``required_secrets`` env-var names;
this module never reads them.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from neuroagent.tools.provider import ToolSpec, spec_from_callable

# Declared (not read) secret/config env-var names per capability.
SECRET_IMAGE = "MODEL_IMAGE_NAME"
SECRET_EDIT = "MODEL_EDIT_NAME"
SECRET_VIDEO = "MODEL_VIDEO_NAME"
SECRET_TTS = "MODEL_TTS_NAME"


@dataclass(frozen=True, slots=True)
class MultimodalArtifact:
    """A generated asset as an artifact/workspace reference - never a raw path.

    *workspace_ref* is optional (empty when the asset lives only as an
    artifact); ``artifact_id`` + ``download_url`` are the primary handle.
    """

    artifact_id: str
    filename: str
    content_type: str
    download_url: str
    sha256: str
    workspace_ref: str = ""


@runtime_checkable
class MultimodalClient(Protocol):
    """The vendor-facing surface the provider needs.

    Each method returns a :class:`MultimodalArtifact`. Real implementations call
    an external API and register the result via the Storage artifact client;
    tests pass a fake. Methods are async - generation is I/O-bound.
    """

    async def image_generate(self, *, prompt: str, size: str) -> MultimodalArtifact: ...

    async def image_edit(self, *, input_ref: str, prompt: str, size: str) -> MultimodalArtifact: ...

    async def video_generate(
        self, *, prompt: str, first_frame_ref: str, last_frame_ref: str
    ) -> MultimodalArtifact: ...

    async def text_to_speech(self, *, text: str, voice: str) -> MultimodalArtifact: ...


def _result(artifact: MultimodalArtifact) -> dict[str, Any]:
    """Normalize an artifact into the tool's JSON-serializable result dict."""
    return dataclasses.asdict(artifact)


class MultimodalToolProvider:
    """Yields the four multimodal tools bound to an injected *client*."""

    def __init__(self, client: MultimodalClient) -> None:
        self._client = client

    def provide(self) -> list[ToolSpec]:
        client = self._client

        async def image_generate(prompt: str, size: str = "1024x1024") -> dict[str, Any]:
            """Generate an image from a text prompt; returns an artifact reference."""
            return _result(await client.image_generate(prompt=prompt, size=size))

        async def image_edit(
            input_ref: str, prompt: str, size: str = "1024x1024"
        ) -> dict[str, Any]:
            """Edit an existing image artifact per a prompt; returns an artifact reference."""
            return _result(await client.image_edit(input_ref=input_ref, prompt=prompt, size=size))

        async def video_generate(
            prompt: str, first_frame_ref: str = "", last_frame_ref: str = ""
        ) -> dict[str, Any]:
            """Generate a video from a prompt (optional first/last frame refs);
            returns an artifact reference."""
            return _result(
                await client.video_generate(
                    prompt=prompt,
                    first_frame_ref=first_frame_ref,
                    last_frame_ref=last_frame_ref,
                )
            )

        async def text_to_speech(text: str, voice: str = "default") -> dict[str, Any]:
            """Synthesize speech audio from text; returns an artifact reference."""
            return _result(await client.text_to_speech(text=text, voice=voice))

        common: dict[str, Any] = {
            "category": "multimodal",
            "side_effect": "external",
            "artifact_output": True,
            "policy_hint": "confirm",
        }
        return [
            spec_from_callable(
                image_generate,
                capabilities=("image", "generate"),
                required_secrets=(SECRET_IMAGE,),
                **common,
            ),
            spec_from_callable(
                image_edit,
                capabilities=("image", "edit"),
                required_secrets=(SECRET_EDIT,),
                **common,
            ),
            spec_from_callable(
                video_generate,
                capabilities=("video", "generate"),
                required_secrets=(SECRET_VIDEO,),
                **common,
            ),
            spec_from_callable(
                text_to_speech,
                capabilities=("audio", "tts"),
                required_secrets=(SECRET_TTS,),
                **common,
            ),
        ]
