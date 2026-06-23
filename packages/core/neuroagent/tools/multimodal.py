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
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from neuroagent.tools.provider import ToolSpec, spec_from_callable

# Declared (not read) secret/config env-var names per capability.
SECRET_IMAGE = "MODEL_IMAGE_NAME"
SECRET_EDIT = "MODEL_EDIT_NAME"
SECRET_VIDEO = "MODEL_VIDEO_NAME"
SECRET_TTS = "MODEL_TTS_NAME"

# The full set of multimodal tool names, in registration order.
ALL_TOOL_NAMES = ("image_generate", "image_edit", "video_generate", "text_to_speech")


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
    """Yields the multimodal tools bound to an injected *client*.

    *supported_tools* optionally narrows the registered tools to a subset of
    :data:`ALL_TOOL_NAMES` (e.g. a real vendor that only covers image+video, not
    speech). ``None`` (the default) yields all four, preserving the prior
    behaviour for the fake/dev full client (a :class:`MultimodalClient`) and
    existing callers. An unknown name is a programming error and raises
    ``ValueError`` rather than silently dropping a tool.

    Each tool name is also its client method name (``image_generate`` ->
    ``client.image_generate``), so a narrowed provider supports a **partial
    client** that implements only the subset's methods: when *supported_tools* is
    given, only those methods must exist on *client*; a missing one raises
    ``ValueError`` naming the tool/method, and methods for *unrequested* tools are
    never required. This lets a future speech-only (VeSpeech / Seed Speech) client
    provide just ``text_to_speech`` without dummy image/video methods, and an
    image/video-only client omit ``text_to_speech``.
    """

    def __init__(
        self, client: MultimodalClient, *, supported_tools: Iterable[str] | None = None
    ) -> None:
        self._client = client
        if supported_tools is None:
            # Full client: register all four; preserve prior behaviour exactly
            # (the MultimodalClient protocol already covers method existence).
            self._supported: frozenset[str] | None = None
            return
        requested = frozenset(supported_tools)
        unknown = requested - frozenset(ALL_TOOL_NAMES)
        if unknown:
            raise ValueError(f"unknown multimodal tool names: {sorted(unknown)}")
        # Partial-client support: only the requested tools' methods must exist.
        # Tool name == method name, so validate those and nothing else.
        missing = sorted(name for name in requested if not callable(getattr(client, name, None)))
        if missing:
            raise ValueError(
                "multimodal client is missing method(s) for requested tool(s): "
                f"{missing}"
            )
        self._supported = requested

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
        specs = [
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
        if self._supported is None:
            return specs
        return [s for s in specs if s.name in self._supported]
