"""Multimodal tool provider (PR3): fake-client provider, artifact-shaped output.

The provider yields four VeADK-style tools (image_generate / image_edit /
video_generate / text_to_speech) over an *injected* client - no real external
API. These tests pin:

1. provide() yields exactly the four tools with the right metadata.
2. each tool forwards the right args to the client and returns a unified
   artifact/workspace shape (never a bare local ``path``).
3. the tools register into a ToolRegistry with real (non-degenerate) schemas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from neuroagent.tools import (
    MultimodalArtifact,
    MultimodalClient,
    MultimodalToolProvider,
)
from neuroagent.tools.multimodal import (
    ALL_TOOL_NAMES,
    SECRET_EDIT,
    SECRET_IMAGE,
    SECRET_TTS,
    SECRET_VIDEO,
)
from neuroagent.tools.provider import register_provider
from neuroagent.tools.registry import ToolRegistry

EXPECTED_NAMES = {"image_generate", "image_edit", "video_generate", "text_to_speech"}


@dataclass
class FakeClient:
    """Records calls; returns predictable artifact references (no real API)."""

    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def _artifact(self, kind: str, ext: str, mime: str) -> MultimodalArtifact:
        return MultimodalArtifact(
            artifact_id=f"art-{kind}",
            filename=f"{kind}.{ext}",
            content_type=mime,
            download_url=f"https://example.com/{kind}.{ext}",
            sha256=f"sha-{kind}",
            workspace_ref=f"workspace://{kind}.{ext}",
        )

    async def image_generate(self, *, prompt: str, size: str) -> MultimodalArtifact:
        self.calls.append(("image_generate", {"prompt": prompt, "size": size}))
        return self._artifact("image", "png", "image/png")

    async def image_edit(self, *, input_ref: str, prompt: str, size: str) -> MultimodalArtifact:
        self.calls.append(("image_edit", {"input_ref": input_ref, "prompt": prompt, "size": size}))
        return self._artifact("edit", "png", "image/png")

    async def video_generate(
        self, *, prompt: str, first_frame_ref: str, last_frame_ref: str
    ) -> MultimodalArtifact:
        self.calls.append(
            (
                "video_generate",
                {
                    "prompt": prompt,
                    "first_frame_ref": first_frame_ref,
                    "last_frame_ref": last_frame_ref,
                },
            )
        )
        return self._artifact("video", "mp4", "video/mp4")

    async def text_to_speech(self, *, text: str, voice: str) -> MultimodalArtifact:
        self.calls.append(("text_to_speech", {"text": text, "voice": voice}))
        return self._artifact("tts", "mp3", "audio/mpeg")


def _specs_by_name() -> dict[str, Any]:
    provider = MultimodalToolProvider(FakeClient())
    return {s.name: s for s in provider.provide()}


# ---------------------------------------------------------------------------
# provide() shape and metadata
# ---------------------------------------------------------------------------


class TestProvideMetadata:
    def test_fake_client_satisfies_protocol(self) -> None:
        assert isinstance(FakeClient(), MultimodalClient)  # runtime-checkable

    def test_yields_exactly_four_named_tools(self) -> None:
        specs = _specs_by_name()
        assert set(specs) == EXPECTED_NAMES
        assert len(specs) == 4

    def test_common_metadata_on_every_spec(self) -> None:
        for spec in _specs_by_name().values():
            assert spec.category == "multimodal"
            assert spec.artifact_output is True
            assert spec.policy_hint == "confirm"
            assert spec.side_effect == "external"

    def test_per_tool_capabilities_and_secrets(self) -> None:
        specs = _specs_by_name()
        assert specs["image_generate"].capabilities == ("image", "generate")
        assert specs["image_edit"].capabilities == ("image", "edit")
        assert specs["video_generate"].capabilities == ("video", "generate")
        assert specs["text_to_speech"].capabilities == ("audio", "tts")

        assert specs["image_generate"].required_secrets == (SECRET_IMAGE,)
        assert specs["image_edit"].required_secrets == (SECRET_EDIT,)
        assert specs["video_generate"].required_secrets == (SECRET_VIDEO,)
        assert specs["text_to_speech"].required_secrets == (SECRET_TTS,)


# ---------------------------------------------------------------------------
# Schemas (OpenAI-style, non-degenerate)
# ---------------------------------------------------------------------------


class TestSchemas:
    def test_image_generate_schema_params(self) -> None:
        schema = _specs_by_name()["image_generate"].schema
        assert schema["name"] == "image_generate"
        props = schema["parameters"]["properties"]
        assert set(props) == {"prompt", "size"}
        assert props["prompt"] == {"type": "string"}
        assert schema["parameters"]["required"] == ["prompt"]  # size has a default

    def test_video_generate_optional_frames_not_required(self) -> None:
        schema = _specs_by_name()["video_generate"].schema
        props = schema["parameters"]["properties"]
        assert set(props) == {"prompt", "first_frame_ref", "last_frame_ref"}
        assert schema["parameters"]["required"] == ["prompt"]

    def test_no_degenerate_arguments_wrapper(self) -> None:
        """A reflected real signature must not collapse to a single *arguments* param."""
        for spec in _specs_by_name().values():
            props = spec.schema["parameters"]["properties"]
            assert "arguments" not in props
            assert len(props) >= 1


# ---------------------------------------------------------------------------
# Execution: args forwarded, unified artifact shape, no bare path
# ---------------------------------------------------------------------------


_ARTIFACT_KEYS = {
    "artifact_id",
    "filename",
    "content_type",
    "download_url",
    "sha256",
    "workspace_ref",
}


class TestExecution:
    async def test_image_generate_forwards_and_returns_artifact(self) -> None:
        client = FakeClient()
        specs = {s.name: s for s in MultimodalToolProvider(client).provide()}

        result = await specs["image_generate"].callable(prompt="a cat", size="512x512")

        assert client.calls == [("image_generate", {"prompt": "a cat", "size": "512x512"})]
        assert set(result) == _ARTIFACT_KEYS
        assert result["artifact_id"] == "art-image"
        assert result["content_type"] == "image/png"
        assert result["download_url"].startswith("https://")
        assert result["sha256"] == "sha-image"
        assert "path" not in result  # never a bare local path

    async def test_image_edit_forwards_input_ref(self) -> None:
        client = FakeClient()
        specs = {s.name: s for s in MultimodalToolProvider(client).provide()}

        result = await specs["image_edit"].callable(input_ref="art-src", prompt="add a hat")

        assert client.calls == [
            (
                "image_edit",
                {"input_ref": "art-src", "prompt": "add a hat", "size": "1024x1024"},
            )
        ]
        assert "path" not in result
        assert result["artifact_id"] == "art-edit"

    async def test_video_generate_defaults_empty_frames(self) -> None:
        client = FakeClient()
        specs = {s.name: s for s in MultimodalToolProvider(client).provide()}

        result = await specs["video_generate"].callable(prompt="a sunrise")

        assert client.calls == [
            (
                "video_generate",
                {"prompt": "a sunrise", "first_frame_ref": "", "last_frame_ref": ""},
            )
        ]
        assert result["content_type"] == "video/mp4"
        assert "path" not in result

    async def test_text_to_speech_forwards_voice(self) -> None:
        client = FakeClient()
        specs = {s.name: s for s in MultimodalToolProvider(client).provide()}

        result = await specs["text_to_speech"].callable(text="hello", voice="alloy")

        assert client.calls == [("text_to_speech", {"text": "hello", "voice": "alloy"})]
        assert result["content_type"] == "audio/mpeg"
        assert result["workspace_ref"] == "workspace://tts.mp3"
        assert "path" not in result


# ---------------------------------------------------------------------------
# Registration into a ToolRegistry
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_register_provider_yields_four_tools(self) -> None:
        registry = ToolRegistry()
        names = register_provider(registry, MultimodalToolProvider(FakeClient()))
        assert set(names) == EXPECTED_NAMES
        assert set(registry.list_tools()) == EXPECTED_NAMES

    def test_registry_tool_defs_are_function_typed(self) -> None:
        registry = ToolRegistry()
        register_provider(registry, MultimodalToolProvider(FakeClient()))
        for tool_def in registry.tool_defs():
            assert tool_def["type"] == "function"
            fn = tool_def["function"]
            assert set(fn) >= {"name", "parameters"}
            assert "arguments" not in fn["parameters"].get("properties", {})


# ---------------------------------------------------------------------------
# Capability gating: supported_tools narrows the registered subset (PR17)
# ---------------------------------------------------------------------------


class TestSupportedToolsSubset:
    def test_all_tool_names_constant(self) -> None:
        assert ALL_TOOL_NAMES == (
            "image_generate", "image_edit", "video_generate", "text_to_speech"
        )

    def test_default_none_yields_all_four(self) -> None:
        provider = MultimodalToolProvider(FakeClient())  # no supported_tools
        assert {s.name for s in provider.provide()} == EXPECTED_NAMES

    def test_subset_yields_only_requested_tools_in_order(self) -> None:
        provider = MultimodalToolProvider(
            FakeClient(), supported_tools=("image_generate", "image_edit", "video_generate")
        )
        specs = provider.provide()
        # excluded, and registration order preserved
        assert [s.name for s in specs] == ["image_generate", "image_edit", "video_generate"]
        assert "text_to_speech" not in {s.name for s in specs}

    def test_single_tool_subset(self) -> None:
        provider = MultimodalToolProvider(FakeClient(), supported_tools={"video_generate"})
        assert [s.name for s in provider.provide()] == ["video_generate"]

    def test_empty_subset_yields_no_tools(self) -> None:
        provider = MultimodalToolProvider(FakeClient(), supported_tools=())
        assert provider.provide() == []

    def test_unknown_tool_name_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown multimodal tool names"):
            MultimodalToolProvider(FakeClient(), supported_tools={"image_generate", "bogus_tool"})

    def test_subset_specs_keep_full_metadata(self) -> None:
        provider = MultimodalToolProvider(FakeClient(), supported_tools={"video_generate"})
        spec = provider.provide()[0]
        assert spec.category == "multimodal"
        assert spec.capabilities == ("video", "generate")
        assert spec.required_secrets == (SECRET_VIDEO,)

    def test_subset_registers_only_subset_into_registry(self) -> None:
        registry = ToolRegistry()
        names = register_provider(
            registry,
            MultimodalToolProvider(
                FakeClient(), supported_tools=("image_generate", "image_edit", "video_generate")
            ),
        )
        assert set(names) == {"image_generate", "image_edit", "video_generate"}
        assert "text_to_speech" not in registry.list_tools()
