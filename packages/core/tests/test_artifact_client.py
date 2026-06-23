"""Tests for StorageArtifactClient."""

import hashlib

import httpx
import pytest
import respx
from neuroagent.artifacts.client import (
    ArtifactRef,
    StorageArtifactClient,
    _is_allowed_content_type,
    _sha256,
)

_ARTIFACT_ID = "art-123"
_UPLOAD_URL = "https://presigned.example.com/upload/art-123"
_DOWNLOAD_URL = "https://storage.example.com/artifacts/art-123"


@pytest.fixture
def client() -> StorageArtifactClient:
    return StorageArtifactClient(
        base_url="https://storage.internal",
        s2s_token_provider=lambda: "test-token",
    )


class TestHelpers:
    def test_is_allowed_content_type(self) -> None:
        assert _is_allowed_content_type("text/plain")
        assert _is_allowed_content_type("image/png")
        assert _is_allowed_content_type("application/json")
        assert _is_allowed_content_type("application/pdf")
        # multimodal media now allowed
        assert _is_allowed_content_type("video/mp4")
        assert _is_allowed_content_type("audio/mpeg")
        # arbitrary binaries / application/* stay rejected
        assert not _is_allowed_content_type("application/octet-stream")
        assert not _is_allowed_content_type("application/zip")

    def test_sha256(self) -> None:
        data = b"hello"
        expected = hashlib.sha256(data).hexdigest()
        assert _sha256(data) == expected


class TestHappyPath:
    @respx.mock
    async def test_create_artifact_happy_path(self, client: StorageArtifactClient) -> None:
        meta_route = respx.post("https://storage.internal/api/v1/storage/artifacts").mock(
            return_value=httpx.Response(
                200,
                json={"artifact_id": _ARTIFACT_ID, "upload_url": _UPLOAD_URL},
            )
        )
        upload_route = respx.put(_UPLOAD_URL).mock(return_value=httpx.Response(200))
        complete_route = respx.post(
            f"https://storage.internal/api/v1/storage/artifacts/{_ARTIFACT_ID}/complete"
        ).mock(
            return_value=httpx.Response(
                200,
                json={"download_url": _DOWNLOAD_URL, "sha256": _sha256(b"hello")},
            )
        )

        ref = await client.create_artifact(
            tenant_id="t1",
            content_bytes=b"hello",
            content_type="text/plain",
            filename="test.txt",
        )

        assert isinstance(ref, ArtifactRef)
        assert ref.id == _ARTIFACT_ID
        assert ref.url == _DOWNLOAD_URL
        assert ref.sha256 == _sha256(b"hello")
        assert ref.filename == "test.txt"
        assert ref.content_type == "text/plain"

        assert meta_route.called
        req = meta_route.calls.last.request
        assert req.headers["Authorization"] == "Bearer test-token"
        import json

        payload = json.loads(req.content)
        assert payload["tenant_id"] == "t1"
        assert payload["filename"] == "test.txt"
        assert payload["content_type"] == "text/plain"
        assert payload["sha256"] == _sha256(b"hello")

        assert upload_route.called
        assert complete_route.called

    @respx.mock
    async def test_create_artifact_with_async_token_provider(self) -> None:
        async_client = StorageArtifactClient(
            base_url="https://storage.internal",
            s2s_token_provider=lambda: __import__("asyncio").sleep(0.01) or "async-token",  # type: ignore[return-value]
        )
        respx.post("https://storage.internal/api/v1/storage/artifacts").mock(
            return_value=httpx.Response(
                200,
                json={"artifact_id": _ARTIFACT_ID, "upload_url": _UPLOAD_URL},
            )
        )
        respx.put(_UPLOAD_URL).mock(return_value=httpx.Response(200))
        respx.post(
            f"https://storage.internal/api/v1/storage/artifacts/{_ARTIFACT_ID}/complete"
        ).mock(
            return_value=httpx.Response(
                200,
                json={"download_url": _DOWNLOAD_URL, "sha256": _sha256(b"hello")},
            )
        )

        ref = await async_client.create_artifact(
            tenant_id="t1",
            content_bytes=b"hello",
            content_type="text/plain",
            filename="test.txt",
        )
        assert ref.id == _ARTIFACT_ID


class TestMediaPersist:
    """Multimodal media (video/audio) persists through the full mocked flow."""

    @respx.mock
    @pytest.mark.parametrize(
        "content_type,filename,payload",
        [
            ("video/mp4", "clip.mp4", b"\x00\x00\x00\x18ftypmp42-video-bytes"),
            ("audio/mpeg", "speech.mp3", b"ID3-audio-bytes"),
        ],
    )
    async def test_uploads_media(
        self,
        client: StorageArtifactClient,
        content_type: str,
        filename: str,
        payload: bytes,
    ) -> None:
        import json

        meta_route = respx.post("https://storage.internal/api/v1/storage/artifacts").mock(
            return_value=httpx.Response(
                200,
                json={"artifact_id": _ARTIFACT_ID, "upload_url": _UPLOAD_URL},
            )
        )
        upload_route = respx.put(_UPLOAD_URL).mock(return_value=httpx.Response(200))
        complete_route = respx.post(
            f"https://storage.internal/api/v1/storage/artifacts/{_ARTIFACT_ID}/complete"
        ).mock(
            return_value=httpx.Response(
                200,
                json={"download_url": _DOWNLOAD_URL, "sha256": _sha256(payload)},
            )
        )

        ref = await client.create_artifact(
            tenant_id="t1",
            content_bytes=payload,
            content_type=content_type,
            filename=filename,
        )

        assert isinstance(ref, ArtifactRef)
        assert ref.url == _DOWNLOAD_URL
        assert ref.content_type == content_type
        assert ref.sha256 == _sha256(payload)

        # metadata request carries the media content_type + payload digest
        meta_payload = json.loads(meta_route.calls.last.request.content)
        assert meta_payload["content_type"] == content_type
        assert meta_payload["filename"] == filename
        assert meta_payload["sha256"] == _sha256(payload)
        assert meta_payload["size"] == len(payload)

        # raw bytes uploaded via PUT with the media Content-Type
        assert upload_route.called
        upload_req = upload_route.calls.last.request
        assert upload_req.content == payload
        assert upload_req.headers["Content-Type"] == content_type
        assert complete_route.called


class TestValidation:
    async def test_rejects_oversized_file(self, client: StorageArtifactClient) -> None:
        big = b"x" * (50 * 1024 * 1024 + 1)
        with pytest.raises(ValueError, match="Payload too large"):
            await client.create_artifact(
                tenant_id="t1",
                content_bytes=big,
                content_type="text/plain",
                filename="big.txt",
            )

    async def test_rejects_octet_stream_content_type(self, client: StorageArtifactClient) -> None:
        # application/octet-stream stays rejected even though video/audio are now allowed.
        with pytest.raises(ValueError, match="Content type not allowed"):
            await client.create_artifact(
                tenant_id="t1",
                content_bytes=b"x",
                content_type="application/octet-stream",
                filename="blob.bin",
            )


class TestRetry:
    @respx.mock
    async def test_retries_once_on_network_error(self, client: StorageArtifactClient) -> None:
        calls = 0

        def _side_effect(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise httpx.ConnectError("connection refused")
            return httpx.Response(
                200,
                json={"artifact_id": _ARTIFACT_ID, "upload_url": _UPLOAD_URL},
            )

        respx.post("https://storage.internal/api/v1/storage/artifacts").mock(
            side_effect=_side_effect
        )
        respx.put(_UPLOAD_URL).mock(return_value=httpx.Response(200))
        respx.post(
            f"https://storage.internal/api/v1/storage/artifacts/{_ARTIFACT_ID}/complete"
        ).mock(return_value=httpx.Response(200, json={"download_url": _DOWNLOAD_URL}))

        ref = await client.create_artifact(
            tenant_id="t1",
            content_bytes=b"hello",
            content_type="text/plain",
            filename="test.txt",
        )
        assert ref.id == _ARTIFACT_ID
        assert calls == 2

    @respx.mock
    async def test_fails_after_two_network_errors(self, client: StorageArtifactClient) -> None:
        respx.post("https://storage.internal/api/v1/storage/artifacts").mock(
            side_effect=httpx.ConnectError("connection refused")
        )

        with pytest.raises(httpx.ConnectError):
            await client.create_artifact(
                tenant_id="t1",
                content_bytes=b"hello",
                content_type="text/plain",
                filename="test.txt",
            )

    @respx.mock
    async def test_does_not_retry_on_4xx(self, client: StorageArtifactClient) -> None:
        respx.post("https://storage.internal/api/v1/storage/artifacts").mock(
            return_value=httpx.Response(400, json={"error": "bad request"})
        )

        with pytest.raises(httpx.HTTPStatusError):
            await client.create_artifact(
                tenant_id="t1",
                content_bytes=b"hello",
                content_type="text/plain",
                filename="test.txt",
            )
