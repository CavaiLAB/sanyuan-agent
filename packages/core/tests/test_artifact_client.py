"""Tests for StorageArtifactClient — pinned to the deployed C# ``ArtifactEndpoints``
wire contract (camelCase; ``objectKey`` required; ``sha256`` = base64 of the raw
digest; ``id``/``uploadUrl`` in the create response; ``/complete`` takes no body;
download URL from ``GET /{id}/download``)."""

import base64
import hashlib
import json

import httpx
import pytest
import respx
from sanyuan_agent.artifacts.client import (
    ArtifactRef,
    StorageArtifactClient,
    _is_allowed_content_type,
    _sha256,
    _sha256_b64,
)

_ARTIFACT_ID = "art-123"
_UPLOAD_URL = "https://presigned.example.com/upload/art-123"
_DOWNLOAD_URL = "https://storage.example.com/artifacts/art-123?sig=abc"
_CREATE = "https://storage.internal/api/v1/artifacts"
_COMPLETE = f"https://storage.internal/api/v1/artifacts/{_ARTIFACT_ID}/complete"
_DOWNLOAD = f"https://storage.internal/api/v1/artifacts/{_ARTIFACT_ID}/download"
_CONV_GUID = "2036b609-c324-4fee-aae1-a25d8d41a264"


@pytest.fixture
def client() -> StorageArtifactClient:
    return StorageArtifactClient(
        base_url="https://storage.internal",
        s2s_token_provider=lambda: "test-token",
    )


def _mock_full_flow(sha_bytes: bytes = b"hello") -> tuple:
    """Mock all three Storage steps + the download lookup. Returns their routes."""
    create = respx.post(_CREATE).mock(
        return_value=httpx.Response(201, json={"id": _ARTIFACT_ID, "uploadUrl": _UPLOAD_URL})
    )
    upload = respx.put(_UPLOAD_URL).mock(return_value=httpx.Response(200))
    complete = respx.post(_COMPLETE).mock(return_value=httpx.Response(200, json={}))
    download = respx.get(_DOWNLOAD).mock(return_value=httpx.Response(200, json={"url": _DOWNLOAD_URL}))
    return create, upload, complete, download


class TestHelpers:
    def test_is_allowed_content_type(self) -> None:
        assert _is_allowed_content_type("text/plain")
        assert _is_allowed_content_type("text/html")  # fireworks.html passes
        assert _is_allowed_content_type("image/png")
        assert _is_allowed_content_type("application/json")
        assert _is_allowed_content_type("video/mp4")
        assert not _is_allowed_content_type("application/octet-stream")

    def test_sha256_hex_vs_base64(self) -> None:
        data = b"hello"
        assert _sha256(data) == hashlib.sha256(data).hexdigest()
        # The wire field is base64 of the RAW digest, not the hex string.
        assert _sha256_b64(data) == base64.b64encode(hashlib.sha256(data).digest()).decode()


class TestHappyPath:
    @respx.mock
    async def test_create_artifact_happy_path(self, client: StorageArtifactClient) -> None:
        create, upload, complete, download = _mock_full_flow()

        ref = await client.create_artifact(
            tenant_id="t1",
            content_bytes=b"hello",
            content_type="text/plain",
            filename="test.txt",
        )

        assert isinstance(ref, ArtifactRef)
        assert ref.id == _ARTIFACT_ID
        assert ref.url == _DOWNLOAD_URL  # browser-usable presigned GET from /download
        assert ref.sha256 == _sha256(b"hello")  # hex for the SSE frame
        assert ref.filename == "test.txt"

        assert create.called and upload.called and complete.called and download.called
        req = create.calls.last.request
        assert req.headers["Authorization"] == "Bearer test-token"
        payload = json.loads(req.content)
        assert payload["objectKey"].startswith("artifacts/")
        assert payload["objectKey"].endswith("/test.txt")
        assert payload["sizeBytes"] == len(b"hello")
        assert payload["sha256"] == _sha256_b64(b"hello")
        assert payload["contentType"] == "text/plain"
        # legacy fields must NOT be present (they silently stored nothing)
        assert "filename" not in payload and "size" not in payload

        # complete carries no body (id is the route param)
        assert complete.calls.last.request.content in (b"", b"null")

    @respx.mock
    async def test_create_artifact_with_async_token_provider(self) -> None:
        async def _provider() -> str:
            return "async-token"

        async_client = StorageArtifactClient(
            base_url="https://storage.internal", s2s_token_provider=_provider
        )
        create, *_ = _mock_full_flow()
        ref = await async_client.create_artifact(
            tenant_id="t1", content_bytes=b"hello", content_type="text/plain", filename="test.txt"
        )
        assert ref.id == _ARTIFACT_ID
        assert create.calls.last.request.headers["Authorization"] == "Bearer async-token"


class TestConversationScoping:
    @respx.mock
    async def test_guid_conversation_id_is_forwarded(self, client: StorageArtifactClient) -> None:
        create, *_ = _mock_full_flow()
        await client.register_sandbox_artifact(
            tenant_id="t1",
            content_bytes=b"<html></html>",
            content_type="text/html",
            filename="fireworks.html",
            run_id="run-1",
            conversation_id=_CONV_GUID,
            kind="produced",
            origin="cloud",
        )
        payload = json.loads(create.calls.last.request.content)
        assert payload["conversationId"] == _CONV_GUID
        assert payload["kind"] == "produced" and payload["origin"] == "cloud"
        assert payload["objectKey"] == "artifacts/run-1/fireworks.html"

    @respx.mock
    async def test_non_guid_conversation_id_is_omitted(self, client: StorageArtifactClient) -> None:
        create, *_ = _mock_full_flow()
        await client.register_sandbox_artifact(
            tenant_id="t1",
            content_bytes=b"x",
            content_type="text/plain",
            filename="a.txt",
            run_id="run-1",
            conversation_id="c-demo-1",  # not a GUID → omit (C# defaults to null)
        )
        payload = json.loads(create.calls.last.request.content)
        assert "conversationId" not in payload


class TestMediaPersist:
    @respx.mock
    @pytest.mark.parametrize(
        "content_type,filename,payload",
        [
            ("video/mp4", "clip.mp4", b"\x00\x00\x00\x18ftypmp42-video-bytes"),
            ("audio/mpeg", "speech.mp3", b"ID3-audio-bytes"),
        ],
    )
    async def test_uploads_media(
        self, client: StorageArtifactClient, content_type: str, filename: str, payload: bytes
    ) -> None:
        create, upload, complete, _ = _mock_full_flow(payload)
        ref = await client.create_artifact(
            tenant_id="t1", content_bytes=payload, content_type=content_type, filename=filename
        )
        assert ref.content_type == content_type
        meta = json.loads(create.calls.last.request.content)
        assert meta["contentType"] == content_type
        assert meta["sha256"] == _sha256_b64(payload)
        assert meta["sizeBytes"] == len(payload)
        assert upload.calls.last.request.content == payload
        assert upload.calls.last.request.headers["Content-Type"] == content_type
        assert complete.called


class TestValidation:
    async def test_rejects_oversized_file(self, client: StorageArtifactClient) -> None:
        big = b"x" * (50 * 1024 * 1024 + 1)
        with pytest.raises(ValueError, match="Payload too large"):
            await client.create_artifact(
                tenant_id="t1", content_bytes=big, content_type="text/plain", filename="big.txt"
            )

    async def test_rejects_octet_stream_content_type(self, client: StorageArtifactClient) -> None:
        with pytest.raises(ValueError, match="Content type not allowed"):
            await client.create_artifact(
                tenant_id="t1", content_bytes=b"x", content_type="application/octet-stream", filename="blob.bin"
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
            return httpx.Response(201, json={"id": _ARTIFACT_ID, "uploadUrl": _UPLOAD_URL})

        respx.post(_CREATE).mock(side_effect=_side_effect)
        respx.put(_UPLOAD_URL).mock(return_value=httpx.Response(200))
        respx.post(_COMPLETE).mock(return_value=httpx.Response(200, json={}))
        respx.get(_DOWNLOAD).mock(return_value=httpx.Response(200, json={"url": _DOWNLOAD_URL}))

        ref = await client.create_artifact(
            tenant_id="t1", content_bytes=b"hello", content_type="text/plain", filename="test.txt"
        )
        assert ref.id == _ARTIFACT_ID
        assert calls == 2

    @respx.mock
    async def test_fails_after_two_network_errors(self, client: StorageArtifactClient) -> None:
        respx.post(_CREATE).mock(side_effect=httpx.ConnectError("connection refused"))
        with pytest.raises(httpx.ConnectError):
            await client.create_artifact(
                tenant_id="t1", content_bytes=b"hello", content_type="text/plain", filename="test.txt"
            )

    @respx.mock
    async def test_does_not_retry_on_4xx(self, client: StorageArtifactClient) -> None:
        respx.post(_CREATE).mock(return_value=httpx.Response(400, json={"error": "bad request"}))
        with pytest.raises(httpx.HTTPStatusError):
            await client.create_artifact(
                tenant_id="t1", content_bytes=b"hello", content_type="text/plain", filename="test.txt"
            )
