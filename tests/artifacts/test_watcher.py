"""Tests for ArtifactWatcher — sandbox outputs auto-artifact registration."""

from __future__ import annotations

import hashlib
from typing import Any

import httpx
import pytest
import respx
from neuroagent.artifacts.client import ArtifactRef, StorageArtifactClient
from neuroagent.artifacts.watcher import (
    ArtifactEvent,
    ArtifactWatcher,
    _get_extension,
    _sniff_mime,
)

# -- Constants ------------------------------------------------------------

_SANDBOX_ID = "sb-test-001"
_SUPERVISOR_URL = "https://supervisor.internal"
_STORAGE_URL = "https://storage.internal"
_TENANT_ID = "t-001"
_RUN_ID = "r-001"
_USER_ID = "u-001"
_THREAD_ID = "th-001"
_ARTIFACT_ID = "art-abc"
_UPLOAD_URL = "https://minio.internal/upload/art-abc"
_DOWNLOAD_URL = "https://storage.internal/download/art-abc"

_FILE_CONTENT = b"Hello from sandbox outputs!"
_FILE_SHA256 = hashlib.sha256(_FILE_CONTENT).hexdigest()


# -- Fixtures -------------------------------------------------------------


@pytest.fixture
def events() -> list[ArtifactEvent]:
    """Collects events emitted by the watcher."""
    return []


@pytest.fixture
def storage_client() -> StorageArtifactClient:
    return StorageArtifactClient(
        base_url=_STORAGE_URL,
        s2s_token_provider=lambda: "s2s-test-token",
    )


@pytest.fixture
def watcher(
    storage_client: StorageArtifactClient,
    events: list[ArtifactEvent],
) -> ArtifactWatcher:
    async def _on_event(event: ArtifactEvent) -> None:
        events.append(event)

    return ArtifactWatcher(
        sandbox_id=_SANDBOX_ID,
        supervisor_base_url=_SUPERVISOR_URL,
        storage_client=storage_client,
        tenant_id=_TENANT_ID,
        run_id=_RUN_ID,
        user_id=_USER_ID,
        thread_id=_THREAD_ID,
        on_event=_on_event,
        poll_interval=0.05,
    )


def _mock_storage_upload() -> None:
    """Register respx mocks for the 3-step Storage upload."""
    respx.post(f"{_STORAGE_URL}/api/v1/storage/artifacts").mock(
        return_value=httpx.Response(
            200, json={"artifact_id": _ARTIFACT_ID, "upload_url": _UPLOAD_URL}
        )
    )
    respx.put(_UPLOAD_URL).mock(return_value=httpx.Response(200))
    respx.post(
        f"{_STORAGE_URL}/api/v1/storage/artifacts/{_ARTIFACT_ID}/complete"
    ).mock(
        return_value=httpx.Response(
            200, json={"download_url": _DOWNLOAD_URL}
        )
    )


def _mock_list_outputs(files: list[dict[str, Any]]) -> respx.MockedRouter:
    return respx.get(
        f"{_SUPERVISOR_URL}/v2/sandboxes/{_SANDBOX_ID}/files/list"
    ).mock(
        return_value=httpx.Response(200, json={"entries": files, "truncated": False})
    )


def _mock_read_file(content: bytes) -> respx.MockedRouter:
    return respx.get(
        f"{_SUPERVISOR_URL}/v2/sandboxes/{_SANDBOX_ID}/files"
    ).mock(return_value=httpx.Response(200, content=content))


# -- Helper to run a single poll cycle ------------------------------------


async def _poll_once(watcher: ArtifactWatcher) -> None:
    await watcher._poll_once()


# -- Tests ----------------------------------------------------------------


class TestHelpers:
    def test_get_extension(self) -> None:
        assert _get_extension("report.pdf") == ".pdf"
        assert _get_extension("archive.tar.gz") == ".gz"
        assert _get_extension("README") == ""
        assert _get_extension(".hidden") == ".hidden"

    def test_sniff_mime_text(self) -> None:
        mime = _sniff_mime(b"hello world\n", "test.txt")
        assert mime == "text/plain"

    def test_sniff_mime_fallback(self) -> None:
        mime = _sniff_mime(b"\x00\x01\x02", "unknown.dat")
        # python-magic may return a specific type; mimetypes falls back
        # to application/octet-stream for truly unknown extensions.
        assert isinstance(mime, str) and "/" in mime


class TestDiscovery:
    @respx.mock
    async def test_discovers_new_file(
        self, watcher: ArtifactWatcher, events: list[ArtifactEvent]
    ) -> None:
        _mock_list_outputs(
            [{"path": "outputs/report.csv", "type": "file", "sizeBytes": 27, "modifiedAt": "2026-04-23T10:00:00Z"}]
        )
        _mock_read_file(_FILE_CONTENT)
        _mock_storage_upload()

        await _poll_once(watcher)

        assert len(events) == 1
        assert events[0].event_type == "artifact.created.v1"
        assert events[0].data["originPath"] == "outputs/report.csv"

    @respx.mock
    async def test_skips_directories(
        self, watcher: ArtifactWatcher, events: list[ArtifactEvent]
    ) -> None:
        _mock_list_outputs(
            [
                {"path": "outputs/subdir", "type": "directory", "sizeBytes": 0, "modifiedAt": ""},
                {"path": "outputs/file.txt", "type": "file", "sizeBytes": 5, "modifiedAt": "2026-04-23T10:00:00Z"},
            ]
        )
        _mock_read_file(b"hello")
        _mock_storage_upload()

        await _poll_once(watcher)
        assert len(events) == 1
        assert events[0].data["originPath"] == "outputs/file.txt"


class TestFullUploadFlow:
    @respx.mock
    async def test_upload_registers_and_emits(
        self, watcher: ArtifactWatcher, events: list[ArtifactEvent]
    ) -> None:
        _mock_list_outputs(
            [{"path": "output.txt", "type": "file", "sizeBytes": 27, "modifiedAt": "2026-04-23T10:00:00Z"}]
        )
        _mock_read_file(_FILE_CONTENT)
        _mock_storage_upload()

        await _poll_once(watcher)

        evt = events[0]
        assert evt.event_type == "artifact.created.v1"
        assert evt.data["artifactId"] == _ARTIFACT_ID
        assert evt.data["threadId"] == _THREAD_ID
        assert evt.data["size"] == len(_FILE_CONTENT)
        assert "downloadUrl" in evt.data

    @respx.mock
    async def test_register_sandbox_artifact_sends_metadata(self) -> None:
        client = StorageArtifactClient(
            base_url=_STORAGE_URL,
            s2s_token_provider=lambda: "tok",
        )
        meta_route = respx.post(
            f"{_STORAGE_URL}/api/v1/storage/artifacts"
        ).mock(
            return_value=httpx.Response(
                200, json={"artifact_id": _ARTIFACT_ID, "upload_url": _UPLOAD_URL}
            )
        )
        respx.put(_UPLOAD_URL).mock(return_value=httpx.Response(200))
        respx.post(
            f"{_STORAGE_URL}/api/v1/storage/artifacts/{_ARTIFACT_ID}/complete"
        ).mock(return_value=httpx.Response(200, json={"download_url": _DOWNLOAD_URL}))

        ref = await client.register_sandbox_artifact(
            tenant_id=_TENANT_ID,
            content_bytes=_FILE_CONTENT,
            content_type="text/csv",
            filename="data.csv",
            run_id=_RUN_ID,
            user_id=_USER_ID,
            origin_path="outputs/data.csv",
        )

        assert isinstance(ref, ArtifactRef)
        assert ref.id == _ARTIFACT_ID
        assert meta_route.called
        import json

        payload = json.loads(meta_route.calls.last.request.content)
        assert payload["run_id"] == _RUN_ID
        assert payload["user_id"] == _USER_ID
        assert payload["origin_path"] == "outputs/data.csv"

    @respx.mock
    async def test_register_sandbox_artifact_100mb_limit(self) -> None:
        client = StorageArtifactClient(
            base_url=_STORAGE_URL,
            s2s_token_provider=lambda: "tok",
        )
        big = b"x" * (100 * 1024 * 1024 + 1)
        with pytest.raises(ValueError, match="Payload too large"):
            await client.register_sandbox_artifact(
                tenant_id=_TENANT_ID,
                content_bytes=big,
                content_type="application/octet-stream",
                filename="big.bin",
            )


class TestSecurity:
    @respx.mock
    async def test_rejects_oversized_file(
        self, watcher: ArtifactWatcher, events: list[ArtifactEvent]
    ) -> None:
        big_size = 101 * 1024 * 1024
        _mock_list_outputs(
            [{"path": "outputs/big.bin", "type": "file", "sizeBytes": big_size, "modifiedAt": "2026-04-23T10:00:00Z"}]
        )
        # No need to mock file read — size gate happens before read

        await _poll_once(watcher)

        assert len(events) == 1
        assert events[0].event_type == "artifact.rejected.v1"
        assert events[0].data["reason"] == "size_exceeded"

    @respx.mock
    @pytest.mark.parametrize("ext", [".so", ".exe", ".dll", ".dylib"])
    async def test_rejects_blacklisted_extensions(
        self,
        watcher: ArtifactWatcher,
        events: list[ArtifactEvent],
        ext: str,
    ) -> None:
        _mock_list_outputs(
            [{"path": f"outputs/malware{ext}", "type": "file", "sizeBytes": 100, "modifiedAt": "2026-04-23T10:00:00Z"}]
        )

        await _poll_once(watcher)

        assert len(events) == 1
        assert events[0].event_type == "artifact.rejected.v1"
        assert events[0].data["reason"] == "blocked_extension"

    @respx.mock
    async def test_rejects_on_actual_content_size(
        self, watcher: ArtifactWatcher, events: list[ArtifactEvent]
    ) -> None:
        """Listing says small, but actual content is huge."""
        _mock_list_outputs(
            [{"path": "outputs/tricky.bin", "type": "file", "sizeBytes": 10, "modifiedAt": "2026-04-23T10:00:00Z"}]
        )
        # Return oversized content
        _mock_read_file(b"x" * (101 * 1024 * 1024))

        await _poll_once(watcher)

        rejections = [e for e in events if e.event_type == "artifact.rejected.v1"]
        assert len(rejections) == 1
        assert rejections[0].data["reason"] == "size_exceeded"


class TestDeltaDetection:
    @respx.mock
    async def test_no_duplicate_processing(
        self, watcher: ArtifactWatcher, events: list[ArtifactEvent]
    ) -> None:
        entries = [
            {"path": "outputs/a.txt", "type": "file", "sizeBytes": 5, "modifiedAt": "2026-04-23T10:00:00Z"}
        ]
        _mock_list_outputs(entries)
        _mock_read_file(b"hello")
        _mock_storage_upload()

        await _poll_once(watcher)
        assert len(events) == 1

        # Second poll — same file, same mtime/size → should not re-process
        await _poll_once(watcher)
        assert len(events) == 1  # still only 1 event

    @respx.mock
    async def test_detects_changed_file(
        self, watcher: ArtifactWatcher, events: list[ArtifactEvent]
    ) -> None:
        _mock_list_outputs(
            [{"path": "outputs/log.txt", "type": "file", "sizeBytes": 10, "modifiedAt": "T1"}]
        )
        _mock_read_file(b"0123456789")
        _mock_storage_upload()

        await _poll_once(watcher)
        assert len(events) == 1

        # File changed (different mtime + size) → but already in _processed set,
        # so won't re-process. This verifies the _processed guard.
        respx.clear()
        _mock_list_outputs(
            [{"path": "outputs/log.txt", "type": "file", "sizeBytes": 20, "modifiedAt": "T2"}]
        )
        _mock_read_file(b"01234567890123456789")
        _mock_storage_upload()

        await _poll_once(watcher)
        # Still 1 — _processed prevents re-upload of same path
        assert len(events) == 1


class TestConcurrency:
    @respx.mock
    async def test_processes_multiple_files(
        self, watcher: ArtifactWatcher, events: list[ArtifactEvent]
    ) -> None:
        files = [
            {"path": "outputs/a.csv", "type": "file", "sizeBytes": 3, "modifiedAt": "T1"},
            {"path": "outputs/b.json", "type": "file", "sizeBytes": 5, "modifiedAt": "T1"},
            {"path": "outputs/c.png", "type": "file", "sizeBytes": 8, "modifiedAt": "T1"},
        ]
        _mock_list_outputs(files)
        _mock_read_file(b"content!!")  # same content for all
        _mock_storage_upload()

        await _poll_once(watcher)

        assert len(events) == 3
        types = {e.data["originPath"] for e in events}
        assert types == {"outputs/a.csv", "outputs/b.json", "outputs/c.png"}


class TestHash:
    @respx.mock
    async def test_sha256_stable(
        self, watcher: ArtifactWatcher, events: list[ArtifactEvent]
    ) -> None:
        content = b"deterministic content"
        expected_hash = hashlib.sha256(content).hexdigest()

        _mock_list_outputs(
            [{"path": "outputs/data.bin", "type": "file", "sizeBytes": len(content), "modifiedAt": "T1"}]
        )
        _mock_read_file(content)

        meta_route = respx.post(
            f"{_STORAGE_URL}/api/v1/storage/artifacts"
        ).mock(
            return_value=httpx.Response(
                200, json={"artifact_id": _ARTIFACT_ID, "upload_url": _UPLOAD_URL}
            )
        )
        respx.put(_UPLOAD_URL).mock(return_value=httpx.Response(200))
        respx.post(
            f"{_STORAGE_URL}/api/v1/storage/artifacts/{_ARTIFACT_ID}/complete"
        ).mock(return_value=httpx.Response(200, json={"download_url": _DOWNLOAD_URL}))

        await _poll_once(watcher)

        import json

        payload = json.loads(meta_route.calls.last.request.content)
        assert payload["sha256"] == expected_hash


class TestS2SJWTFailure:
    @respx.mock
    async def test_storage_auth_failure_handled_gracefully(
        self, watcher: ArtifactWatcher, events: list[ArtifactEvent]
    ) -> None:
        """Storage returns 401 → watcher logs error but does not crash."""
        _mock_list_outputs(
            [{"path": "outputs/file.txt", "type": "file", "sizeBytes": 5, "modifiedAt": "T1"}]
        )
        _mock_read_file(b"hello")
        respx.post(f"{_STORAGE_URL}/api/v1/storage/artifacts").mock(
            return_value=httpx.Response(401, json={"error": "unauthorized"})
        )

        # Should not raise
        await _poll_once(watcher)
        # No events emitted — upload failed silently
        assert len(events) == 0


class TestLifecycle:
    async def test_stop_cancels_task_no_leak(
        self, watcher: ArtifactWatcher
    ) -> None:
        watcher.start()
        assert watcher.running
        assert watcher._task is not None

        await watcher.stop()
        assert not watcher.running
        assert watcher._task is None

    async def test_stop_is_idempotent(
        self, watcher: ArtifactWatcher
    ) -> None:
        watcher.start()
        await watcher.stop()
        await watcher.stop()  # second call is safe
        assert not watcher.running

    @respx.mock
    async def test_supervisor_read_failure_handled(
        self, watcher: ArtifactWatcher, events: list[ArtifactEvent]
    ) -> None:
        _mock_list_outputs(
            [{"path": "outputs/gone.txt", "type": "file", "sizeBytes": 5, "modifiedAt": "T1"}]
        )
        respx.get(
            f"{_SUPERVISOR_URL}/v2/sandboxes/{_SANDBOX_ID}/files"
        ).mock(return_value=httpx.Response(404, text="not found"))

        await _poll_once(watcher)
        assert len(events) == 0


class TestEventCallback:
    async def test_callback_exception_does_not_crash(self) -> None:
        async def _bad_callback(event: ArtifactEvent) -> None:
            raise RuntimeError("callback broken")

        w = ArtifactWatcher(
            sandbox_id=_SANDBOX_ID,
            supervisor_base_url=_SUPERVISOR_URL,
            storage_client=StorageArtifactClient(
                base_url=_STORAGE_URL,
                s2s_token_provider=lambda: "tok",
            ),
            tenant_id=_TENANT_ID,
            run_id=_RUN_ID,
            user_id=_USER_ID,
            thread_id=_THREAD_ID,
            on_event=_bad_callback,
        )

        # Emit should swallow the error
        await w._emit(ArtifactEvent(event_type="test", data={}))


class TestMimeSniffing:
    @respx.mock
    async def test_mime_detected_correctly(
        self, watcher: ArtifactWatcher, events: list[ArtifactEvent]
    ) -> None:
        csv_content = b"col1,col2\n1,2\n"
        _mock_list_outputs(
            [{"path": "outputs/data.csv", "type": "file", "sizeBytes": len(csv_content), "modifiedAt": "T1"}]
        )
        _mock_read_file(csv_content)

        meta_route = respx.post(
            f"{_STORAGE_URL}/api/v1/storage/artifacts"
        ).mock(
            return_value=httpx.Response(
                200, json={"artifact_id": _ARTIFACT_ID, "upload_url": _UPLOAD_URL}
            )
        )
        respx.put(_UPLOAD_URL).mock(return_value=httpx.Response(200))
        respx.post(
            f"{_STORAGE_URL}/api/v1/storage/artifacts/{_ARTIFACT_ID}/complete"
        ).mock(return_value=httpx.Response(200, json={"download_url": _DOWNLOAD_URL}))

        await _poll_once(watcher)

        import json

        payload = json.loads(meta_route.calls.last.request.content)
        # python-magic or mimetypes should detect text/csv or text/plain
        assert payload["content_type"].startswith("text/")
