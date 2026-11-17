"""ArtifactWatcher — monitors sandbox outputs/ for new/changed files.

When a file is detected the watcher:
1. Reads bytes from the Sandbox Supervisor API.
2. Validates (size ≤ 100 MB, extension not blacklisted).
3. Uploads to MinIO via *StorageArtifactClient.register_sandbox_artifact*.
4. Emits ``artifact.created.v1`` (or ``artifact.rejected.v1``) events
   through an async callback consumed by the SSE streaming layer.
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx

from sanyuan_agent.artifacts.client import ArtifactRef, StorageArtifactClient

logger = logging.getLogger(__name__)

_MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
_BLOCKED_EXTENSIONS: frozenset[str] = frozenset(
    {".so", ".exe", ".dll", ".dylib", ".bat", ".cmd", ".com", ".scr"}
)
_POLL_INTERVAL_DEFAULT = 2.0

# Type alias for the async event callback.
EventCallback = Callable[["ArtifactEvent"], Awaitable[None]]


# -- Optional python-magic import ----------------------------------------
try:
    import magic

    _HAS_MAGIC: bool = True
except ImportError:  # pragma: no cover - graceful fallback
    _HAS_MAGIC = False


# -- Public dataclasses ---------------------------------------------------


@dataclass(frozen=True, slots=True)
class ArtifactEvent:
    """Event emitted by the watcher."""

    event_type: str  # "artifact.created.v1" | "artifact.rejected.v1"
    data: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _FileSnapshot:
    """Last-known state of a single file in outputs/."""

    path: str
    size: int
    modified_at: str


# -- Watcher --------------------------------------------------------------


class ArtifactWatcher:
    """Poll the Sandbox Supervisor for changes in ``outputs/``.

    Parameters
    ----------
    sandbox_id:
        Sandbox identifier used in Supervisor API paths.
    supervisor_base_url:
        Root URL of the Sandbox Supervisor service.
    storage_client:
        An initialised :class:`StorageArtifactClient` for uploading to
        Kun.Storage / MinIO.
    tenant_id / run_id / user_id / thread_id:
        Context identifiers forwarded to Storage and SSE payloads.
    on_event:
        Async callback invoked for every ``ArtifactEvent``.
    supervisor_token:
        Optional JWT for authenticating with the Supervisor.
    poll_interval:
        Seconds between directory listing polls.
    max_file_size:
        Per-file size cap in bytes (default 100 MB).
    """

    def __init__(
        self,
        *,
        sandbox_id: str,
        supervisor_base_url: str,
        storage_client: StorageArtifactClient,
        tenant_id: str,
        run_id: str,
        user_id: str,
        thread_id: str,
        on_event: EventCallback,
        supervisor_token: str | None = None,
        poll_interval: float = _POLL_INTERVAL_DEFAULT,
        max_file_size: int = _MAX_FILE_SIZE,
    ) -> None:
        self._sandbox_id = sandbox_id
        self._supervisor_url = supervisor_base_url.rstrip("/")
        self._storage = storage_client
        self._tenant_id = tenant_id
        self._run_id = run_id
        self._user_id = user_id
        self._thread_id = thread_id
        self._on_event = on_event
        self._supervisor_token = supervisor_token
        self._poll_interval = poll_interval
        self._max_file_size = max_file_size

        self._snapshot: dict[str, _FileSnapshot] = {}
        self._task: asyncio.Task[None] | None = None
        self._running = False
        # Tracks paths already processed in *this* watcher lifetime to
        # avoid re-uploading on every poll when mtime is unstable.
        self._processed: set[str] = set()

    # -- lifecycle ---------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        """Start the background polling task."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        """Stop the watcher and cancel its background task."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    # -- internal ----------------------------------------------------------

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                await self._poll_once()
            except Exception:
                logger.exception("artifact watcher poll error")
            await asyncio.sleep(self._poll_interval)

    async def _poll_once(self) -> None:
        entries = await self._list_outputs()
        current: dict[str, _FileSnapshot] = {}

        for entry in entries:
            if entry.get("type") != "file":
                continue
            path: str = entry.get("path", "")
            size: int = entry.get("sizeBytes", 0) or 0
            mtime: str = entry.get("modifiedAt", "") or ""

            snap = _FileSnapshot(path=path, size=size, modified_at=mtime)
            current[path] = snap

            prev = self._snapshot.get(path)
            if prev is None or (prev.size != snap.size or prev.modified_at != snap.modified_at):
                if path not in self._processed:
                    await self._process_file(path, snap)

        self._snapshot = current

    # -- supervisor calls --------------------------------------------------

    def _supervisor_headers(self) -> dict[str, str]:
        hdrs: dict[str, str] = {}
        if self._supervisor_token:
            hdrs["Authorization"] = f"Bearer {self._supervisor_token}"
        return hdrs

    async def _list_outputs(self) -> list[dict[str, Any]]:
        url = (
            f"{self._supervisor_url}/v2/sandboxes/{self._sandbox_id}"
            f"/files/list"
        )
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                url,
                params={"path": "outputs/", "recursive": True},
                headers=self._supervisor_headers(),
            )
            resp.raise_for_status()
            body: Any = resp.json()
            entries: list[dict[str, Any]] = body.get("entries", [])
            return entries

    async def _read_file(self, file_path: str) -> bytes:
        url = (
            f"{self._supervisor_url}/v2/sandboxes/{self._sandbox_id}/files"
        )
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(
                url,
                params={"path": file_path},
                headers=self._supervisor_headers(),
            )
            resp.raise_for_status()
            return resp.content

    # -- processing --------------------------------------------------------

    async def _process_file(self, file_path: str, snap: _FileSnapshot) -> None:
        filename = file_path.rsplit("/", 1)[-1] if "/" in file_path else file_path

        # Extension blacklist
        if _get_extension(filename) in _BLOCKED_EXTENSIONS:
            logger.info("skipping blacklisted file: %s", file_path)
            await self._emit_rejected(file_path, snap.size, "blocked_extension")
            self._processed.add(file_path)
            return

        # Size gate (from listing metadata)
        if snap.size > self._max_file_size:
            logger.warning("file too large: %s (%d bytes)", file_path, snap.size)
            await self._emit_rejected(file_path, snap.size, "size_exceeded")
            self._processed.add(file_path)
            return

        try:
            content = await self._read_file(file_path)
        except Exception:
            logger.exception("failed to read sandbox file: %s", file_path)
            return

        # Re-check actual content size
        if len(content) > self._max_file_size:
            await self._emit_rejected(file_path, len(content), "size_exceeded")
            self._processed.add(file_path)
            return

        content_type = _sniff_mime(content, filename)

        try:
            ref = await self._storage.register_sandbox_artifact(
                tenant_id=self._tenant_id,
                content_bytes=content,
                content_type=content_type,
                filename=filename,
                run_id=self._run_id,
                user_id=self._user_id,
                origin_path=file_path,
            )
        except Exception:
            logger.exception("failed to register artifact: %s", file_path)
            return

        await self._emit_created(file_path, ref, content_type, len(content))
        self._processed.add(file_path)

    # -- events ------------------------------------------------------------

    async def _emit_created(
        self,
        origin_path: str,
        ref: ArtifactRef,
        mime: str,
        size: int,
    ) -> None:
        await self._emit(
            ArtifactEvent(
                event_type="artifact.created.v1",
                data={
                    "threadId": self._thread_id,
                    "artifactId": ref.id,
                    "objectKey": ref.filename,
                    "originPath": origin_path,
                    "size": size,
                    "mime": mime,
                    "downloadUrl": ref.url,
                },
            )
        )

    async def _emit_rejected(
        self, origin_path: str, size: int, reason: str
    ) -> None:
        await self._emit(
            ArtifactEvent(
                event_type="artifact.rejected.v1",
                data={
                    "threadId": self._thread_id,
                    "originPath": origin_path,
                    "size": size,
                    "reason": reason,
                },
            )
        )

    async def _emit(self, event: ArtifactEvent) -> None:
        try:
            result = self._on_event(event)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            logger.exception("artifact event callback error")


# -- helpers --------------------------------------------------------------


def _get_extension(filename: str) -> str:
    dot = filename.rfind(".")
    return filename[dot:].lower() if dot != -1 else ""


def _sniff_mime(content: bytes, filename: str) -> str:
    if _HAS_MAGIC:  # pragma: no branch
        mime: str = magic.from_buffer(content[:8192], mime=True)
        if mime:
            return mime
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or "application/octet-stream"
