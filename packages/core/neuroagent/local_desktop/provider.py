"""Fail-closed ``ISandboxProvider`` for a bound desktop workspace.

The provider is deliberately narrower than the cloud provider.  One instance is
bound to one :class:`CodexSandbox` and therefore one already-authorized workspace.
Shell commands always pass through the probed Codex launcher.  File operations use
no-follow ``openat`` traversal rooted at the verified workspace descriptor; host
uploads remain disabled until Workspace v3 supplies an immutable grant/materializer
instead of an arbitrary caller path.
"""

from __future__ import annotations

import asyncio
import os
import secrets
import stat
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import PurePosixPath

from neuroagent.local_desktop.codex_sandbox import (
    CodexSandbox,
    SandboxNotEnforcedError,
    SandboxRunResult,
    SandboxUnavailableError,
)
from neuroagent.sandbox.abc import ISandboxProvider, SandboxResult, SandboxSession

_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_MAX_TEXT_BYTES = 4 * 1024 * 1024
_MAX_COMMAND_BYTES = 64 * 1024
_SHELL_ARGV_PREFIX = ("/bin/sh", "-c")


class LocalDesktopProvider(ISandboxProvider):
    """One-session provider backed by a build-pinned :class:`CodexSandbox`.

    The provider owns no workspace lifecycle: Workspace v3 creates and seals the
    directory, while the surrounding local-run runtime removes it after every run.
    ``close_session`` therefore invalidates the handle but never recursively deletes
    a path supplied by another layer.
    """

    provider_kind = "local-desktop-codex"

    def __init__(self, sandbox: CodexSandbox) -> None:
        if (
            os.name != "posix"
            or not _DIRECTORY
            or not _NOFOLLOW
            or not _NONBLOCK
            or os.open not in os.supports_dir_fd
            or os.mkdir not in os.supports_dir_fd
            or os.stat not in os.supports_dir_fd
            or os.unlink not in os.supports_dir_fd
            or os.listdir not in os.supports_fd
        ):
            raise SandboxUnavailableError(
                "local desktop provider requires POSIX no-follow dir-fd filesystem APIs"
            )
        self._sandbox = sandbox
        self._active_session_id: str | None = None
        self._session_lock = asyncio.Lock()
        self._closing = False
        self._current_run_task: asyncio.Task[SandboxRunResult] | None = None

    async def create_session(self) -> SandboxSession:
        async with self._session_lock:
            if self._active_session_id is not None:
                raise ValueError("local desktop provider already has an active session")
            if not await self._sandbox.verify_enforcement():
                raise SandboxNotEnforcedError(
                    "local desktop sandbox enforcement could not be verified"
                )
            root, _identity = self._sandbox.require_workspace_binding()
            session_id = str(uuid.uuid4())
            self._active_session_id = session_id
            return SandboxSession(
                id=session_id,
                created_at=datetime.now(UTC),
                workdir=str(root),
            )

    async def run_shell(self, session_id: str, cmd: str) -> SandboxResult:
        if not isinstance(cmd, str) or not cmd.strip():
            raise ValueError("shell command must be a non-empty string")
        if "\x00" in cmd:
            raise ValueError("shell command must not contain NUL bytes")
        if len(cmd.encode("utf-8")) > _MAX_COMMAND_BYTES:
            raise ValueError(f"shell command exceeds {_MAX_COMMAND_BYTES} bytes")
        async with self._session_lock:
            self._require_session(session_id)
            if self._current_run_task is not None:
                raise ValueError("local desktop sandbox already has a running command")
            run_task = asyncio.create_task(self._sandbox.run([*_SHELL_ARGV_PREFIX, cmd]))
            self._current_run_task = run_task
        cancelled = False
        try:
            while not run_task.done():
                try:
                    await asyncio.shield(run_task)
                except asyncio.CancelledError:
                    cancelled = True
                    if not run_task.done() and not run_task.cancelling():
                        run_task.cancel()
            result = run_task.result()
            if cancelled:
                raise asyncio.CancelledError
        finally:
            async with self._session_lock:
                if self._current_run_task is run_task:
                    self._current_run_task = None
        return SandboxResult(
            success=result["exitCode"] == 0,
            stdout=result["stdout"],
            stderr=result["stderr"],
            exit_code=result["exitCode"],
        )

    async def read_file(self, session_id: str, path: str) -> str:
        self._require_session(session_id)
        parts = self._path_parts(path, allow_root=False)
        with self._open_parent(parts, create=False) as (parent_fd, leaf):
            try:
                fd = os.open(
                    leaf,
                    os.O_RDONLY | _NOFOLLOW | _NONBLOCK,
                    dir_fd=parent_fd,
                )
            except OSError as exc:
                raise ValueError("workspace file is unavailable or not no-follow") from exc
            try:
                info = os.fstat(fd)
                root_device = os.fstat(parent_fd).st_dev
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_nlink != 1
                    or info.st_dev != root_device
                ):
                    raise ValueError("workspace path is not a regular file")
                if info.st_size > _MAX_TEXT_BYTES:
                    raise ValueError(f"workspace text file exceeds {_MAX_TEXT_BYTES} bytes")
                chunks: list[bytes] = []
                size = 0
                while chunk := os.read(fd, min(64 * 1024, _MAX_TEXT_BYTES + 1 - size)):
                    size += len(chunk)
                    if size > _MAX_TEXT_BYTES:
                        raise ValueError(f"workspace text file exceeds {_MAX_TEXT_BYTES} bytes")
                    chunks.append(chunk)
            finally:
                os.close(fd)
        try:
            return b"".join(chunks).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("workspace file is not valid UTF-8 text") from exc

    async def write_file(self, session_id: str, path: str, content: str) -> None:
        self._require_session(session_id)
        if not isinstance(content, str):
            raise TypeError("workspace content must be text")
        payload = content.encode("utf-8")
        if len(payload) > _MAX_TEXT_BYTES:
            raise ValueError(f"workspace text content exceeds {_MAX_TEXT_BYTES} bytes")
        parts = self._path_parts(path, allow_root=False)
        with self._open_parent(parts, create=True) as (parent_fd, leaf):
            temporary = f".iclaw-{secrets.token_hex(8)}.tmp"
            fd = -1
            destination_fd = -1
            try:
                fd = os.open(
                    temporary,
                    os.O_RDWR | os.O_CREAT | os.O_EXCL | _NOFOLLOW,
                    0o600,
                    dir_fd=parent_fd,
                )
                view = memoryview(payload)
                while view:
                    written = os.write(fd, view)
                    view = view[written:]
                os.fsync(fd)
                temporary_info = os.fstat(fd)
                root_device = os.fstat(parent_fd).st_dev
                try:
                    current = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
                except FileNotFoundError:
                    current = None
                if current is not None and (
                    not stat.S_ISREG(current.st_mode)
                    or current.st_nlink != 1
                    or current.st_dev != root_device
                ):
                    raise ValueError("workspace destination is not a regular file")
                os.replace(
                    temporary,
                    leaf,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                )
                destination_fd = os.open(
                    leaf,
                    os.O_RDONLY | _NOFOLLOW | _NONBLOCK,
                    dir_fd=parent_fd,
                )
                destination_info = os.fstat(destination_fd)
                if (
                    not stat.S_ISREG(destination_info.st_mode)
                    or destination_info.st_nlink != 1
                    or destination_info.st_dev != root_device
                    or (destination_info.st_dev, destination_info.st_ino)
                    != (temporary_info.st_dev, temporary_info.st_ino)
                ):
                    raise SandboxUnavailableError(
                        "workspace destination changed during atomic write"
                    )
                os.lseek(fd, 0, os.SEEK_SET)
                verified = bytearray()
                while chunk := os.read(fd, min(64 * 1024, len(payload) + 1)):
                    verified.extend(chunk)
                    if len(verified) > len(payload):
                        break
                final_path_info = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
                if (
                    not secrets.compare_digest(bytes(verified), payload)
                    or (final_path_info.st_dev, final_path_info.st_ino)
                    != (destination_info.st_dev, destination_info.st_ino)
                    or final_path_info.st_nlink != 1
                ):
                    raise SandboxUnavailableError(
                        "workspace destination changed during atomic write"
                    )
                os.fsync(parent_fd)
            finally:
                if destination_fd >= 0:
                    os.close(destination_fd)
                if fd >= 0:
                    os.close(fd)
                try:
                    os.unlink(temporary, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass

    async def list_dir(self, session_id: str, path: str) -> list[str]:
        self._require_session(session_id)
        parts = self._path_parts(path, allow_root=True)
        with self._open_directory(parts, create=False) as directory_fd:
            return sorted(os.listdir(directory_fd))

    async def upload_file(self, session_id: str, local_path: str, remote_path: str) -> None:
        self._require_session(session_id)
        raise PermissionError(
            "host uploads require a sealed Workspace v3 source grant and materializer"
        )

    async def close_session(self, session_id: str) -> None:
        async with self._session_lock:
            self._require_session(session_id)
            self._closing = True
            run_task = self._current_run_task
            if run_task is not None and not run_task.done() and not run_task.cancelling():
                run_task.cancel()
        drain_task = asyncio.create_task(self._drain_close(session_id, run_task))
        cancelled = False
        while not drain_task.done():
            try:
                await asyncio.shield(drain_task)
            except asyncio.CancelledError:
                cancelled = True
        await drain_task
        if cancelled:
            raise asyncio.CancelledError

    async def _drain_close(
        self,
        session_id: str,
        run_task: asyncio.Task[SandboxRunResult] | None,
    ) -> None:
        if run_task is not None:
            await asyncio.gather(run_task, return_exceptions=True)
        async with self._session_lock:
            if self._active_session_id != session_id:
                raise ValueError("local desktop sandbox session changed while closing")
            self._current_run_task = None
            self._active_session_id = None
            self._closing = False

    def _require_session(self, session_id: str) -> None:
        if not session_id or session_id != self._active_session_id:
            raise ValueError("local desktop sandbox session is not active")
        if self._closing:
            raise ValueError("local desktop sandbox session is closing")

    @staticmethod
    def _path_parts(path: str, *, allow_root: bool) -> tuple[str, ...]:
        if not isinstance(path, str) or "\x00" in path:
            raise ValueError("workspace path must be text without NUL bytes")
        candidate = PurePosixPath(path)
        if candidate.is_absolute():
            raise ValueError("workspace path must be relative")
        parts = tuple(part for part in candidate.parts if part not in ("", "."))
        if any(part == ".." for part in parts):
            raise ValueError("workspace path traversal is forbidden")
        if not parts and not allow_root:
            raise ValueError("workspace file path must not be empty")
        if any(len(part.encode("utf-8")) > 255 for part in parts):
            raise ValueError("workspace path component is too long")
        return parts

    @contextmanager
    def _open_root(self) -> Iterator[int]:
        root, expected_identity = self._sandbox.require_workspace_binding()
        try:
            fd = os.open(root, os.O_RDONLY | _DIRECTORY | _NOFOLLOW)
        except OSError as exc:
            raise SandboxUnavailableError("authorized workspace root is unavailable") from exc
        try:
            info = os.fstat(fd)
            if (
                not stat.S_ISDIR(info.st_mode)
                or (
                    info.st_dev,
                    info.st_ino,
                )
                != expected_identity
            ):
                raise SandboxUnavailableError("authorized workspace identity changed")
            yield fd
        finally:
            os.close(fd)

    @contextmanager
    def _open_directory(self, parts: tuple[str, ...], *, create: bool) -> Iterator[int]:
        with self._open_root() as root_fd:
            root_device = os.fstat(root_fd).st_dev
            current_fd = os.dup(root_fd)
            try:
                for part in parts:
                    if create:
                        try:
                            os.mkdir(part, 0o700, dir_fd=current_fd)
                        except FileExistsError:
                            pass
                    next_fd = os.open(
                        part,
                        os.O_RDONLY | _DIRECTORY | _NOFOLLOW,
                        dir_fd=current_fd,
                    )
                    next_info = os.fstat(next_fd)
                    if next_info.st_dev != root_device:
                        os.close(next_fd)
                        raise SandboxUnavailableError(
                            "workspace path crosses a filesystem boundary"
                        )
                    os.close(current_fd)
                    current_fd = next_fd
                yield current_fd
            finally:
                os.close(current_fd)

    @contextmanager
    def _open_parent(self, parts: tuple[str, ...], *, create: bool) -> Iterator[tuple[int, str]]:
        with self._open_directory(parts[:-1], create=create) as parent_fd:
            yield parent_fd, parts[-1]


__all__ = ["LocalDesktopProvider"]
