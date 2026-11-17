"""Agent-facing provider over the fail-closed local Codex sandbox."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
from sanyuan_agent.local_desktop import (
    LocalDesktopProvider,
    SandboxNotEnforcedError,
    SandboxUnavailableError,
)
from sanyuan_agent.local_desktop import provider as provider_module


class _FakeCodexSandbox:
    def __init__(self, root: Path, *, enforced: bool = True) -> None:
        self.root = root
        info = root.stat()
        self.identity = (info.st_dev, info.st_ino)
        self.enforced = enforced
        self.commands: list[list[str]] = []
        self.run_started: asyncio.Event | None = None
        self.run_cancelled: asyncio.Event | None = None
        self.allow_cancel_finish: asyncio.Event | None = None
        self.run_cleanup_finished: asyncio.Event | None = None

    async def verify_enforcement(self, *, force: bool = False) -> bool:
        return self.enforced

    def require_workspace_binding(self) -> tuple[Path, tuple[int, int]]:
        return self.root, self.identity

    async def run(self, command: list[str], **_kwargs: Any) -> dict[str, Any]:
        self.commands.append(command)
        if self.run_started is not None:
            self.run_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                if self.run_cancelled is not None:
                    self.run_cancelled.set()
                if self.allow_cancel_finish is not None:
                    await self.allow_cancel_finish.wait()
                if self.run_cleanup_finished is not None:
                    self.run_cleanup_finished.set()
                raise
        return {
            "stdout": "ok",
            "stderr": "",
            "exitCode": 0,
            "durationMs": 1,
            "sandboxProfile": ":workspace",
            "enforced": True,
        }


@pytest.fixture
def bound(tmp_path: Path) -> tuple[LocalDesktopProvider, _FakeCodexSandbox]:
    root = tmp_path / "workspace"
    root.mkdir(mode=0o700)
    sandbox = _FakeCodexSandbox(root)
    return LocalDesktopProvider(sandbox), sandbox  # type: ignore[arg-type]


async def test_create_session_probes_and_exposes_only_bound_workspace(
    bound: tuple[LocalDesktopProvider, _FakeCodexSandbox],
) -> None:
    provider, sandbox = bound
    session = await provider.create_session()
    assert session.workdir == str(sandbox.root)
    with pytest.raises(ValueError, match="active session"):
        await provider.create_session()

    await provider.close_session(session.id)
    results = await asyncio.gather(
        provider.create_session(), provider.create_session(), return_exceptions=True
    )
    assert sum(not isinstance(result, Exception) for result in results) == 1
    assert sum(isinstance(result, ValueError) for result in results) == 1


async def test_create_session_fails_closed_when_probe_fails(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir(mode=0o700)
    provider = LocalDesktopProvider(  # type: ignore[arg-type]
        _FakeCodexSandbox(root, enforced=False)
    )
    with pytest.raises(SandboxNotEnforcedError):
        await provider.create_session()


async def test_shell_is_fixed_argv_and_cancellation_propagates(
    bound: tuple[LocalDesktopProvider, _FakeCodexSandbox],
) -> None:
    provider, sandbox = bound
    session = await provider.create_session()
    result = await provider.run_shell(session.id, "printf ok | sed s/o/O/")
    assert result.success is True
    assert sandbox.commands == [["/bin/sh", "-c", "printf ok | sed s/o/O/"]]

    sandbox.run_started = asyncio.Event()
    task = asyncio.create_task(provider.run_shell(session.id, "sleep 100"))
    await sandbox.run_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_close_cancels_in_flight_run_before_new_session(
    bound: tuple[LocalDesktopProvider, _FakeCodexSandbox],
) -> None:
    provider, sandbox = bound
    session = await provider.create_session()
    sandbox.run_started = asyncio.Event()
    run_task = asyncio.create_task(provider.run_shell(session.id, "sleep 100"))
    await sandbox.run_started.wait()

    await provider.close_session(session.id)
    with pytest.raises(asyncio.CancelledError):
        await run_task
    replacement = await provider.create_session()
    assert replacement.id != session.id


async def test_cancelled_close_still_drains_run_and_releases_session(
    bound: tuple[LocalDesktopProvider, _FakeCodexSandbox],
) -> None:
    provider, sandbox = bound
    session = await provider.create_session()
    sandbox.run_started = asyncio.Event()
    sandbox.run_cancelled = asyncio.Event()
    sandbox.allow_cancel_finish = asyncio.Event()
    run_task = asyncio.create_task(provider.run_shell(session.id, "sleep 100"))
    await sandbox.run_started.wait()

    close_task = asyncio.create_task(provider.close_session(session.id))
    await sandbox.run_cancelled.wait()
    close_task.cancel()
    await asyncio.sleep(0)
    close_task.cancel()
    sandbox.allow_cancel_finish.set()
    with pytest.raises(asyncio.CancelledError):
        await close_task
    with pytest.raises(asyncio.CancelledError):
        await run_task

    replacement = await provider.create_session()
    assert replacement.id != session.id


async def test_cancelled_run_then_close_does_not_interrupt_process_cleanup(
    bound: tuple[LocalDesktopProvider, _FakeCodexSandbox],
) -> None:
    provider, sandbox = bound
    session = await provider.create_session()
    sandbox.run_started = asyncio.Event()
    sandbox.run_cancelled = asyncio.Event()
    sandbox.allow_cancel_finish = asyncio.Event()
    sandbox.run_cleanup_finished = asyncio.Event()
    run_task = asyncio.create_task(provider.run_shell(session.id, "sleep 100"))
    await sandbox.run_started.wait()

    run_task.cancel()
    await sandbox.run_cancelled.wait()
    close_task = asyncio.create_task(provider.close_session(session.id))
    await asyncio.sleep(0)
    assert not close_task.done()

    sandbox.allow_cancel_finish.set()
    await close_task
    with pytest.raises(asyncio.CancelledError):
        await run_task
    assert sandbox.run_cleanup_finished.is_set()


async def test_repeated_run_cancellation_still_drains_process_cleanup(
    bound: tuple[LocalDesktopProvider, _FakeCodexSandbox],
) -> None:
    provider, sandbox = bound
    session = await provider.create_session()
    sandbox.run_started = asyncio.Event()
    sandbox.run_cancelled = asyncio.Event()
    sandbox.allow_cancel_finish = asyncio.Event()
    sandbox.run_cleanup_finished = asyncio.Event()
    run_task = asyncio.create_task(provider.run_shell(session.id, "sleep 100"))
    await sandbox.run_started.wait()

    run_task.cancel()
    await sandbox.run_cancelled.wait()
    await asyncio.sleep(0)
    run_task.cancel()
    await asyncio.sleep(0)
    assert not run_task.done()

    sandbox.allow_cancel_finish.set()
    with pytest.raises(asyncio.CancelledError):
        await run_task
    assert sandbox.run_cleanup_finished.is_set()


async def test_shell_rejects_empty_nul_and_oversized_commands(
    bound: tuple[LocalDesktopProvider, _FakeCodexSandbox],
) -> None:
    provider, _sandbox = bound
    session = await provider.create_session()
    for command in ("", "a\x00b", "x" * (64 * 1024 + 1)):
        with pytest.raises(ValueError):
            await provider.run_shell(session.id, command)


async def test_file_ops_are_no_follow_and_workspace_bound(
    bound: tuple[LocalDesktopProvider, _FakeCodexSandbox], tmp_path: Path
) -> None:
    provider, sandbox = bound
    session = await provider.create_session()
    await provider.write_file(session.id, "nested/report.md", "hello")
    assert await provider.read_file(session.id, "nested/report.md") == "hello"
    assert await provider.list_dir(session.id, ".") == ["nested"]
    assert await provider.list_dir(session.id, "nested") == ["report.md"]

    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (sandbox.root / "link").symlink_to(outside)
    with pytest.raises(ValueError, match="no-follow"):
        await provider.read_file(session.id, "link")
    with pytest.raises(ValueError):
        await provider.write_file(session.id, "link", "replace")
    assert outside.read_text(encoding="utf-8") == "secret"

    (sandbox.root / "dir-link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(OSError):
        await provider.list_dir(session.id, "dir-link")

    hardlink = sandbox.root / "hardlink"
    hardlink.hardlink_to(outside)
    with pytest.raises(ValueError, match="regular file"):
        await provider.read_file(session.id, "hardlink")
    with pytest.raises(ValueError, match="regular file"):
        await provider.write_file(session.id, "hardlink", "replace")
    assert outside.read_text(encoding="utf-8") == "secret"


async def test_fifo_read_is_rejected_without_blocking_event_loop(
    bound: tuple[LocalDesktopProvider, _FakeCodexSandbox],
) -> None:
    provider, sandbox = bound
    session = await provider.create_session()
    os.mkfifo(sandbox.root / "pipe")
    with pytest.raises(ValueError, match="regular file"):
        await asyncio.wait_for(provider.read_file(session.id, "pipe"), timeout=0.2)


async def test_atomic_write_detects_destination_replacement(
    bound: tuple[LocalDesktopProvider, _FakeCodexSandbox],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, sandbox = bound
    session = await provider.create_session()
    real_replace = os.replace

    def attacked_replace(
        src: str,
        dst: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        real_replace(
            src,
            dst,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )
        (sandbox.root / dst).write_text("ATTACKER", encoding="utf-8")

    monkeypatch.setattr(provider_module.os, "replace", attacked_replace)
    with pytest.raises(SandboxUnavailableError, match="changed during atomic write"):
        await provider.write_file(session.id, "report.txt", "EXPECTED")
    assert (sandbox.root / "report.txt").read_text(encoding="utf-8") == "ATTACKER"


@pytest.mark.parametrize("path", ["/etc/passwd", "../outside", "a/../../b", "a\x00b"])
async def test_file_ops_reject_absolute_traversal_and_nul(
    bound: tuple[LocalDesktopProvider, _FakeCodexSandbox], path: str
) -> None:
    provider, _sandbox = bound
    session = await provider.create_session()
    with pytest.raises(ValueError):
        await provider.read_file(session.id, path)


async def test_text_limits_utf8_and_host_upload_are_fail_closed(
    bound: tuple[LocalDesktopProvider, _FakeCodexSandbox], tmp_path: Path
) -> None:
    provider, sandbox = bound
    session = await provider.create_session()
    (sandbox.root / "binary").write_bytes(b"\xff")
    with pytest.raises(ValueError, match="UTF-8"):
        await provider.read_file(session.id, "binary")

    (sandbox.root / "large").write_bytes(b"x" * (4 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="exceeds"):
        await provider.read_file(session.id, "large")
    with pytest.raises(ValueError, match="exceeds"):
        await provider.write_file(session.id, "too-large", "x" * (4 * 1024 * 1024 + 1))

    host_file = tmp_path / "host.txt"
    host_file.write_text("private", encoding="utf-8")
    with pytest.raises(PermissionError, match="Workspace v3"):
        await provider.upload_file(session.id, str(host_file), "copied.txt")
    assert not (sandbox.root / "copied.txt").exists()


async def test_workspace_identity_and_closed_session_are_rejected(
    bound: tuple[LocalDesktopProvider, _FakeCodexSandbox], tmp_path: Path
) -> None:
    provider, sandbox = bound
    session = await provider.create_session()
    old_root = tmp_path / "old-workspace"
    sandbox.root.rename(old_root)
    sandbox.root.mkdir(mode=0o700)
    with pytest.raises(SandboxUnavailableError, match="identity"):
        await provider.list_dir(session.id, ".")

    sandbox.root.rmdir()
    old_root.rename(sandbox.root)
    await provider.close_session(session.id)
    with pytest.raises(ValueError, match="not active"):
        await provider.list_dir(session.id, ".")


def test_missing_no_follow_capability_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "workspace"
    root.mkdir(mode=0o700)
    monkeypatch.setattr(provider_module, "_NOFOLLOW", 0)
    with pytest.raises(SandboxUnavailableError, match="no-follow"):
        LocalDesktopProvider(_FakeCodexSandbox(root))  # type: ignore[arg-type]
