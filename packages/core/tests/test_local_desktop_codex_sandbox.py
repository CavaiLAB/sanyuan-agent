"""Security invariants for the trusted local Codex sandbox launcher."""

from __future__ import annotations

import asyncio
import hashlib
import os
import signal
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from sanyuan_agent.cli import app
from sanyuan_agent.local_desktop import (
    PROFILE_READ_ONLY,
    PROFILE_WORKSPACE,
    CodexSandbox,
    LocalSandboxError,
    SandboxNotEnforcedError,
    SandboxOutputLimitError,
    SandboxUnavailableError,
    build_sandbox_argv,
)
from sanyuan_agent.local_desktop import codex_sandbox as sandbox_module
from sanyuan_agent.local_desktop.probe import run_probe
from typer.testing import CliRunner

SandboxFactory = Callable[[], CodexSandbox]


def _native_file(path: Path, payload: bytes = b"trusted") -> tuple[str, str]:
    path.write_bytes(b"\x7fELF" + payload)
    path.chmod(0o700)
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def sandbox_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SandboxFactory:
    codex_bin, codex_sha = _native_file(tmp_path / "codex", b"codex")
    probe_bin, probe_sha = _native_file(tmp_path / "probe", b"probe")
    control = tmp_path / "control"
    workspace = control / "workspace"
    control.mkdir(mode=0o700)
    workspace.mkdir(mode=0o700)
    control_stat = control.stat()
    workspace_stat = workspace.stat()
    monkeypatch.setattr(
        sandbox_module,
        "_validate_control_layout",
        lambda _control, _workspace: (
            control,
            (control_stat.st_dev, control_stat.st_ino),
            workspace,
            (workspace_stat.st_dev, workspace_stat.st_ino),
        ),
    )

    def _factory() -> CodexSandbox:
        return CodexSandbox(
            codex_bin=codex_bin,
            codex_sha256=codex_sha,
            probe_bin=probe_bin,
            probe_sha256=probe_sha,
            probe_args_prefix=("sandbox-probe",),
            control_root=str(control),
            authorized_workspace_root=str(workspace),
        )

    return _factory


def test_build_argv_is_explicit_and_shell_free() -> None:
    assert build_sandbox_argv(
        "/bundle/codex",
        profile=PROFILE_WORKSPACE,
        command=["/bundle/probe", "workspace"],
        cwd="/run/workspace",
    ) == [
        "/bundle/codex",
        "sandbox",
        "--permissions-profile",
        ":workspace",
        "--cd",
        ".",
        "--",
        "/bundle/probe",
        "workspace",
    ]


@pytest.mark.parametrize("profile", ["danger-full-access", "custom-profile", ""])
def test_build_argv_rejects_non_builtin_profile(profile: str) -> None:
    with pytest.raises(ValueError, match="unsupported sandbox profile"):
        build_sandbox_argv("codex", profile=profile, command=["echo"], cwd="/w")


def test_build_argv_rejects_empty_or_nul_values() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        build_sandbox_argv("codex", profile=PROFILE_READ_ONLY, command=[], cwd="/w")
    with pytest.raises(ValueError, match="NUL"):
        build_sandbox_argv("codex", profile=PROFILE_READ_ONLY, command=["a\x00b"], cwd="/w")
    with pytest.raises(ValueError, match="cwd"):
        build_sandbox_argv("codex", profile=PROFILE_READ_ONLY, command=["echo"], cwd="")


def test_script_launcher_is_rejected(tmp_path: Path) -> None:
    script = tmp_path / "codex"
    script.write_text("#!/usr/bin/env node\n", encoding="utf-8")
    script.chmod(0o700)
    digest = hashlib.sha256(script.read_bytes()).hexdigest()
    with pytest.raises(SandboxUnavailableError, match="native executable"):
        sandbox_module._resolve_trusted_native_binary(str(script), digest, label="codex_bin")


def test_digest_mismatch_is_rejected(tmp_path: Path) -> None:
    binary, _digest = _native_file(tmp_path / "codex")
    with pytest.raises(SandboxUnavailableError, match="SHA-256"):
        sandbox_module._resolve_trusted_native_binary(binary, "0" * 64, label="codex_bin")


def test_staging_rejects_symlink_source_and_digest_mismatch(tmp_path: Path) -> None:
    binary, digest = _native_file(tmp_path / "codex")
    symlink = tmp_path / "codex-link"
    symlink.symlink_to(binary)
    bin_dir = tmp_path / "bin"

    with pytest.raises(SandboxUnavailableError, match="non-symlink"):
        sandbox_module._stage_trusted_native_binary(
            str(symlink), digest, label="codex_bin", bin_dir=bin_dir
        )
    with pytest.raises(SandboxUnavailableError, match="SHA-256"):
        sandbox_module._stage_trusted_native_binary(
            binary, "0" * 64, label="codex_bin", bin_dir=bin_dir
        )


def test_binary_replacement_after_init_is_rejected(
    sandbox_factory: SandboxFactory,
) -> None:
    sandbox = sandbox_factory()
    sandbox._codex_path.unlink()
    sandbox._codex_path.write_bytes(b"\x7fELFreplaced")
    sandbox._codex_path.chmod(0o500)
    with pytest.raises(SandboxUnavailableError, match="SHA-256"):
        sandbox._require_trusted_assets()


def test_sources_are_copied_into_control_root_before_use(
    sandbox_factory: SandboxFactory,
    tmp_path: Path,
) -> None:
    sandbox = sandbox_factory()
    assert sandbox._codex_path.is_relative_to(sandbox._control_root / "bin")
    assert sandbox._probe_path.is_relative_to(sandbox._control_root / "bin")

    (tmp_path / "codex").write_bytes(b"\x7fELFsource-replaced")
    (tmp_path / "probe").write_bytes(b"\x7fELFsource-replaced")
    sandbox._require_trusted_assets()


def test_launcher_environment_is_allowlist_not_caller_blacklist(
    sandbox_factory: SandboxFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NODE_OPTIONS", "--require=/tmp/evil.js")
    monkeypatch.setenv("PYTHONPATH", "/tmp/evil")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("DATABASE_URL", "postgres://secret")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/agent.sock")
    monkeypatch.setenv("PATH", "/tmp/attacker-bin")
    sandbox = sandbox_factory()
    assert sandbox._launcher_env["PATH"] == "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
    assert Path(sandbox._launcher_env["CODEX_HOME"]).is_relative_to(sandbox._control_root)
    assert not Path(sandbox._launcher_env["CODEX_HOME"]).is_relative_to(sandbox._workspace_root)
    assert Path(sandbox._launcher_env["HOME"]).is_relative_to(sandbox._control_root)
    assert not Path(sandbox._launcher_env["HOME"]).is_relative_to(sandbox._workspace_root)
    assert all(key not in sandbox._launcher_env for key in ("TEMP", "TMP", "TMPDIR"))
    for forbidden in (
        "NODE_OPTIONS",
        "PYTHONPATH",
        "OPENAI_API_KEY",
        "DATABASE_URL",
        "SSH_AUTH_SOCK",
    ):
        assert forbidden not in sandbox._launcher_env


def test_control_layout_rejects_broad_and_temp_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    with pytest.raises(ValueError, match="system temp"):
        sandbox_module._validate_control_layout(str(tmp_path), str(workspace))
    with pytest.raises(ValueError, match="too broad"):
        sandbox_module._validate_control_layout(str(Path.home()), str(Path.home()))
    with pytest.raises(ValueError, match="too broad"):
        sandbox_module._validate_control_layout(Path.home().anchor, Path.home().anchor)


def test_control_layout_requires_direct_private_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sandbox_module, "_is_within", lambda _path, _root: False)
    control = tmp_path / "control"
    workspace = tmp_path / "workspace"
    control.mkdir(mode=0o700)
    workspace.mkdir(mode=0o700)
    with pytest.raises(ValueError, match="direct child"):
        sandbox_module._validate_control_layout(str(control), str(workspace))

    workspace.rmdir()
    workspace = control / "workspace"
    workspace.mkdir(mode=0o700)
    control.chmod(0o755)
    with pytest.raises(ValueError, match="control_root must have mode 0700"):
        sandbox_module._validate_control_layout(str(control), str(workspace))


def test_non_posix_fails_closed_before_launcher_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sandbox_module.os, "name", "nt")
    with pytest.raises(SandboxUnavailableError, match="contained process launcher"):
        CodexSandbox(
            codex_bin=str(tmp_path / "unused"),
            codex_sha256="0" * 64,
            probe_bin=str(tmp_path / "unused"),
            probe_sha256="0" * 64,
            probe_args_prefix=("sandbox-probe",),
            control_root=str(tmp_path),
            authorized_workspace_root=str(tmp_path),
        )


def test_probe_prefix_supports_frozen_sidecar(sandbox_factory: SandboxFactory) -> None:
    sandbox = sandbox_factory()
    command = sandbox._probe_command(
        "/Applications/iClaw.app/Contents/Resources/neuroagent",
        mode="workspace",
        nonce="a" * 32,
        inside=Path("/run/inside"),
        outside=Path("/run/outside"),
    )
    assert command == [
        "/Applications/iClaw.app/Contents/Resources/neuroagent",
        "sandbox-probe",
        "workspace",
        "a" * 32,
        "/run/inside",
        "--outside",
        "/run/outside",
    ]


def _probe_spawn(*, outside_writes: bool = False, nonzero: bool = False):
    async def _spawn(argv: list[str], *, cwd: str, timeout_ms: int) -> tuple[int, bytes, bytes]:
        if "read-only" in argv:
            index = argv.index("read-only")
            nonce = argv[index + 1]
            tokens = [
                f"ICLAW_PROBE_{nonce}_RAN",
                f"ICLAW_PROBE_{nonce}_READ_ONLY_DENIED",
            ]
            return (2 if nonzero else 0), ("\n".join(tokens) + "\n").encode(), b""

        index = argv.index("workspace")
        nonce = argv[index + 1]
        inside = Path(argv[index + 2])
        assert argv[index + 3] == "--outside"
        outside = Path(argv[index + 4])
        inside.write_bytes(b"")
        outside_token = "WROTE" if outside_writes else "DENIED"
        if outside_writes:
            outside.write_bytes(b"")
        tokens = [
            f"ICLAW_PROBE_{nonce}_RAN",
            f"ICLAW_PROBE_{nonce}_INSIDE_WROTE",
            f"ICLAW_PROBE_{nonce}_OUTSIDE_{outside_token}",
        ]
        return 0, ("\n".join(tokens) + "\n").encode(), b""

    return _spawn


async def test_probe_proves_read_only_and_workspace_boundaries(
    sandbox_factory: SandboxFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = sandbox_factory()
    monkeypatch.setattr(sandbox, "_spawn", _probe_spawn())
    assert await sandbox.verify_enforcement() is True
    assert sandbox._probe_diagnostic is None


async def test_fixed_fake_tokens_cannot_satisfy_random_nonce(
    sandbox_factory: SandboxFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = sandbox_factory()

    async def _fake(argv: list[str], *, cwd: str, timeout_ms: int) -> tuple[int, bytes, bytes]:
        return 0, b"ICLAW_PROBE_RAN\nICLAW_PROBE_DENIED\n", b""

    monkeypatch.setattr(sandbox, "_spawn", _fake)
    assert await sandbox.verify_enforcement() is False


async def test_nonzero_probe_exit_is_rejected(
    sandbox_factory: SandboxFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = sandbox_factory()
    monkeypatch.setattr(sandbox, "_spawn", _probe_spawn(nonzero=True))
    assert await sandbox.verify_enforcement() is False


async def test_workspace_outside_write_is_rejected(
    sandbox_factory: SandboxFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = sandbox_factory()
    monkeypatch.setattr(sandbox, "_spawn", _probe_spawn(outside_writes=True))
    assert await sandbox.verify_enforcement() is False
    assert "workspace boundary" in (sandbox._probe_diagnostic or "")


async def test_unenforced_probe_never_runs_user_command(
    sandbox_factory: SandboxFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = sandbox_factory()
    calls: list[list[str]] = []

    async def _fake(argv: list[str], *, cwd: str, timeout_ms: int) -> tuple[int, bytes, bytes]:
        calls.append(argv)
        return 0, b"fake fixed tokens\n", b""

    monkeypatch.setattr(sandbox, "_spawn", _fake)
    with pytest.raises(SandboxNotEnforcedError):
        await sandbox.run(["touch", "must-not-run"])
    assert len(calls) == 1
    assert "touch" not in calls[0]


async def test_run_is_bound_to_relative_cwd(
    sandbox_factory: SandboxFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox = sandbox_factory()
    subdir = sandbox._workspace_root / "subdir"
    subdir.mkdir()

    async def _enforced(*, force: bool = False) -> bool:
        return True

    recorded: dict[str, Any] = {}

    async def _spawn(argv: list[str], *, cwd: str, timeout_ms: int) -> tuple[int, bytes, bytes]:
        recorded.update(argv=argv, cwd=cwd, timeout_ms=timeout_ms)
        return 0, b"ok", b""

    monkeypatch.setattr(sandbox, "_verify_enforcement_locked", _enforced)
    monkeypatch.setattr(sandbox, "_spawn", _spawn)
    result = await sandbox.run(["echo", "ok"], relative_cwd="subdir")
    assert recorded["cwd"] == str(subdir)
    assert result["enforced"] is True

    with pytest.raises(ValueError, match="safe relative"):
        await sandbox.run(["echo"], relative_cwd="/etc")
    with pytest.raises(ValueError, match="escapes"):
        await sandbox.run(["echo"], relative_cwd="../")


def test_probe_function_and_hidden_cli(tmp_path: Path) -> None:
    inside = tmp_path / "inside.txt"
    nonce = "a" * 32
    assert run_probe("read-only", nonce, inside) == 0
    assert inside.exists()
    assert run_probe("read-only", "bad-nonce", inside) == 2

    runner = CliRunner()
    other = tmp_path / "other.txt"
    result = runner.invoke(
        app,
        [
            "sandbox-probe",
            "workspace",
            nonce,
            str(inside),
            "--outside",
            str(other),
        ],
    )
    assert result.exit_code == 0
    assert f"ICLAW_PROBE_{nonce}_INSIDE_WROTE" in result.output
    assert f"ICLAW_PROBE_{nonce}_OUTSIDE_WROTE" in result.output


class _FakeProcess:
    def __init__(
        self,
        output: tuple[bytes, bytes] | None,
        *,
        pid: int = 424242,
        wait_event: asyncio.Event | None = None,
    ) -> None:
        self.pid = pid
        self.returncode: int | None = None
        self.waited = False
        self.wait_event = wait_event
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        if output is not None:
            self.stdout.feed_data(output[0])
            self.stdout.feed_eof()
            self.stderr.feed_data(output[1])
            self.stderr.feed_eof()

    def kill(self) -> None:
        self.returncode = -9

    async def wait(self) -> int:
        self.waited = True
        if self.wait_event is not None:
            await self.wait_event.wait()
        if self.returncode is None:
            self.returncode = 0
        return self.returncode or 0


async def test_success_reaps_process_group_and_cleans_unique_temp(
    sandbox_factory: SandboxFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc = _FakeProcess((b"ok", b"warning"))
    seen_tmp: list[Path] = []

    async def _create(*args: Any, **kwargs: Any) -> _FakeProcess:
        env = kwargs["env"]
        assert env["TEMP"] == env["TMP"] == env["TMPDIR"]
        run_tmp = Path(env["TMPDIR"])
        assert run_tmp.parent == sandbox._workspace_root
        assert run_tmp.is_dir()
        seen_tmp.append(run_tmp)
        return proc

    killed: list[tuple[int, int]] = []

    def _killpg(pid: int, sig: int) -> None:
        killed.append((pid, sig))

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _create)
    monkeypatch.setattr(os, "killpg", _killpg)
    sandbox = sandbox_factory()
    result = await sandbox._spawn(["codex"], cwd=str(sandbox._workspace_root), timeout_ms=1_000)
    assert result == (0, b"ok", b"warning")
    assert killed == [(proc.pid, signal.SIGKILL)]
    assert proc.waited
    assert len(seen_tmp) == 1
    assert not seen_tmp[0].exists()


async def test_timeout_terminates_process_group(
    sandbox_factory: SandboxFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc = _FakeProcess(None)

    async def _create(*args: Any, **kwargs: Any) -> _FakeProcess:
        assert kwargs["start_new_session"] is True
        return proc

    killed: list[tuple[int, int]] = []

    def _killpg(pid: int, sig: int) -> None:
        killed.append((pid, sig))
        proc.returncode = -sig

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _create)
    monkeypatch.setattr(os, "killpg", _killpg)
    sandbox = sandbox_factory()
    with pytest.raises(LocalSandboxError, match="timed out"):
        await sandbox._spawn(["codex"], cwd=str(sandbox._workspace_root), timeout_ms=1)
    assert killed == [(proc.pid, signal.SIGKILL)]
    assert proc.waited


async def test_timeout_also_bounds_process_wait_after_pipes_close(
    sandbox_factory: SandboxFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    wait_event = asyncio.Event()
    proc = _FakeProcess((b"", b""), wait_event=wait_event)

    async def _create(*args: Any, **kwargs: Any) -> _FakeProcess:
        return proc

    killed: list[tuple[int, int]] = []

    def _killpg(pid: int, sig: int) -> None:
        killed.append((pid, sig))
        proc.returncode = -sig
        wait_event.set()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _create)
    monkeypatch.setattr(os, "killpg", _killpg)
    sandbox = sandbox_factory()
    with pytest.raises(LocalSandboxError, match="timed out"):
        await asyncio.wait_for(
            sandbox._spawn(["codex"], cwd=str(sandbox._workspace_root), timeout_ms=1),
            timeout=0.2,
        )
    assert killed == [(proc.pid, signal.SIGKILL)]
    assert proc.waited


async def test_cancellation_terminates_process_group_even_if_parent_exited(
    sandbox_factory: SandboxFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = asyncio.Event()
    proc = _FakeProcess(None)
    proc.returncode = 0  # descendant still holds stdout/stderr after launcher exit

    async def _create(*args: Any, **kwargs: Any) -> _FakeProcess:
        started.set()
        return proc

    killed: list[tuple[int, int]] = []

    def _killpg(pid: int, sig: int) -> None:
        killed.append((pid, sig))

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _create)
    monkeypatch.setattr(os, "killpg", _killpg)
    sandbox = sandbox_factory()
    task = asyncio.create_task(
        sandbox._spawn(["codex"], cwd=str(sandbox._workspace_root), timeout_ms=10_000)
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert killed == [(proc.pid, signal.SIGKILL)]
    assert proc.waited


async def test_output_limit_terminates_process_group(
    sandbox_factory: SandboxFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc = _FakeProcess((b"12345", b""))

    async def _create(*args: Any, **kwargs: Any) -> _FakeProcess:
        return proc

    killed: list[tuple[int, int]] = []

    def _killpg(pid: int, sig: int) -> None:
        killed.append((pid, sig))
        proc.returncode = -sig

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _create)
    monkeypatch.setattr(os, "killpg", _killpg)
    monkeypatch.setattr(sandbox_module, "MAX_CAPTURE_BYTES", 4)
    sandbox = sandbox_factory()
    with pytest.raises(SandboxOutputLimitError, match="stdout exceeded 4 bytes"):
        await sandbox._spawn(["codex"], cwd=str(sandbox._workspace_root), timeout_ms=1_000)
    assert killed == [(proc.pid, signal.SIGKILL)]
    assert proc.waited
