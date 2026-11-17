"""Trusted, fail-closed local execution through ``codex sandbox``.

This primitive is deliberately narrower than a general subprocess wrapper:

* the Codex launcher and probe helper must be native executables whose SHA-256
  digests were pinned by the desktop build;
* the launcher receives an allow-listed environment, never caller overrides;
* every instance is bound to one explicit, non-temporary workspace root;
* before the first command, random-nonce behavior probes prove read-only write
  denial and workspace inside-write/outside-denial semantics;
* only the built-in ``:read-only`` and ``:workspace`` profiles are accepted.

The class currently supports POSIX process groups.  Windows must use a Job
Object-backed launcher before it is enabled; failing closed is safer than
pretending ``CREATE_NEW_PROCESS_GROUP`` contains descendants.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import re
import secrets
import shutil
import signal
import stat
import tempfile
import time
from pathlib import Path
from typing import Any, TypedDict

PROFILE_READ_ONLY = ":read-only"
PROFILE_WORKSPACE = ":workspace"
_ALLOWED_PROFILES = frozenset({PROFILE_READ_ONLY, PROFILE_WORKSPACE})
_PERMISSIONS_PROFILE_FLAG = "--permissions-profile"

DEFAULT_TIMEOUT_MS = 120_000
PROBE_TIMEOUT_MS = 30_000
MAX_CAPTURE_BYTES = 4 * 1024 * 1024

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_NATIVE_MAGICS = (
    b"\x7fELF",  # Linux ELF
    b"MZ",  # Windows PE
    b"\xfe\xed\xfa\xce",  # Mach-O 32-bit, big endian
    b"\xce\xfa\xed\xfe",  # Mach-O 32-bit, little endian
    b"\xfe\xed\xfa\xcf",  # Mach-O 64-bit, big endian
    b"\xcf\xfa\xed\xfe",  # Mach-O 64-bit, little endian
    b"\xca\xfe\xba\xbe",  # Mach-O universal
    b"\xbe\xba\xfe\xca",  # Mach-O universal, reversed
    b"\xca\xfe\xba\xbf",  # Mach-O universal 64-bit
    b"\xbf\xba\xfe\xca",  # Mach-O universal 64-bit, reversed
)

_LAUNCHER_ENV_KEYS = frozenset(
    {
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LOGNAME",
        "PATHEXT",
        "SYSTEMROOT",
        "TERM",
        "USER",
        "WINDIR",
    }
)


class SandboxRunResult(TypedDict):
    """Serializable result returned after an enforced sandbox execution."""

    stdout: str
    stderr: str
    exitCode: int
    durationMs: int
    sandboxProfile: str
    enforced: bool


class LocalSandboxError(Exception):
    """Base error for local sandbox execution."""


class SandboxUnavailableError(LocalSandboxError):
    """A trusted launcher, probe helper, or supported OS is unavailable."""


class SandboxNotEnforcedError(LocalSandboxError):
    """Behavior probes could not prove the requested isolation boundary."""


class SandboxOutputLimitError(LocalSandboxError):
    """A sandboxed command exceeded the bounded stdout/stderr capture limit."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_trusted_native_binary(
    path_value: str,
    expected_sha256: str,
    *,
    label: str,
) -> Path:
    expected = expected_sha256.lower()
    if not _SHA256_RE.fullmatch(expected):
        raise ValueError(f"{label}_sha256 must be 64 lowercase hex characters")

    try:
        requested = Path(path_value).expanduser()
        if not requested.is_absolute():
            requested = (Path.cwd() / requested).absolute()
        if requested.is_symlink():
            raise SandboxUnavailableError(f"trusted {label} must not be a symlink")
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        raise SandboxUnavailableError(f"trusted {label} is unavailable") from exc
    if resolved != requested:
        raise SandboxUnavailableError(
            f"trusted {label} must not traverse symlinked path components"
        )
    if not resolved.is_file():
        raise SandboxUnavailableError(f"trusted {label} is not a file")
    if os.name == "posix" and not os.access(resolved, os.X_OK):
        raise SandboxUnavailableError(f"trusted {label} is not executable")

    with resolved.open("rb") as stream:
        header = stream.read(8)
    if not any(header.startswith(magic) for magic in _NATIVE_MAGICS):
        raise SandboxUnavailableError(
            f"trusted {label} must be a native executable; script launchers are rejected"
        )
    if not hmac.compare_digest(_sha256(resolved), expected):
        raise SandboxUnavailableError(f"trusted {label} SHA-256 does not match build manifest")
    return resolved


def _stage_trusted_native_binary(
    source_value: str,
    expected_sha256: str,
    *,
    label: str,
    bin_dir: Path,
) -> Path:
    """Copy and hash one opened source inode into the sandbox-external control dir."""

    expected = expected_sha256.lower()
    if not _SHA256_RE.fullmatch(expected):
        raise ValueError(f"{label}_sha256 must be 64 lowercase hex characters")
    source = Path(source_value).expanduser()
    if not source.is_absolute():
        source = (Path.cwd() / source).absolute()

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        source_fd = os.open(source, flags)
    except OSError as exc:
        raise SandboxUnavailableError(
            f"trusted {label} source must be a readable non-symlink"
        ) from exc

    bin_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = bin_dir / f".{label}-{secrets.token_hex(8)}.tmp"
    destination = bin_dir / f"{label}-{expected}"
    digest = hashlib.sha256()
    try:
        source_stat = os.fstat(source_fd)
        if not stat.S_ISREG(source_stat.st_mode) or not (source_stat.st_mode & 0o111):
            raise SandboxUnavailableError(f"trusted {label} source is not executable")
        with os.fdopen(source_fd, "rb", closefd=True) as source_stream:
            source_fd = -1
            with temporary.open("xb") as destination_stream:
                first = True
                for chunk in iter(lambda: source_stream.read(1024 * 1024), b""):
                    if first:
                        first = False
                        if not any(chunk.startswith(magic) for magic in _NATIVE_MAGICS):
                            raise SandboxUnavailableError(
                                f"trusted {label} must be a native executable"
                            )
                    digest.update(chunk)
                    destination_stream.write(chunk)
                destination_stream.flush()
                os.fsync(destination_stream.fileno())
        if not hmac.compare_digest(digest.hexdigest(), expected):
            raise SandboxUnavailableError(f"trusted {label} SHA-256 does not match build manifest")
        temporary.chmod(0o500)
        os.replace(temporary, destination)
    finally:
        if source_fd >= 0:
            os.close(source_fd)
        if temporary.exists():
            temporary.unlink()
    return _resolve_trusted_native_binary(str(destination), expected, label=label)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_control_layout(
    control_value: str,
    workspace_value: str,
) -> tuple[Path, tuple[int, int], Path, tuple[int, int]]:
    try:
        control_requested = Path(control_value).expanduser().absolute()
        workspace_requested = Path(workspace_value).expanduser().absolute()
        if control_requested.is_symlink() or workspace_requested.is_symlink():
            raise ValueError("control_root and authorized_workspace_root must not be symlinks")
        control = control_requested.resolve(strict=True)
        workspace = workspace_requested.resolve(strict=True)
    except OSError as exc:
        raise ValueError("control_root and authorized_workspace_root must exist") from exc
    if control != control_requested or workspace != workspace_requested:
        raise ValueError("control/workspace paths must not traverse symlinked components")
    if not control.is_dir() or not workspace.is_dir():
        raise ValueError("control_root and authorized_workspace_root must be directories")

    anchor = Path(control.anchor).resolve()
    home = Path.home().resolve()
    if control in {anchor, home, home.parent} or workspace in {anchor, home, home.parent}:
        raise ValueError("control/workspace root is too broad")
    if workspace.parent != control:
        raise ValueError("authorized_workspace_root must be a direct child of control_root")

    temp_roots = {
        Path(tempfile.gettempdir()).resolve(),
        Path("/tmp").resolve(),
        Path("/private/tmp").resolve(),
    }
    if any(
        _is_within(path, temp_root) for path in (control, workspace) for temp_root in temp_roots
    ):
        raise ValueError("control/workspace roots must not be under a system temp directory")

    control_stat = control.stat()
    workspace_stat = workspace.stat()
    if control_stat.st_uid != os.getuid() or workspace_stat.st_uid != os.getuid():
        raise ValueError("control/workspace roots must be owned by the current user")
    if stat.S_IMODE(control_stat.st_mode) != 0o700:
        raise ValueError("control_root must have mode 0700")
    if stat.S_IMODE(workspace_stat.st_mode) != 0o700:
        raise ValueError("authorized_workspace_root must have mode 0700")
    return (
        control,
        (control_stat.st_dev, control_stat.st_ino),
        workspace,
        (workspace_stat.st_dev, workspace_stat.st_ino),
    )


def _private_directory(
    root: Path,
    name: str,
) -> tuple[Path, tuple[int, int]]:
    """Create or validate a fixed-name 0700 directory below a trusted root."""

    path = root / name
    try:
        path.mkdir(mode=0o700, exist_ok=True)
        if path.is_symlink():
            raise SandboxUnavailableError(f"private directory {name!r} is a symlink")
        resolved = path.resolve(strict=True)
        directory_stat = resolved.stat()
    except OSError as exc:
        raise SandboxUnavailableError(f"private directory {name!r} is unavailable") from exc
    if resolved != path or resolved.parent != root or not resolved.is_dir():
        raise SandboxUnavailableError(f"private directory {name!r} is unsafe")
    if directory_stat.st_uid != os.getuid():
        raise SandboxUnavailableError(
            f"private directory {name!r} is not owned by the current user"
        )
    if stat.S_IMODE(directory_stat.st_mode) != 0o700:
        raise SandboxUnavailableError(f"private directory {name!r} must have mode 0700")
    return resolved, (directory_stat.st_dev, directory_stat.st_ino)


def _require_private_directory(
    path: Path,
    root: Path,
    identity: tuple[int, int],
    *,
    label: str,
) -> None:
    try:
        if path.is_symlink():
            raise SandboxUnavailableError(f"{label} became a symlink")
        resolved = path.resolve(strict=True)
        directory_stat = resolved.stat()
    except OSError as exc:
        raise SandboxUnavailableError(f"{label} is unavailable") from exc
    if (
        resolved != path
        or resolved.parent != root
        or not resolved.is_dir()
        or directory_stat.st_uid != os.getuid()
        or stat.S_IMODE(directory_stat.st_mode) != 0o700
        or (directory_stat.st_dev, directory_stat.st_ino) != identity
    ):
        raise SandboxUnavailableError(f"{label} identity or permissions changed")


def build_sandbox_argv(
    codex_bin: str,
    *,
    profile: str,
    command: list[str],
    cwd: str,
) -> list[str]:
    """Build an argv-only invocation using a built-in permissions profile."""

    if profile not in _ALLOWED_PROFILES:
        raise ValueError(f"unsupported sandbox profile: {profile!r}")
    if not command:
        raise ValueError("command must not be empty")
    if any(not isinstance(part, str) or "\x00" in part for part in command):
        raise ValueError("command arguments must be strings without NUL bytes")
    if not cwd or "\x00" in cwd:
        raise ValueError("cwd must be a non-empty path without NUL bytes")
    # The subprocess is already started with the verified directory as its cwd.
    # Keeping Codex at "." avoids a second absolute-path lookup after exec.
    return [
        codex_bin,
        "sandbox",
        _PERMISSIONS_PROFILE_FLAG,
        profile,
        "--cd",
        ".",
        "--",
        *command,
    ]


class CodexSandbox:
    """Run commands inside one build-pinned, workspace-bound Codex sandbox."""

    def __init__(
        self,
        *,
        codex_bin: str,
        codex_sha256: str,
        probe_bin: str,
        probe_sha256: str,
        probe_args_prefix: tuple[str, ...],
        control_root: str,
        authorized_workspace_root: str,
        default_profile: str = PROFILE_WORKSPACE,
        default_timeout_ms: int = DEFAULT_TIMEOUT_MS,
    ) -> None:
        if os.name != "posix":
            raise SandboxUnavailableError(
                "local Codex sandbox is disabled until this OS has a contained process launcher"
            )
        if default_profile not in _ALLOWED_PROFILES:
            raise ValueError(f"unsupported sandbox profile: {default_profile!r}")
        if default_timeout_ms <= 0:
            raise ValueError("default_timeout_ms must be positive")
        if probe_args_prefix != ("sandbox-probe",):
            raise ValueError("probe_args_prefix must select the bundled sandbox-probe command")

        (
            self._control_root,
            self._control_identity,
            self._workspace_root,
            self._workspace_identity,
        ) = _validate_control_layout(control_root, authorized_workspace_root)

        bin_dir, bin_identity = _private_directory(self._control_root, "bin")
        codex_home, codex_home_identity = _private_directory(self._control_root, "codex-home")
        isolated_home, isolated_home_identity = _private_directory(self._control_root, "home")
        self._private_directories = (
            (bin_dir, self._control_root, bin_identity, "trusted binary directory"),
            (
                codex_home,
                self._control_root,
                codex_home_identity,
                "isolated CODEX_HOME",
            ),
            (
                isolated_home,
                self._control_root,
                isolated_home_identity,
                "isolated HOME",
            ),
        )

        self._codex_path = _stage_trusted_native_binary(
            codex_bin,
            codex_sha256,
            label="codex_bin",
            bin_dir=bin_dir,
        )
        self._codex_sha256 = codex_sha256.lower()
        self._probe_path = _stage_trusted_native_binary(
            probe_bin,
            probe_sha256,
            label="probe_bin",
            bin_dir=bin_dir,
        )
        self._probe_sha256 = probe_sha256.lower()
        self._probe_args_prefix = probe_args_prefix
        self._default_profile = default_profile
        self._default_timeout_ms = default_timeout_ms
        self._enforced: bool | None = None
        self._probe_diagnostic: str | None = None
        self._run_lock = asyncio.Lock()

        self._launcher_env = {
            key: value for key, value in os.environ.items() if key.upper() in _LAUNCHER_ENV_KEYS
        }
        self._launcher_env["CODEX_HOME"] = str(codex_home)
        self._launcher_env["HOME"] = str(isolated_home)
        self._launcher_env["PATH"] = "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

    def require_workspace_binding(self) -> tuple[Path, tuple[int, int]]:
        """Return the verified workspace path and its immutable filesystem identity.

        Local-desktop adapters must not rediscover the workspace from caller input.
        This method revalidates the build-pinned assets and private control layout,
        then exposes the exact ``(device, inode)`` pair captured at construction so
        an adapter can compare it with an opened no-follow directory descriptor.
        """

        self._require_trusted_assets()
        return self._workspace_root, self._workspace_identity

    def _require_trusted_assets(self) -> tuple[str, str]:
        codex = _resolve_trusted_native_binary(
            str(self._codex_path), self._codex_sha256, label="codex_bin"
        )
        probe = _resolve_trusted_native_binary(
            str(self._probe_path), self._probe_sha256, label="probe_bin"
        )

        try:
            (
                control,
                control_identity,
                workspace,
                workspace_identity,
            ) = _validate_control_layout(str(self._control_root), str(self._workspace_root))
        except ValueError as exc:
            raise SandboxUnavailableError(
                "sandbox control/workspace layout is unavailable"
            ) from exc
        if (
            control != self._control_root
            or control_identity != self._control_identity
            or workspace != self._workspace_root
            or workspace_identity != self._workspace_identity
        ):
            raise SandboxUnavailableError("sandbox control/workspace identity changed")
        for path, root, identity, label in self._private_directories:
            _require_private_directory(path, root, identity, label=label)
        return str(codex), str(probe)

    def _resolve_run_cwd(self, relative_cwd: str | None) -> str:
        if relative_cwd is None:
            return str(self._workspace_root)
        candidate = Path(relative_cwd)
        if candidate.is_absolute() or "\x00" in relative_cwd:
            raise ValueError("relative_cwd must be a safe relative path")
        try:
            resolved = (self._workspace_root / candidate).resolve(strict=True)
        except OSError as exc:
            raise ValueError("relative_cwd must exist") from exc
        if not resolved.is_dir() or not _is_within(resolved, self._workspace_root):
            raise ValueError("relative_cwd escapes the authorized workspace")
        return str(resolved)

    def _probe_command(
        self,
        probe_bin: str,
        *,
        mode: str,
        nonce: str,
        inside: Path,
        outside: Path | None = None,
    ) -> list[str]:
        command = [
            probe_bin,
            *self._probe_args_prefix,
            mode,
            nonce,
            str(inside),
        ]
        if outside is not None:
            command.extend(["--outside", str(outside)])
        return command

    async def verify_enforcement(self, *, force: bool = False) -> bool:
        """Prove read-only and workspace boundaries, then cache the verdict."""

        async with self._run_lock:
            return await self._verify_enforcement_locked(force=force)

    async def _verify_enforcement_locked(self, *, force: bool = False) -> bool:
        if self._enforced is not None and not force:
            return self._enforced
        codex, probe = self._require_trusted_assets()
        self._enforced = await self._probe(codex, probe)
        return self._enforced

    async def _probe(self, codex: str, probe_bin: str) -> bool:
        inside_dir: str | None = None
        outside_dir: str | None = None
        try:
            inside_dir = tempfile.mkdtemp(
                prefix=".iclaw-sandbox-probe-", dir=str(self._workspace_root)
            )
            outside_dir = tempfile.mkdtemp(
                prefix=f".{self._workspace_root.name}-sandbox-probe-",
                dir=str(self._control_root),
            )
            inside = Path(inside_dir) / "inside.txt"
            outside = Path(outside_dir) / "outside.txt"

            # Positive controls: both paths must be writable without a sandbox.
            for path in (inside, outside):
                path.write_bytes(b"")
                path.unlink()

            read_nonce = secrets.token_hex(16)
            read_command = self._probe_command(
                probe_bin,
                mode="read-only",
                nonce=read_nonce,
                inside=inside,
            )
            read_argv = build_sandbox_argv(
                codex,
                profile=PROFILE_READ_ONLY,
                command=read_command,
                cwd=str(self._workspace_root),
            )
            code, stdout, stderr = await self._spawn(
                read_argv, cwd=str(self._workspace_root), timeout_ms=PROBE_TIMEOUT_MS
            )
            read_tokens = set(stdout.decode("utf-8", "replace").splitlines())
            read_expected = {
                f"ICLAW_PROBE_{read_nonce}_RAN",
                f"ICLAW_PROBE_{read_nonce}_READ_ONLY_DENIED",
            }
            if code != 0 or read_tokens != read_expected or inside.exists():
                self._probe_diagnostic = self._diagnostic(
                    "read-only write-denial probe failed", code, stdout, stderr
                )
                return False

            workspace_nonce = secrets.token_hex(16)
            workspace_command = self._probe_command(
                probe_bin,
                mode="workspace",
                nonce=workspace_nonce,
                inside=inside,
                outside=outside,
            )
            workspace_argv = build_sandbox_argv(
                codex,
                profile=PROFILE_WORKSPACE,
                command=workspace_command,
                cwd=str(self._workspace_root),
            )
            code, stdout, stderr = await self._spawn(
                workspace_argv,
                cwd=str(self._workspace_root),
                timeout_ms=PROBE_TIMEOUT_MS,
            )
            workspace_tokens = set(stdout.decode("utf-8", "replace").splitlines())
            workspace_expected = {
                f"ICLAW_PROBE_{workspace_nonce}_RAN",
                f"ICLAW_PROBE_{workspace_nonce}_INSIDE_WROTE",
                f"ICLAW_PROBE_{workspace_nonce}_OUTSIDE_DENIED",
            }
            if (
                code != 0
                or workspace_tokens != workspace_expected
                or not inside.exists()
                or outside.exists()
            ):
                self._probe_diagnostic = self._diagnostic(
                    "workspace boundary probe failed", code, stdout, stderr
                )
                return False

            self._probe_diagnostic = None
            return True
        except (OSError, LocalSandboxError) as exc:
            self._probe_diagnostic = (
                f"sandbox behavior probe could not complete ({type(exc).__name__})"
            )
            return False
        finally:
            if inside_dir is not None:
                shutil.rmtree(inside_dir, ignore_errors=True)
            if outside_dir is not None:
                shutil.rmtree(outside_dir, ignore_errors=True)

    @staticmethod
    def _diagnostic(
        message: str,
        code: int,
        stdout: bytes,
        stderr: bytes,
    ) -> str:
        detail = (stderr.decode("utf-8", "replace") or stdout.decode("utf-8", "replace")).strip()
        return f"{message} (exit={code})" + (f": {detail[:500]}" if detail else "")

    async def run(
        self,
        command: list[str],
        *,
        relative_cwd: str | None = None,
        profile: str | None = None,
        timeout_ms: int | None = None,
    ) -> SandboxRunResult:
        """Run a command inside the instance's authorized workspace."""

        selected_profile = profile or self._default_profile
        if selected_profile not in _ALLOWED_PROFILES:
            raise ValueError(f"unsupported sandbox profile: {selected_profile!r}")
        selected_timeout = self._default_timeout_ms if timeout_ms is None else timeout_ms
        if selected_timeout <= 0:
            raise ValueError("timeout_ms must be positive")

        async with self._run_lock:
            return await self._run_locked(
                command,
                relative_cwd=relative_cwd,
                profile=selected_profile,
                timeout_ms=selected_timeout,
            )

    async def _run_locked(
        self,
        command: list[str],
        *,
        relative_cwd: str | None,
        profile: str,
        timeout_ms: int,
    ) -> SandboxRunResult:
        codex, _probe = self._require_trusted_assets()
        run_cwd = self._resolve_run_cwd(relative_cwd)
        if not await self._verify_enforcement_locked():
            raise SandboxNotEnforcedError(
                self._probe_diagnostic or "sandbox enforcement could not be verified"
            )
        # Re-check immediately before the user-command spawn.  This cannot
        # replace platform code signing, but narrows replacement after probes.
        codex, _probe = self._require_trusted_assets()

        argv = build_sandbox_argv(
            codex,
            profile=profile,
            command=command,
            cwd=run_cwd,
        )
        started = time.monotonic()
        code, stdout, stderr = await self._spawn(
            argv,
            cwd=run_cwd,
            timeout_ms=timeout_ms,
        )
        return SandboxRunResult(
            stdout=stdout.decode("utf-8", "replace"),
            stderr=stderr.decode("utf-8", "replace"),
            exitCode=code,
            durationMs=int((time.monotonic() - started) * 1000),
            sandboxProfile=profile,
            enforced=True,
        )

    async def _spawn(
        self,
        argv: list[str],
        *,
        cwd: str,
        timeout_ms: int,
    ) -> tuple[int, bytes, bytes]:
        run_tmp = Path(tempfile.mkdtemp(prefix=".iclaw-run-tmp-", dir=str(self._workspace_root)))
        launcher_env = dict(self._launcher_env)
        for key in ("TEMP", "TMP", "TMPDIR"):
            launcher_env[key] = str(run_tmp)
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=cwd,
                env=launcher_env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            if proc.stdout is None or proc.stderr is None:
                await self._terminate_process_group(proc)
                raise LocalSandboxError("codex sandbox output pipes are unavailable")

            stdout_task = asyncio.create_task(self._read_bounded(proc.stdout, "stdout"))
            stderr_task = asyncio.create_task(self._read_bounded(proc.stderr, "stderr"))
            wait_task = asyncio.create_task(proc.wait())
            try:
                stdout, stderr, returncode = await asyncio.wait_for(
                    asyncio.gather(stdout_task, stderr_task, wait_task),
                    timeout=timeout_ms / 1000,
                )
            except TimeoutError:
                await self._terminate_process_group(proc)
                await self._cancel_tasks(stdout_task, stderr_task, wait_task)
                raise LocalSandboxError(
                    f"codex sandbox execution timed out after {timeout_ms}ms"
                ) from None
            except SandboxOutputLimitError:
                await self._terminate_process_group(proc)
                await self._cancel_tasks(stdout_task, stderr_task, wait_task)
                raise
            except asyncio.CancelledError:
                await self._terminate_process_group(proc)
                await self._cancel_tasks(stdout_task, stderr_task, wait_task)
                raise
            except Exception as exc:
                await self._terminate_process_group(proc)
                await self._cancel_tasks(stdout_task, stderr_task, wait_task)
                raise LocalSandboxError("codex sandbox execution failed") from exc

            # A successful direct child may leave background descendants after
            # closing its pipes.  Persistent daemons are outside this primitive.
            await self._terminate_process_group(proc)
            return returncode, stdout or b"", stderr or b""
        finally:
            self._cleanup_run_tmp(run_tmp)

    @staticmethod
    def _cleanup_run_tmp(path: Path) -> None:
        """Remove one exact run temp path without following a replacement symlink."""

        try:
            path_stat = path.lstat()
        except FileNotFoundError:
            return
        try:
            if stat.S_ISDIR(path_stat.st_mode) and not stat.S_ISLNK(path_stat.st_mode):
                shutil.rmtree(path)
            else:
                path.unlink()
        except OSError:
            # Never broaden cleanup beyond this random writable child.  The
            # next spawn receives a new path even if this one was tampered with.
            return

    @staticmethod
    async def _read_bounded(
        stream: asyncio.StreamReader,
        label: str,
    ) -> bytes:
        captured = bytearray()
        while chunk := await stream.read(64 * 1024):
            if len(captured) + len(chunk) > MAX_CAPTURE_BYTES:
                raise SandboxOutputLimitError(
                    f"codex sandbox {label} exceeded {MAX_CAPTURE_BYTES} bytes"
                )
            captured.extend(chunk)
        return bytes(captured)

    @staticmethod
    async def _cancel_tasks(*tasks: asyncio.Task[Any]) -> None:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    async def _terminate_process_group(proc: asyncio.subprocess.Process) -> None:
        """Best-effort termination of the dedicated POSIX process group."""

        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            if proc.returncode is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
        await proc.wait()
