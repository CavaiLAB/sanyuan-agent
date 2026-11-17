"""Local (dev-only) sandbox provider.

Runs commands in a temporary directory on the host filesystem.
NOT suitable for production — zero isolation boundaries.
"""

import asyncio
import shlex
import shutil
import tempfile
import uuid
from pathlib import Path

from sanyuan_agent.sandbox.abc import ISandboxProvider, SandboxResult, SandboxSession


class LocalSandboxProvider(ISandboxProvider):
    """Dev-only sandbox using local temp directories and subprocess."""

    def __init__(self) -> None:
        self._sessions: dict[str, Path] = {}

    async def create_session(self) -> SandboxSession:
        workdir = Path(tempfile.mkdtemp(prefix="sandbox-"))
        sid = str(uuid.uuid4())
        self._sessions[sid] = workdir
        return SandboxSession(
            id=sid,
            created_at=__import__("datetime").datetime.now(),
            workdir=str(workdir),
        )

    async def run_shell(self, session_id: str, cmd: str) -> SandboxResult:
        workdir = self._resolve(session_id)
        args = shlex.split(cmd)
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=str(workdir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await proc.communicate()
        return SandboxResult(
            success=proc.returncode == 0,
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
            exit_code=proc.returncode or 0,
        )

    async def read_file(self, session_id: str, path: str) -> str:
        workdir = self._resolve(session_id)
        target = self._safe_path(workdir, path)
        return target.read_text(encoding="utf-8")

    async def write_file(self, session_id: str, path: str, content: str) -> None:
        workdir = self._resolve(session_id)
        target = self._safe_path(workdir, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    async def list_dir(self, session_id: str, path: str) -> list[str]:
        workdir = self._resolve(session_id)
        target = self._safe_path(workdir, path)
        return sorted([p.name for p in target.iterdir()])

    async def upload_file(
        self, session_id: str, local_path: str, remote_path: str
    ) -> None:
        workdir = self._resolve(session_id)
        dest = self._safe_path(workdir, remote_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest)

    async def close_session(self, session_id: str) -> None:
        workdir = self._sessions.pop(session_id, None)
        if workdir is not None:
            shutil.rmtree(workdir, ignore_errors=True)

    # -- helpers --

    def _resolve(self, session_id: str) -> Path:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise ValueError(f"Session {session_id!r} not found") from exc

    @staticmethod
    def _safe_path(workdir: Path, path: str) -> Path:
        """Resolve a path inside the sandbox workdir, reject traversal."""
        resolved = (workdir / path.lstrip("/")).resolve()
        resolved.relative_to(workdir.resolve())
        return resolved
