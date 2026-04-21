"""Sandbox provider abstract interface.

Pure typing + dataclass — AOT-friendly, no reflection.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class SandboxSession:
    """A running sandbox session handle."""

    id: str
    created_at: datetime
    workdir: str


@dataclass(frozen=True, slots=True)
class SandboxResult:
    """Output of a command executed inside a sandbox."""

    success: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0


class ISandboxProvider(ABC):
    """Abstract sandbox provider.

    Implementations: LocalSandboxProvider, DockerSandboxProvider, etc.
    """

    @abstractmethod
    async def create_session(self) -> SandboxSession:
        """Spin up a new sandbox session and return its handle."""

    @abstractmethod
    async def run_shell(self, session_id: str, cmd: str) -> SandboxResult:
        """Execute a shell command inside the given session."""

    @abstractmethod
    async def read_file(self, session_id: str, path: str) -> str:
        """Read a text file from the session filesystem."""

    @abstractmethod
    async def write_file(self, session_id: str, path: str, content: str) -> None:
        """Write a text file into the session filesystem."""

    @abstractmethod
    async def list_dir(self, session_id: str, path: str) -> list[str]:
        """List entries in a session directory."""

    @abstractmethod
    async def upload_file(
        self, session_id: str, local_path: str, remote_path: str
    ) -> None:
        """Upload a file from the host into the session."""

    @abstractmethod
    async def close_session(self, session_id: str) -> None:
        """Tear down the sandbox session and release resources."""
