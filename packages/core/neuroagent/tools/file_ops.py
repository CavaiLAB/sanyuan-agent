"""File-operation tool builders — wraps ISandboxProvider file methods."""

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from neuroagent.sandbox.abc import ISandboxProvider, SandboxSession


@dataclass(frozen=True, slots=True)
class ReadFileResult:
    content: str


@dataclass(frozen=True, slots=True)
class WriteFileResult:
    ok: bool = True


@dataclass(frozen=True, slots=True)
class ListDirResult:
    entries: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class UploadFileResult:
    ok: bool = True


def build_file_ops_tools(
    provider: ISandboxProvider,
    session: SandboxSession,
) -> dict[str, Callable[..., Awaitable[Any]]]:
    """Return a dict of bound file-operation tools for the given session."""

    async def read_file(path: str) -> ReadFileResult:
        content = await provider.read_file(session.id, path)
        return ReadFileResult(content=content)

    async def write_file(path: str, content: str) -> WriteFileResult:
        await provider.write_file(session.id, path, content)
        return WriteFileResult()

    async def list_dir(path: str) -> ListDirResult:
        entries = await provider.list_dir(session.id, path)
        return ListDirResult(entries=entries)

    async def upload_file(local_path: str, remote_path: str) -> UploadFileResult:
        await provider.upload_file(session.id, local_path, remote_path)
        return UploadFileResult()

    return {
        "read_file": read_file,
        "write_file": write_file,
        "list_dir": list_dir,
        "upload_file": upload_file,
    }
