"""File-operation tool builders — wraps ISandboxProvider file methods."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from neuroagent.sandbox.abc import ISandboxProvider, SandboxSession
from neuroagent.tools.provider import ToolSpec, spec_from_callable
from neuroagent.tools.registry import ToolRegistry, callable_to_schema


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
    registry: ToolRegistry | None = None,
) -> dict[str, Callable[..., Awaitable[Any]]]:
    """Return a dict of bound file-operation tools for the given session.

    If *registry* is provided, each tool is registered under its function
    name with its OpenAI-style function schema.
    """

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

    tools: dict[str, Callable[..., Awaitable[Any]]] = {
        "read_file": read_file,
        "write_file": write_file,
        "list_dir": list_dir,
        "upload_file": upload_file,
    }

    if registry is not None:
        registry.register("read_file", read_file, callable_to_schema(read_file))
        registry.register("write_file", write_file, callable_to_schema(write_file))
        registry.register("list_dir", list_dir, callable_to_schema(list_dir))
        registry.register(
            "upload_file", upload_file, callable_to_schema(upload_file)
        )

    return tools


# Descriptive provider metadata per file op. Reads (read_file / list_dir) stay
# read with no hint; mutations (write_file / upload_file) are write + confirm.
_FILE_OPS_META: dict[str, dict[str, Any]] = {
    "read_file": {"capabilities": ("file", "read"), "side_effect": "read"},
    "list_dir": {"capabilities": ("file", "list"), "side_effect": "read"},
    "write_file": {
        "capabilities": ("file", "write"),
        "side_effect": "write",
        "policy_hint": "confirm",
    },
    "upload_file": {
        "capabilities": ("file", "upload"),
        "side_effect": "write",
        "policy_hint": "confirm",
    },
}


class FileOpsToolProvider:
    """Thin ToolProvider wrapper over :func:`build_file_ops_tools`.

    Yields the existing file-op tools (read_file / write_file / list_dir /
    upload_file) as specs with ``category="filesystem"`` plus per-tool metadata.
    It changes no behavior and no schema: each callable and its reflected schema
    come straight from the builder; per PR1 the metadata never gates execution.

    Selectable through the provider path: reference the provider in
    ``AgentSpec.providers`` and enable individual file-op names via
    ``AgentSpec.tools`` / ``tool_sets``.
    """

    def __init__(self, provider: ISandboxProvider, session: SandboxSession) -> None:
        self._provider = provider
        self._session = session

    def provide(self) -> list[ToolSpec]:
        scratch = ToolRegistry()
        build_file_ops_tools(self._provider, self._session, scratch)
        specs: list[ToolSpec] = []
        for name in scratch.list_tools():
            meta = _FILE_OPS_META.get(name, {})
            specs.append(
                spec_from_callable(
                    scratch.get(name),
                    name=name,
                    schema=scratch.get_schema(name),
                    category="filesystem",
                    capabilities=meta.get("capabilities", ("file",)),
                    side_effect=meta.get("side_effect", "read"),
                    policy_hint=meta.get("policy_hint"),
                )
            )
        return specs
