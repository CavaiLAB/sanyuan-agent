"""Bash tool builder — wraps ISandboxProvider.run_shell for agent use."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from sanyuan_agent.sandbox.abc import ISandboxProvider, SandboxSession
from sanyuan_agent.tools.provider import ToolSpec, spec_from_callable
from sanyuan_agent.tools.registry import ToolRegistry, callable_to_schema


@dataclass(frozen=True, slots=True)
class BashOutput:
    """Structured result of a sandbox shell command."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0


BashTool = Callable[[str, int], Awaitable[dict[str, str | int]]]


def build_bash_tool(
    provider: ISandboxProvider,
    session: SandboxSession,
    registry: ToolRegistry | None = None,
) -> BashTool:
    """Return a bound bash tool for the given session.

    If *registry* is provided, the tool is registered under the name
    ``"bash"`` with its OpenAI-style function schema.
    """

    async def tool_bash(cmd: str, timeout: int = 60) -> dict[str, str | int]:
        result = await asyncio.wait_for(
            provider.run_shell(session.id, cmd),
            timeout=timeout,
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code,
        }

    if registry is not None:
        registry.register("bash", tool_bash, callable_to_schema(tool_bash))

    return tool_bash


class BashToolProvider:
    """Thin ToolProvider wrapper over :func:`build_bash_tool`.

    Yields the existing ``bash`` tool as a spec with descriptive metadata
    (``category="sandbox"``, ``side_effect="external"``, ``policy_hint="confirm"``).
    It changes no behavior and no schema: the callable and its reflected schema
    come straight from the builder (the registry name stays ``"bash"`` while the
    schema's function name stays ``"tool_bash"``, exactly as the legacy path); per
    PR1 the metadata never gates execution.

    Selectable through the provider path: reference the provider in
    ``AgentSpec.providers`` and enable ``"bash"`` via ``AgentSpec.tools`` /
    ``tool_sets``.
    """

    def __init__(self, provider: ISandboxProvider, session: SandboxSession) -> None:
        self._provider = provider
        self._session = session

    def provide(self) -> list[ToolSpec]:
        scratch = ToolRegistry()
        build_bash_tool(self._provider, self._session, scratch)
        return [
            spec_from_callable(
                scratch.get("bash"),
                name="bash",
                schema=scratch.get_schema("bash"),
                category="sandbox",
                capabilities=("sandbox", "execute"),
                side_effect="external",
                policy_hint="confirm",
            )
        ]
