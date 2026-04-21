"""Bash tool builder — wraps ISandboxProvider.run_shell for agent use."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from neuroagent.sandbox.abc import ISandboxProvider, SandboxSession
from neuroagent.tools.registry import ToolRegistry, callable_to_schema


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
