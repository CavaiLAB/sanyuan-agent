"""Bash tool builder — wraps ISandboxProvider.run_shell for agent use."""

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable

from neuroagent.sandbox.abc import ISandboxProvider, SandboxSession


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
) -> BashTool:
    """Return a bound bash tool for the given session."""

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

    return tool_bash
