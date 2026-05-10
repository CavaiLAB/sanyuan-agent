"""LangGraph tool adapters for sandbox execution.

Exposes ``sandbox_exec_shell`` and ``sandbox_exec_python`` as LangGraph
``@tool``-decorated callables.  The tools automatically create a sandbox
on first use (per conversation) and reuse it for subsequent calls.

Usage::

    from neuroagent.tools.sandbox_tool import build_sandbox_tools

    tools = build_sandbox_tools(client=my_sandbox_client)
    # tools is a list of LangGraph BaseTool instances
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Protocol, runtime_checkable

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Context vars — set by the gateway before each agent invocation
# ---------------------------------------------------------------------------

_current_tenant_id: ContextVar[str] = ContextVar("sandbox_tenant_id", default="")
_current_user_id: ContextVar[str] = ContextVar("sandbox_user_id", default="")


def set_sandbox_context(tenant_id: str, user_id: str) -> None:
    """Set the request-scoped identity for sandbox operations."""
    _current_tenant_id.set(tenant_id)
    _current_user_id.set(user_id)


# ---------------------------------------------------------------------------
# Protocol — avoids importing SandboxClient from the app package
# ---------------------------------------------------------------------------


@runtime_checkable
class SandboxClientProto(Protocol):
    """Minimal interface the tool adapters require."""

    async def create_sandbox(
        self, template_id: str, tenant_id: str, user_id: str
    ) -> Any: ...

    async def exec_shell(
        self, sandbox_id: str, cmd: str, timeout: int = 30
    ) -> Any: ...

    async def exec_python(
        self, sandbox_id: str, code: str, timeout: int = 30
    ) -> Any: ...


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_sandbox_tools(
    client: SandboxClientProto,
    *,
    template_id: str = "neuroagent-default",
) -> list[Any]:
    """Return ``[sandbox_exec_shell, sandbox_exec_python]`` bound to *client*.

    The returned tools are ``BaseTool`` instances usable directly in
    LangGraph agent graphs.
    """
    _sandbox_id: str | None = None

    async def _ensure_sandbox() -> str:
        nonlocal _sandbox_id
        if _sandbox_id is not None:
            return _sandbox_id
        tenant_id = _current_tenant_id.get()
        user_id = _current_user_id.get()
        sb = await client.create_sandbox(
            template_id=template_id,
            tenant_id=tenant_id,
            user_id=user_id,
        )
        _sandbox_id = sb.id
        return _sandbox_id

    @tool
    async def sandbox_exec_shell(cmd: str, timeout: int = 30) -> str:
        """Execute a shell command inside a sandbox.

        A sandbox is automatically created on first call per conversation.
        Returns the combined stdout output.
        """
        sid = await _ensure_sandbox()
        result = await client.exec_shell(sid, cmd, timeout=timeout)
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr] {result.stderr}"
        if result.exit_code != 0:
            output += f"\n[exit_code={result.exit_code}]"
        return output

    @tool
    async def sandbox_exec_python(code: str, timeout: int = 30) -> str:
        """Execute Python code inside a sandbox.

        A sandbox is automatically created on first call per conversation.
        Returns the stdout output.
        """
        sid = await _ensure_sandbox()
        result = await client.exec_python(sid, code, timeout=timeout)
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr] {result.stderr}"
        if result.exit_code != 0:
            output += f"\n[exit_code={result.exit_code}]"
        return output

    return [sandbox_exec_shell, sandbox_exec_python]
