"""Transports carrying JSON-RPC 2.0 messages to and from an MCP server."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable


class McpTransportError(Exception):
    """The byte stream to the MCP server failed (spawn / EOF / bad framing)."""


@runtime_checkable
class McpTransport(Protocol):
    """A bidirectional, message-oriented JSON-RPC pipe."""

    async def send(self, message: dict[str, Any]) -> None:
        """Deliver one JSON-RPC message to the server."""
        ...

    async def receive(self) -> dict[str, Any]:
        """Block until the server's next JSON-RPC message arrives."""
        ...

    async def close(self) -> None:
        """Tear down the connection."""
        ...


class StdioTransport:
    """Spawn an MCP server subprocess; newline-delimited JSON-RPC over stdio."""

    def __init__(
        self,
        command: str,
        args: Sequence[str] = (),
        *,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.command = command
        self.args = tuple(args)
        # None inherits the parent environment (asyncio subprocess default).
        self.env = dict(env) if env is not None else None
        self._proc: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        """Spawn the server process. Idempotent."""
        if self._proc is not None:
            return
        try:
            self._proc = await asyncio.create_subprocess_exec(
                self.command,
                *self.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                env=self.env,
            )
        except OSError as exc:
            raise McpTransportError(
                f"failed to spawn MCP server {self.command!r}: {exc}"
            ) from exc

    async def send(self, message: dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise McpTransportError("transport not started")
        proc.stdin.write(json.dumps(message, ensure_ascii=False).encode() + b"\n")
        await proc.stdin.drain()

    async def receive(self) -> dict[str, Any]:
        proc = self._proc
        if proc is None or proc.stdout is None:
            raise McpTransportError("transport not started")
        while True:
            line = await proc.stdout.readline()
            if not line:
                raise McpTransportError("MCP server closed the connection")
            text = line.decode().strip()
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise McpTransportError(
                    f"invalid JSON from MCP server: {text[:200]}"
                ) from exc
            if not isinstance(parsed, dict):
                raise McpTransportError(f"expected a JSON object, got: {text[:200]}")
            return parsed

    async def close(self) -> None:
        proc = self._proc
        if proc is None:
            return
        self._proc = None
        if proc.stdin is not None:
            proc.stdin.close()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except TimeoutError:
            proc.kill()
            await proc.wait()
