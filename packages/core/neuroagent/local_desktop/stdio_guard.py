"""Process-level stdio and environment guard for ``desktop-run``.

This module intentionally imports only the Python standard library.  The CLI
enters here before importing Pydantic, LangGraph, LiteLLM, or provider SDKs so
third-party import/runtime output can never share the stdout SSE descriptor.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Callable, Mapping
from typing import BinaryIO, Protocol, TextIO

_PASSTHROUGH_ENV = frozenset(
    {
        "ICLAW_LOCAL_LLM_MODEL",
        "ICLAW_LOCAL_LLM_API_BASE",
        "ICLAW_LOCAL_LLM_API_KEY",
        "PATH",
        "PATHEXT",
        "SystemRoot",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "TEMP",
        "TMP",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
    }
)
_CONTROLLED_ENV = {
    "SSL_VERIFY": "True",
    "AIOHTTP_TRUST_ENV": "False",
    "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    "PYTHONUNBUFFERED": "1",
    "PYTHONUTF8": "1",
}


class DesktopRunner(Protocol):
    async def __call__(
        self,
        stdin: BinaryIO,
        stdout: TextIO,
        *,
        env: Mapping[str, str],
    ) -> int: ...


def _sanitize_process_environment(source: Mapping[str, str]) -> None:
    sanitized = {key: value for key, value in source.items() if key in _PASSTHROUGH_ENV}
    sanitized.update(_CONTROLLED_ENV)
    os.environ.clear()
    os.environ.update(sanitized)


def _guard_error(stdout: TextIO) -> None:
    payload = {
        "code": "LOCAL_RUNTIME_CRASHED",
        "message": "local desktop runtime crashed",
    }
    stdout.write(f"event: error\ndata: {json.dumps(payload)}\n\n")
    stdout.flush()


def run_guarded_desktop_process(
    load_runner: Callable[[], DesktopRunner] | None = None,
) -> int:
    """Run one desktop request with stdout/stderr and ambient env quarantined.

    ``load_runner`` is a subprocess-test seam.  Production leaves it unset so
    the model stack is imported only after fd 1/2 have been redirected.
    """

    sse_fd = os.dup(1)
    sink_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(sink_fd, 1)
        os.dup2(sink_fd, 2)
    finally:
        os.close(sink_fd)

    _sanitize_process_environment(dict(os.environ))
    with os.fdopen(
        sse_fd,
        "w",
        encoding="utf-8",
        buffering=1,
        newline="\n",
    ) as sse_stdout:
        try:
            if load_runner is None:
                from .desktop_run import run_desktop_stdio

                runner: DesktopRunner = run_desktop_stdio
            else:
                runner = load_runner()
            return asyncio.run(runner(sys.stdin.buffer, sse_stdout, env=os.environ))
        except Exception:
            _guard_error(sse_stdout)
            return 1


__all__ = ["run_guarded_desktop_process"]
