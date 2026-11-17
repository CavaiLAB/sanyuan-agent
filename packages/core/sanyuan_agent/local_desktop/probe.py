"""Hidden behavior-probe entrypoint for the local Codex sandbox."""

from __future__ import annotations

import re
from pathlib import Path

_NONCE_RE = re.compile(r"^[0-9a-f]{32}$")


def _attempt_write(path: Path, *, token: str) -> None:
    try:
        path.write_bytes(b"")
    except PermissionError:
        print(f"{token}_DENIED")
    except OSError as exc:
        print(f"{token}_ERR_{getattr(exc, 'errno', 'UNKNOWN')}")
    else:
        print(f"{token}_WROTE")


def run_probe(
    mode: str,
    nonce: str,
    inside: Path,
    outside: Path | None = None,
) -> int:
    """Emit nonce-bound outcomes consumed by :class:`CodexSandbox`."""

    if not _NONCE_RE.fullmatch(nonce):
        return 2
    print(f"ICLAW_PROBE_{nonce}_RAN")
    if mode == "read-only" and outside is None:
        _attempt_write(inside, token=f"ICLAW_PROBE_{nonce}_READ_ONLY")
        return 0
    if mode == "workspace" and outside is not None:
        _attempt_write(inside, token=f"ICLAW_PROBE_{nonce}_INSIDE")
        _attempt_write(outside, token=f"ICLAW_PROBE_{nonce}_OUTSIDE")
        return 0
    return 2
