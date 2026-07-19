"""Write the deterministic integrity manifest consumed by Electron."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

_PROTOCOL = "iclaw.desktop-run.v1"
_BASENAME = "iclaw-desktop-harness"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(binary: Path, destination: Path, platform: str, arch: str) -> None:
    expected = f"{_BASENAME}.exe" if platform == "win32" else _BASENAME
    if platform not in {"darwin", "win32"} or arch not in {"arm64", "x64"}:
        raise ValueError("unsupported desktop harness target")
    if (platform, arch) == ("win32", "arm64"):
        raise ValueError("unsupported desktop harness target")
    if binary.name != expected or not binary.is_file() or binary.is_symlink():
        raise ValueError("desktop harness binary does not match the fixed filename")
    stat = binary.stat()
    if stat.st_size <= 0:
        raise ValueError("desktop harness binary is empty")

    payload = {
        "schemaVersion": 1,
        "protocol": _PROTOCOL,
        "platform": platform,
        "arch": arch,
        "filename": expected,
        "sha256": _sha256(binary),
        "size": stat.st_size,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=True, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--platform", required=True, choices=("darwin", "win32"))
    parser.add_argument("--arch", required=True, choices=("arm64", "x64"))
    args = parser.parse_args()
    write_manifest(args.binary.resolve(), args.manifest.resolve(), args.platform, args.arch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
