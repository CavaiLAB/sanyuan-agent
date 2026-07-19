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
_CODEX_BASENAME = "iclaw-codex-sandbox"
_CODEX_NOTICE = "NOTICE.txt"
_CODEX_LICENSE = "LICENSE.txt"
_CODEX_VERSION = "0.136.0"
_PROBE_BASENAME = "iclaw-desktop-probe"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(
    binary: Path,
    codex_binary: Path,
    codex_notice: Path,
    codex_license: Path,
    probe_binary: Path | None,
    codex_version: str,
    destination: Path,
    platform: str,
    arch: str,
) -> None:
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

    expected_codex = f"{_CODEX_BASENAME}.exe" if platform == "win32" else _CODEX_BASENAME
    if (
        codex_binary.name != expected_codex
        or not codex_binary.is_file()
        or codex_binary.is_symlink()
    ):
        raise ValueError("Codex sandbox binary does not match the fixed filename")
    codex_stat = codex_binary.stat()
    if codex_stat.st_size <= 0:
        raise ValueError("Codex sandbox binary is empty")
    if (
        codex_notice.name != _CODEX_NOTICE
        or not codex_notice.is_file()
        or codex_notice.is_symlink()
        or codex_notice.stat().st_size <= 0
    ):
        raise ValueError("Codex notice does not match the fixed resource")
    if (
        codex_license.name != _CODEX_LICENSE
        or not codex_license.is_file()
        or codex_license.is_symlink()
        or codex_license.stat().st_size <= 0
    ):
        raise ValueError("Codex license does not match the fixed resource")
    if codex_version != _CODEX_VERSION:
        raise ValueError("Codex version does not match the pinned package")

    probe_payload: dict[str, object] | None = None
    if platform == "darwin":
        expected_probe = _PROBE_BASENAME
        if (
            probe_binary is None
            or probe_binary.name != expected_probe
            or not probe_binary.is_file()
            or probe_binary.is_symlink()
        ):
            raise ValueError("desktop sandbox probe does not match the fixed filename")
        probe_stat = probe_binary.stat()
        if probe_stat.st_size <= 0:
            raise ValueError("desktop sandbox probe is empty")
        probe_payload = {
            "filename": expected_probe,
            "sha256": _sha256(probe_binary),
            "size": probe_stat.st_size,
        }
    elif probe_binary is not None:
        raise ValueError("desktop sandbox probe is not enabled for this target")

    payload = {
        "schemaVersion": 2,
        "protocol": _PROTOCOL,
        "platform": platform,
        "arch": arch,
        "filename": expected,
        "sha256": _sha256(binary),
        "size": stat.st_size,
        "codexSandbox": {
            "package": "@openai/codex",
            "version": codex_version,
            "filename": expected_codex,
            "sha256": _sha256(codex_binary),
            "size": codex_stat.st_size,
            "noticeFilename": _CODEX_NOTICE,
            "noticeSha256": _sha256(codex_notice),
            "licenseFilename": _CODEX_LICENSE,
            "licenseSha256": _sha256(codex_license),
        },
    }
    if probe_payload is not None:
        payload["sandboxProbe"] = probe_payload
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
    parser.add_argument("--codex-binary", required=True, type=Path)
    parser.add_argument("--codex-notice", required=True, type=Path)
    parser.add_argument("--codex-license", required=True, type=Path)
    parser.add_argument("--probe-binary", type=Path)
    parser.add_argument("--codex-version", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--platform", required=True, choices=("darwin", "win32"))
    parser.add_argument("--arch", required=True, choices=("arm64", "x64"))
    args = parser.parse_args()
    write_manifest(
        args.binary.resolve(),
        args.codex_binary.resolve(),
        args.codex_notice.resolve(),
        args.codex_license.resolve(),
        args.probe_binary.resolve() if args.probe_binary is not None else None,
        args.codex_version,
        args.manifest.resolve(),
        args.platform,
        args.arch,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
