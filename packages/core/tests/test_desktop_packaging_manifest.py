"""Deterministic manifest writer tests for the packaged desktop harness."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "packaging" / "write_desktop_manifest.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("write_desktop_manifest", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("platform", "arch", "filename"),
    [
        ("darwin", "arm64", "iclaw-desktop-harness"),
        ("darwin", "x64", "iclaw-desktop-harness"),
        ("win32", "x64", "iclaw-desktop-harness.exe"),
    ],
)
def test_manifest_matches_the_fixed_runtime_contract(
    tmp_path: Path,
    platform: str,
    arch: str,
    filename: str,
) -> None:
    module = _module()
    binary = tmp_path / filename
    binary.write_bytes(b"native-binary")
    destination = tmp_path / "app" / "harness-manifest.json"

    module.write_manifest(binary, destination, platform, arch)

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload == {
        "schemaVersion": 1,
        "protocol": "iclaw.desktop-run.v1",
        "platform": platform,
        "arch": arch,
        "filename": filename,
        "sha256": hashlib.sha256(b"native-binary").hexdigest(),
        "size": len(b"native-binary"),
    }
    assert destination.read_bytes().endswith(b"\n")


def test_manifest_writer_rejects_wrong_names_and_unsupported_targets(tmp_path: Path) -> None:
    module = _module()
    binary = tmp_path / "attacker"
    binary.write_bytes(b"x")

    with pytest.raises(ValueError, match="fixed filename"):
        module.write_manifest(binary, tmp_path / "manifest.json", "darwin", "arm64")

    windows_binary = tmp_path / "iclaw-desktop-harness.exe"
    windows_binary.write_bytes(b"x")
    with pytest.raises(ValueError, match="unsupported"):
        module.write_manifest(windows_binary, tmp_path / "manifest.json", "win32", "arm64")
