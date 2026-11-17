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
    codex_filename = "iclaw-codex-sandbox.exe" if platform == "win32" else "iclaw-codex-sandbox"
    codex_binary = tmp_path / codex_filename
    codex_binary.write_bytes(b"native-codex")
    codex_notice = tmp_path / "NOTICE.txt"
    codex_notice.write_bytes(b"Codex Apache-2.0 notice\n")
    codex_license = tmp_path / "LICENSE.txt"
    codex_license.write_bytes(b"Apache License 2.0\n")
    probe = tmp_path / "iclaw-desktop-probe" if platform == "darwin" else None
    if probe is not None:
        probe.write_bytes(b"native-probe")
    destination = tmp_path / "app" / "harness-manifest.json"

    module.write_manifest(
        binary,
        codex_binary,
        codex_notice,
        codex_license,
        probe,
        "0.136.0",
        destination,
        platform,
        arch,
    )

    payload = json.loads(destination.read_text(encoding="utf-8"))
    expected = {
        "schemaVersion": 2,
        "protocol": "iclaw.desktop-run.v1",
        "platform": platform,
        "arch": arch,
        "filename": filename,
        "sha256": hashlib.sha256(b"native-binary").hexdigest(),
        "size": len(b"native-binary"),
        "codexSandbox": {
            "package": "@openai/codex",
            "version": "0.136.0",
            "filename": codex_filename,
            "sha256": hashlib.sha256(b"native-codex").hexdigest(),
            "size": len(b"native-codex"),
            "noticeFilename": "NOTICE.txt",
            "noticeSha256": hashlib.sha256(b"Codex Apache-2.0 notice\n").hexdigest(),
            "licenseFilename": "LICENSE.txt",
            "licenseSha256": hashlib.sha256(b"Apache License 2.0\n").hexdigest(),
        },
    }
    if platform == "darwin":
        expected["sandboxProbe"] = {
            "filename": "iclaw-desktop-probe",
            "sha256": hashlib.sha256(b"native-probe").hexdigest(),
            "size": len(b"native-probe"),
        }
    assert payload == expected
    assert destination.read_bytes().endswith(b"\n")


def test_manifest_writer_rejects_wrong_names_and_unsupported_targets(tmp_path: Path) -> None:
    module = _module()
    binary = tmp_path / "attacker"
    binary.write_bytes(b"x")
    codex_binary = tmp_path / "iclaw-codex-sandbox"
    codex_binary.write_bytes(b"codex")
    notice = tmp_path / "NOTICE.txt"
    notice.write_text("notice", encoding="utf-8")
    license_file = tmp_path / "LICENSE.txt"
    license_file.write_text("license", encoding="utf-8")

    with pytest.raises(ValueError, match="fixed filename"):
        module.write_manifest(
            binary,
            codex_binary,
            notice,
            license_file,
            None,
            "0.136.0",
            tmp_path / "manifest.json",
            "darwin",
            "arm64",
        )

    windows_binary = tmp_path / "iclaw-desktop-harness.exe"
    windows_binary.write_bytes(b"x")
    windows_codex = tmp_path / "iclaw-codex-sandbox.exe"
    windows_codex.write_bytes(b"codex")
    with pytest.raises(ValueError, match="unsupported"):
        module.write_manifest(
            windows_binary,
            windows_codex,
            notice,
            license_file,
            None,
            "0.136.0",
            tmp_path / "manifest.json",
            "win32",
            "arm64",
        )


def test_manifest_writer_rejects_codex_resources_or_version_drift(tmp_path: Path) -> None:
    module = _module()
    harness = tmp_path / "iclaw-desktop-harness"
    harness.write_bytes(b"harness")
    wrong_codex = tmp_path / "codex"
    wrong_codex.write_bytes(b"codex")
    notice = tmp_path / "NOTICE.txt"
    notice.write_text("notice", encoding="utf-8")
    license_file = tmp_path / "LICENSE.txt"
    license_file.write_text("license", encoding="utf-8")
    probe = tmp_path / "iclaw-desktop-probe"
    probe.write_bytes(b"probe")

    with pytest.raises(ValueError, match="Codex sandbox binary"):
        module.write_manifest(
            harness,
            wrong_codex,
            notice,
            license_file,
            probe,
            "0.136.0",
            tmp_path / "manifest.json",
            "darwin",
            "arm64",
        )

    codex = tmp_path / "iclaw-codex-sandbox"
    codex.write_bytes(b"codex")
    wrong_notice = tmp_path / "THIRD-PARTY.txt"
    wrong_notice.write_text("notice", encoding="utf-8")
    with pytest.raises(ValueError, match="Codex notice"):
        module.write_manifest(
            harness,
            codex,
            wrong_notice,
            license_file,
            probe,
            "0.136.0",
            tmp_path / "manifest.json",
            "darwin",
            "arm64",
        )

    wrong_license = tmp_path / "COPYING.txt"
    wrong_license.write_text("license", encoding="utf-8")
    with pytest.raises(ValueError, match="Codex license"):
        module.write_manifest(
            harness,
            codex,
            notice,
            wrong_license,
            probe,
            "0.136.0",
            tmp_path / "manifest.json",
            "darwin",
            "arm64",
        )

    with pytest.raises(ValueError, match="Codex version"):
        module.write_manifest(
            harness,
            codex,
            notice,
            license_file,
            probe,
            "0.137.0",
            tmp_path / "manifest.json",
            "darwin",
            "arm64",
        )

    wrong_probe = tmp_path / "probe"
    wrong_probe.write_bytes(b"probe")
    with pytest.raises(ValueError, match="desktop sandbox probe"):
        module.write_manifest(
            harness,
            codex,
            notice,
            license_file,
            wrong_probe,
            "0.136.0",
            tmp_path / "manifest.json",
            "darwin",
            "arm64",
        )
