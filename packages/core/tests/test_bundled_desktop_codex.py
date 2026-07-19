"""Black-box checks for the native Codex sandbox resource in desktop packages."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from neuroagent.local_desktop import CodexSandbox

_BUNDLE_ENV = "ICLAW_BUNDLED_CODEX_TEST_BINARY"
_PROBE_BUNDLE_ENV = "ICLAW_BUNDLED_PROBE_TEST_BINARY"
_MANIFEST_ENV = "ICLAW_BUNDLED_MANIFEST_TEST_PATH"
_MANIFEST = Path(__file__).resolve().parents[5] / "apps" / "desktop" / "harness-manifest.json"


def _bundle() -> tuple[Path, dict[str, object]]:
    value = os.environ.get(_BUNDLE_ENV)
    if not value:
        pytest.skip(f"{_BUNDLE_ENV} is set only by the native packaging job")
    binary = Path(value).resolve()
    assert binary.is_file()
    manifest_path = Path(os.environ.get(_MANIFEST_ENV, str(_MANIFEST))).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["codexSandbox"]
    return binary, manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_bundled_codex_is_the_pinned_native_executable() -> None:
    binary, manifest = _bundle()
    assert binary.name == manifest["filename"]
    assert binary.stat().st_size == manifest["size"]
    assert _sha256(binary) == manifest["sha256"]
    with binary.open("rb") as stream:
        header = stream.read(4)
    assert header.startswith(b"MZ") if sys.platform == "win32" else header in {
        b"\xfe\xed\xfa\xcf",
        b"\xcf\xfa\xed\xfe",
        b"\xca\xfe\xba\xbe",
        b"\xbe\xba\xfe\xca",
    }

    result = subprocess.run(
        [str(binary), "--version"],
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0
    assert result.stdout.decode().strip() == f"codex-cli {manifest['version']}"


@pytest.mark.skipif(sys.platform != "darwin", reason="current provider enables POSIX macOS only")
def test_bundled_codex_read_only_profile_denies_workspace_writes(tmp_path: Path) -> None:
    binary, _manifest = _bundle()
    probe = Path(os.environ[_PROBE_BUNDLE_ENV]).resolve()
    assert probe.is_file()
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    home.mkdir(mode=0o700)
    workspace.mkdir(mode=0o700)
    denied = workspace / "must-not-exist"
    nonce = "a" * 32
    env = {
        "HOME": str(home),
        "CODEX_HOME": str(home),
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
    }
    result = subprocess.run(
        [
            str(binary),
            "sandbox",
            "--permissions-profile",
            ":read-only",
            "--cd",
            str(workspace),
            "--",
            str(probe),
            "sandbox-probe",
            "read-only",
            nonce,
            str(denied),
        ],
        capture_output=True,
        check=False,
        env=env,
        timeout=30,
    )
    assert result.returncode == 0
    assert result.stdout.decode().splitlines() == [
        f"ICLAW_PROBE_{nonce}_RAN",
        f"ICLAW_PROBE_{nonce}_READ_ONLY_DENIED",
    ]
    assert not denied.exists()


@pytest.mark.skipif(sys.platform != "darwin", reason="current provider enables POSIX macOS only")
def test_bundled_codex_and_probe_satisfy_provider_enforcement() -> None:
    codex, codex_manifest = _bundle()
    probe = Path(os.environ[_PROBE_BUNDLE_ENV]).resolve()
    manifest_path = Path(os.environ.get(_MANIFEST_ENV, str(_MANIFEST))).resolve()
    full_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    probe_manifest = full_manifest["sandboxProbe"]
    assert probe.name == probe_manifest["filename"]
    assert probe.stat().st_size == probe_manifest["size"]
    assert _sha256(probe) == probe_manifest["sha256"]

    control = Path(tempfile.mkdtemp(prefix=".iclaw-native-probe-", dir=Path.home()))
    workspace = control / "workspace"
    workspace.mkdir(mode=0o700)
    try:
        sandbox = CodexSandbox(
            codex_bin=str(codex),
            codex_sha256=str(codex_manifest["sha256"]),
            probe_bin=str(probe),
            probe_sha256=str(probe_manifest["sha256"]),
            probe_args_prefix=("sandbox-probe",),
            control_root=str(control),
            authorized_workspace_root=str(workspace),
        )
        assert asyncio.run(sandbox.verify_enforcement(force=True)) is True
    finally:
        shutil.rmtree(control)
