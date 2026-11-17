"""Subprocess proofs for desktop fd and ambient-environment quarantine."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_HARNESS_ROOT = Path(__file__).resolve().parents[1]


def _run(script: str) -> subprocess.CompletedProcess[bytes]:
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(_HARNESS_ROOT),
            "ICLAW_LOCAL_LLM_MODEL": "openai/local-test",
            "ICLAW_LOCAL_LLM_API_BASE": "https://llm.example.test/v1",
            "ICLAW_LOCAL_LLM_API_KEY": "allowed-key",
            "ICLAW_LOCAL_CODEX_BIN": "/trusted/codex",
            "ICLAW_LOCAL_CODEX_SHA256": "a" * 64,
            "ICLAW_LOCAL_CODEX_VERSION": "0.136.0",
            "ICLAW_LOCAL_PROBE_BIN": "/trusted/harness",
            "ICLAW_LOCAL_PROBE_SHA256": "b" * 64,
            "AWS_ACCESS_KEY_ID": "forbidden-aws-key",
            "GOOGLE_APPLICATION_CREDENTIALS": "/forbidden/gcp.json",
            "HTTPS_PROXY": "http://forbidden-proxy.invalid",
            "SSL_VERIFY": "False",
            "AIOHTTP_TRUST_ENV": "True",
        }
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        input=b"",
        capture_output=True,
        check=False,
        env=env,
        cwd=_HARNESS_ROOT,
    )


def test_guard_quarantines_dependency_output_and_ambient_credentials() -> None:
    result = _run(
        r"""
import json
import os
import sys
from sanyuan_agent.local_desktop.stdio_guard import run_guarded_desktop_process

async def runner(_stdin, stdout, *, env):
    print("RAW_STDOUT sk-live-VERY-SECRET")
    sys.stderr.write("RAW_STDERR sk-live-VERY-SECRET\n")
    forbidden = any(
        key in env
        for key in (
            "AWS_ACCESS_KEY_ID",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "HTTPS_PROXY",
            "PYTHONPATH",
        )
    )
    controlled = (
        env.get("SSL_VERIFY") == "True"
        and env.get("AIOHTTP_TRUST_ENV") == "False"
        and env.get("LITELLM_LOCAL_MODEL_COST_MAP") == "True"
        and env.get("ICLAW_LOCAL_LLM_API_KEY") == "allowed-key"
        and env.get("ICLAW_LOCAL_CODEX_BIN") == "/trusted/codex"
        and env.get("ICLAW_LOCAL_CODEX_SHA256") == "a" * 64
        and env.get("ICLAW_LOCAL_CODEX_VERSION") == "0.136.0"
        and env.get("ICLAW_LOCAL_PROBE_BIN") == "/trusted/harness"
        and env.get("ICLAW_LOCAL_PROBE_SHA256") == "b" * 64
    )
    stdout.write(
        "event: final_answer\ndata: "
        + json.dumps({"content": "clean" if controlled and not forbidden else "dirty"})
        + "\n\n"
    )
    return 0

raise SystemExit(run_guarded_desktop_process(lambda: runner))
"""
    )

    assert result.returncode == 0
    assert result.stderr == b""
    assert b"event: final_answer\n" in result.stdout
    assert b'"content": "clean"' in result.stdout
    assert b"VERY-SECRET" not in result.stdout
    assert b"dirty" not in result.stdout


def test_guard_sanitizes_unexpected_loader_failure() -> None:
    result = _run(
        r"""
from sanyuan_agent.local_desktop.stdio_guard import run_guarded_desktop_process

def fail():
    print("INIT_STDOUT sk-live-VERY-SECRET")
    raise RuntimeError("loader sk-live-VERY-SECRET")

raise SystemExit(run_guarded_desktop_process(fail))
"""
    )

    assert result.returncode == 1
    assert result.stderr == b""
    assert b"LOCAL_RUNTIME_CRASHED" in result.stdout
    assert b"VERY-SECRET" not in result.stdout
