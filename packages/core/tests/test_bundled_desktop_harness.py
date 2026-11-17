"""Black-box contract tests for the native desktop harness bundle."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

_BUNDLE_ENV = "ICLAW_BUNDLED_HARNESS_TEST_BINARY"


def _bundle() -> Path:
    value = os.environ.get(_BUNDLE_ENV)
    if not value:
        pytest.skip(f"{_BUNDLE_ENV} is set only by the native packaging job")
    binary = Path(value).resolve()
    assert binary.is_file()
    return binary


def _run_bundle(
    payload: bytes,
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    clean_env = {
        key: value
        for key, value in os.environ.items()
        if key
        in {
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
    }
    clean_env.update(env or {})
    return subprocess.run(
        [str(_bundle())],
        input=payload,
        capture_output=True,
        check=False,
        env=clean_env,
        timeout=90,
    )


def test_bundle_rejects_empty_input_without_stderr() -> None:
    result = _run_bundle(b"")
    assert result.returncode == 2
    assert result.stderr == b""
    assert result.stdout.startswith(b"event: error\n")
    assert b"EMPTY_REQUEST" in result.stdout


def test_bundle_rejects_missing_byok_without_echoing_input_or_host_env() -> None:
    secret = "bundle-secret-must-not-leak"
    result = _run_bundle(
        json.dumps({"content": secret}).encode() + b"\n",
        env={
            "AWS_ACCESS_KEY_ID": secret,
            "GOOGLE_APPLICATION_CREDENTIALS": f"/tmp/{secret}",
            "HTTPS_PROXY": f"http://{secret}.invalid",
            "PYTHONPATH": f"/tmp/{secret}",
            "SSL_VERIFY": "False",
        },
    )
    assert result.returncode == 2
    assert result.stderr == b""
    assert b"LOCAL_LLM_CONFIG_UNAVAILABLE" in result.stdout
    assert secret.encode() not in result.stdout


class _OpenAIStreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        request = json.loads(body)
        assert self.path.endswith("/chat/completions")
        assert request["stream"] is True
        chunks: list[dict[str, Any]] = [
            {
                "id": "chatcmpl-bundled",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "local-test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "bundled answer"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-bundled",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "local-test-model",
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                ],
            },
        ]
        response = b"".join(
            b"data: " + json.dumps(chunk).encode() + b"\n\n" for chunk in chunks
        ) + b"data: [DONE]\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(response)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, _format: str, *_args: object) -> None:
        return


def test_bundle_streams_a_real_openai_compatible_response() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OpenAIStreamHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = _run_bundle(
            b'{"content":"hello from the packaged harness"}\n',
            env={
                "ICLAW_LOCAL_LLM_MODEL": "openai/local-test-model",
                "ICLAW_LOCAL_LLM_API_BASE": f"http://127.0.0.1:{server.server_port}/v1",
                "ICLAW_LOCAL_LLM_API_KEY": "local-test-key",
                "AWS_ACCESS_KEY_ID": "forbidden",
                "HTTPS_PROXY": "http://forbidden.invalid",
                "SSL_VERIFY": "False",
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert result.returncode == 0, result.stdout.decode("utf-8", errors="replace")
    assert result.stderr == b""
    assert b"event: final_answer\n" in result.stdout
    assert b"bundled answer" in result.stdout
    assert b"event: done\n" in result.stdout
    assert b"forbidden" not in result.stdout


if sys.platform == "win32":
    pytestmark = pytest.mark.filterwarnings("ignore:.*unclosed.*:ResourceWarning")
