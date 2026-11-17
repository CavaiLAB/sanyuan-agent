"""Standalone desktop stdin JSON to stdout SSE protocol tests."""

from __future__ import annotations

import json
from io import BytesIO, StringIO
from typing import Any

import pytest
from sanyuan_agent.local_desktop import desktop_run as desktop_run_module
from sanyuan_agent.local_desktop.desktop_run import run_desktop_stdio


class _FinalAgent:
    def __init__(self, content: str = "local answer") -> None:
        self.content = content
        self.state: dict[str, Any] | None = None

    async def astream(self, state: dict[str, Any], stream_mode: list[str]):
        self.state = state
        assert stream_mode == ["updates", "custom"]
        yield {
            "plan": {
                "usage": {"promptTokens": 4, "completionTokens": 2},
                "messages": [
                    {
                        "role": "assistant",
                        "content": self.content,
                        "tool_calls": None,
                    }
                ],
            }
        }


class _FailingAgent:
    async def astream(self, _state: dict[str, Any], stream_mode: list[str]):
        assert stream_mode == ["updates", "custom"]
        raise RuntimeError("provider failure")
        yield  # pragma: no cover - makes this an async generator


def _request(**overrides: Any) -> dict[str, Any]:
    request: dict[str, Any] = {
        "conversation_id": "conv-1",
        "content": "hello",
        "mode": "agent",
        "agent": True,
        "target": "computer",
        "autonomy": "once",
        "target_device_id": "desktop-1",
        "history": [{"role": "assistant", "content": "previous"}],
    }
    request.update(overrides)
    return request


def _stdin(payload: dict[str, Any]) -> BytesIO:
    return BytesIO((json.dumps(payload) + "\n").encode())


def _env() -> dict[str, str]:
    return {
        "ICLAW_LOCAL_LLM_MODEL": "openai/local-test-model",
        "ICLAW_LOCAL_LLM_API_BASE": "https://llm.example.test/v1",
        "ICLAW_LOCAL_LLM_API_KEY": "local-test-key",
    }


@pytest.mark.asyncio
async def test_valid_request_streams_harness_sse_from_trusted_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _FinalAgent()
    captured: dict[str, Any] = {}

    def fake_build(provider: Any, tools: list[Any], **kwargs: Any) -> _FinalAgent:
        captured.update(provider=provider, tools=tools, kwargs=kwargs)
        return agent

    monkeypatch.setattr(desktop_run_module, "build_simple_agent", fake_build)
    stdout = StringIO()
    payload = _request(
        model="attacker-controlled-model",
        reply_language="zh",
        custom_instructions="Keep it short.",
    )

    exit_code = await run_desktop_stdio(_stdin(payload), stdout, env=_env())

    assert exit_code == 0
    assert captured["provider"].default_model == "openai/local-test-model"
    assert captured["provider"].api_key_env == "ICLAW_LOCAL_LLM_API_KEY"
    assert captured["tools"] == []
    assert captured["kwargs"]["conversation_id"] == "conv-1"
    assert agent.state is not None
    assert agent.state["messages"][-1] == {"role": "user", "content": "hello"}
    assert "attacker-controlled-model" not in repr(agent.state)
    frames = stdout.getvalue()
    assert frames.startswith("event: final_answer\n")
    assert '"content": "local answer"' in frames
    assert "event: done\n" in frames
    assert '"totalTokens": 6' in frames


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        # Pre-mode clients expressed local Agent intent solely with a device id.
        {"content": "legacy", "target_device_id": "desktop-legacy"},
        # Edge rewrites ASK hints before a persisted computer-project location
        # authoritatively routes the request to its bound device.
        {
            "content": "project fixed",
            "conversation_id": "conv-project",
            "mode": "ask",
            "agent": False,
            "target": "cloud",
            "autonomy": "once",
        },
    ],
)
async def test_edge_authorized_routing_variants_do_not_depend_on_body_hints(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, Any],
) -> None:
    agent = _FinalAgent()
    monkeypatch.setattr(
        desktop_run_module,
        "build_simple_agent",
        lambda *_args, **_kwargs: agent,
    )
    stdout = StringIO()

    exit_code = await run_desktop_stdio(_stdin(payload), stdout, env=_env())

    assert exit_code == 0
    assert agent.state is not None
    assert agent.state["messages"][-1]["content"] == payload["content"]
    assert "event: done\n" in stdout.getvalue()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        {"autonomy": "continuous"},
        {"browser": True},
        {"selected_skills": ["unsafe"]},
        {"inputs": [{"path": "/etc/passwd"}]},
        {"workspace": {"path": "/"}},
        {"sandbox": {}},
        {"tools": ["bash"]},
    ],
)
async def test_unwired_capabilities_are_rejected_before_agent_build(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, Any],
) -> None:
    def fail_build(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("agent build must not run")

    monkeypatch.setattr(desktop_run_module, "build_simple_agent", fail_build)
    stdout = StringIO()
    exit_code = await run_desktop_stdio(_stdin(_request(**overrides)), stdout, env=_env())
    assert exit_code == 2
    assert "event: error\n" in stdout.getvalue()
    assert "INVALID_REQUEST" in stdout.getvalue()


@pytest.mark.asyncio
async def test_missing_trusted_config_is_stable_and_does_not_echo_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        desktop_run_module,
        "build_simple_agent",
        lambda *_args, **_kwargs: pytest.fail("agent build must not run"),
    )
    stdout = StringIO()
    exit_code = await run_desktop_stdio(
        _stdin(_request(content="private user text")), stdout, env={}
    )
    assert exit_code == 2
    assert "LOCAL_LLM_CONFIG_UNAVAILABLE" in stdout.getvalue()
    assert "private user text" not in stdout.getvalue()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw", "code"),
    [
        (b"", "EMPTY_REQUEST"),
        (b"not-json\n", "MALFORMED_JSON"),
        (b"{}\n{}\n", "INVALID_FRAMING"),
        (b"\xff\n", "INVALID_UTF8"),
        (b"x" * (256 * 1024 + 1), "REQUEST_TOO_LARGE"),
    ],
    ids=[
        "empty-request",
        "malformed-json",
        "invalid-framing",
        "invalid-utf8",
        "request-too-large",
    ],
)
async def test_malformed_and_oversized_input_fails_closed(raw: bytes, code: str) -> None:
    stdout = StringIO()
    exit_code = await run_desktop_stdio(BytesIO(raw), stdout, env=_env())
    assert exit_code == 2
    assert code in stdout.getvalue()


@pytest.mark.asyncio
async def test_remote_plain_http_llm_endpoint_is_rejected() -> None:
    env = _env()
    env["ICLAW_LOCAL_LLM_API_BASE"] = "http://llm.example.test/v1"
    stdout = StringIO()
    exit_code = await run_desktop_stdio(_stdin(_request()), stdout, env=env)
    assert exit_code == 2
    assert "LOCAL_LLM_CONFIG_UNAVAILABLE" in stdout.getvalue()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    [
        "bedrock/anthropic.claude-3-haiku",
        "vertex_ai/gemini-2.0-flash",
        "ollama/llama3",
        "bare-model",
        "openai/",
    ],
)
async def test_non_openai_compatible_model_routes_are_rejected(model: str) -> None:
    env = _env()
    env["ICLAW_LOCAL_LLM_MODEL"] = model
    stdout = StringIO()

    exit_code = await run_desktop_stdio(_stdin(_request()), stdout, env=env)

    assert exit_code == 2
    assert "LOCAL_LLM_CONFIG_UNAVAILABLE" in stdout.getvalue()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoint",
    ["https://[::1", "https://example.test:bad", "https://example.test:99999"],
)
async def test_malformed_endpoint_is_a_sanitized_protocol_rejection(endpoint: str) -> None:
    env = _env()
    env["ICLAW_LOCAL_LLM_API_BASE"] = endpoint
    stdout = StringIO()

    exit_code = await run_desktop_stdio(_stdin(_request()), stdout, env=env)

    assert exit_code == 2
    assert "LOCAL_LLM_CONFIG_UNAVAILABLE" in stdout.getvalue()


@pytest.mark.asyncio
async def test_agent_stream_failure_returns_nonzero_after_error_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        desktop_run_module,
        "build_simple_agent",
        lambda *_args, **_kwargs: _FailingAgent(),
    )
    stdout = StringIO()
    exit_code = await run_desktop_stdio(_stdin(_request()), stdout, env=_env())
    assert exit_code == 1
    assert "event: error\n" in stdout.getvalue()
    assert "LOCAL_RUNTIME_FAILED" in stdout.getvalue()
    assert "provider failure" not in stdout.getvalue()
    assert "event: done\n" not in stdout.getvalue()


@pytest.mark.asyncio
async def test_agent_initialization_failure_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_build(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("secret initialization detail")

    monkeypatch.setattr(desktop_run_module, "build_simple_agent", fail_build)
    stdout = StringIO()
    exit_code = await run_desktop_stdio(_stdin(_request()), stdout, env=_env())
    assert exit_code == 1
    assert "LOCAL_RUNTIME_INITIALIZATION_FAILED" in stdout.getvalue()
    assert "secret initialization detail" not in stdout.getvalue()
