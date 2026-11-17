"""Fail-closed stdin JSON to stdout SSE entry for one desktop agent run.

The Electron sidecar writes exactly one JSON line and treats stdout as an SSE-only
channel.  This module validates that untrusted payload at the process boundary,
builds a no-tools agent from an explicit local BYOK configuration, and streams the
harness-owned SSE protocol.  Local shell, files, browser, skills, attachments and
host credentials stay disabled until their grant/materialization seams are wired.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, BinaryIO, Literal, TextIO
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from sanyuan_agent.agents import build_simple_agent
from sanyuan_agent.models import LLMProvider
from sanyuan_agent.sse import EVENT_ERROR, _format_sse_event, sse_stream

_MAX_REQUEST_BYTES = 256 * 1024
_LOCAL_API_KEY_ENV = "ICLAW_LOCAL_LLM_API_KEY"
_LOCAL_API_BASE_ENV = "ICLAW_LOCAL_LLM_API_BASE"
_LOCAL_MODEL_ENV = "ICLAW_LOCAL_LLM_MODEL"


class DesktopRunProtocolError(ValueError):
    """Stable, non-sensitive desktop protocol rejection."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class _HistoryMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=10_000)

    @field_validator("content")
    @classmethod
    def _content_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("history content must not be blank")
        return value


class _DesktopRunRequest(BaseModel):
    """Safe subset of the Edge-authorized payload consumed by the local runner.

    Edge, not this process, owns execution routing.  Legacy device requests and
    project-fixed conversations deliberately arrive with different (or stale)
    ``mode``/``target``/``target_device_id`` fields, so those routing hints stay
    outside this model and are ignored.  Invoking the dedicated ``desktop-run``
    command is the trusted local-routing boundary; this model only gates the
    capabilities the process would actually materialize.
    """

    model_config = ConfigDict(extra="ignore", strict=True)

    conversation_id: str | None = Field(default=None, max_length=256)
    project_id: str | None = Field(default=None, max_length=256)
    content: str = Field(min_length=1, max_length=10_000)
    autonomy: Literal["once", "continuous"] | None = None
    browser: bool = False
    history: list[_HistoryMessage] = Field(default_factory=list, max_length=20)
    reply_language: Literal["zh", "en"] | None = None
    personality: Literal["amiable", "pragmatic"] | None = None
    timezone: str | None = Field(default=None, max_length=64)
    custom_instructions: str | None = Field(default=None, max_length=4_000)
    selected_skills: list[str] = Field(default_factory=list, max_length=100)
    inputs: list[Any] = Field(default_factory=list, max_length=100)
    workspace: Any | None = None
    sandbox: Any | None = None
    tools: Any | None = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    @field_validator("content")
    @classmethod
    def _required_text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("required text must not be blank")
        return value

    @model_validator(mode="after")
    def _enforce_current_capability_boundary(self) -> _DesktopRunRequest:
        if self.autonomy == "continuous":
            raise ValueError("continuous autonomy is not implemented")
        if self.browser:
            raise ValueError("browser execution is not implemented")
        if self.selected_skills:
            raise ValueError("desktop skill materialization is not implemented")
        if self.inputs:
            raise ValueError("desktop input materialization is not implemented")
        if any(value is not None for value in (self.workspace, self.sandbox, self.tools)):
            raise ValueError("caller-supplied execution objects are forbidden")
        return self


def _read_request(stdin: BinaryIO) -> _DesktopRunRequest:
    raw = stdin.read(_MAX_REQUEST_BYTES + 1)
    if len(raw) > _MAX_REQUEST_BYTES:
        raise DesktopRunProtocolError(
            "REQUEST_TOO_LARGE", "local desktop request exceeds the byte limit"
        )
    if not raw:
        raise DesktopRunProtocolError("EMPTY_REQUEST", "local desktop request is empty")

    if raw.endswith(b"\r\n"):
        body = raw[:-2]
    elif raw.endswith(b"\n"):
        body = raw[:-1]
    else:
        body = raw
    if not body or b"\n" in body or b"\r" in body:
        raise DesktopRunProtocolError(
            "INVALID_FRAMING", "local desktop protocol requires exactly one JSON line"
        )
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DesktopRunProtocolError(
            "INVALID_UTF8", "local desktop request must be UTF-8 JSON"
        ) from exc
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DesktopRunProtocolError(
            "MALFORMED_JSON", "local desktop request is not valid JSON"
        ) from exc
    try:
        return _DesktopRunRequest.model_validate(value)
    except ValidationError as exc:
        raise DesktopRunProtocolError(
            "INVALID_REQUEST", "local desktop request failed capability validation"
        ) from exc


def _trusted_llm_provider(env: Mapping[str, str]) -> LLMProvider:
    model = env.get(_LOCAL_MODEL_ENV, "").strip()
    api_base = env.get(_LOCAL_API_BASE_ENV, "").strip()
    api_key = env.get(_LOCAL_API_KEY_ENV, "")
    model_id = model.removeprefix("openai/")
    if (
        not model.startswith("openai/")
        or not model_id
        or len(model) > 200
        or any(char.isspace() or char == "\x00" for char in model)
    ):
        raise DesktopRunProtocolError(
            "LOCAL_LLM_CONFIG_UNAVAILABLE",
            "local desktop LLM model must use the openai/<model> route",
        )
    if not api_key or any(char in api_key for char in "\r\n\x00"):
        raise DesktopRunProtocolError(
            "LOCAL_LLM_CONFIG_UNAVAILABLE",
            "local desktop LLM credential is not explicitly configured",
        )
    try:
        parsed = urlsplit(api_base)
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError as exc:
        raise DesktopRunProtocolError(
            "LOCAL_LLM_CONFIG_UNAVAILABLE",
            "local desktop LLM endpoint is invalid",
        ) from exc
    loopback = hostname in {"localhost", "127.0.0.1", "::1"}
    if (
        not hostname
        or parsed.scheme not in {"http", "https"}
        or (parsed.scheme == "http" and not loopback)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise DesktopRunProtocolError(
            "LOCAL_LLM_CONFIG_UNAVAILABLE",
            "local desktop LLM endpoint must be HTTPS or loopback HTTP",
        )
    return LLMProvider(
        name="local-desktop",
        provider_kind="openai_compatible",
        api_base=api_base,
        api_key_env=_LOCAL_API_KEY_ENV,
        default_model=model,
    )


def _messages(request: _DesktopRunRequest) -> list[dict[str, str]]:
    preferences: list[str] = []
    if request.reply_language:
        preferences.append(
            "Reply in Simplified Chinese."
            if request.reply_language == "zh"
            else "Reply in English."
        )
    if request.personality:
        preferences.append(
            "Use a warm and approachable tone."
            if request.personality == "amiable"
            else "Use a concise and pragmatic tone."
        )
    if request.timezone:
        preferences.append(f"The user's timezone is {request.timezone}.")
    if request.custom_instructions:
        preferences.append("User-provided instructions:\n" + request.custom_instructions)

    messages: list[dict[str, str]] = []
    if preferences:
        messages.append({"role": "system", "content": "\n".join(preferences)})
    messages.extend(message.model_dump() for message in request.history)
    messages.append({"role": "user", "content": request.content})
    return messages


def _event_type(frame: str) -> str | None:
    first = frame.splitlines()[0] if frame else ""
    if not first.startswith("event:"):
        return None
    return first.removeprefix("event:").strip()


def _emit(stdout: TextIO, frame: str) -> None:
    stdout.write(frame)
    stdout.flush()


async def run_desktop_stdio(
    stdin: BinaryIO,
    stdout: TextIO,
    *,
    env: Mapping[str, str],
) -> int:
    """Run one validated desktop request; return a process-style exit code."""

    try:
        request = _read_request(stdin)
        provider = _trusted_llm_provider(env)
    except DesktopRunProtocolError as exc:
        _emit(
            stdout,
            _format_sse_event(EVENT_ERROR, {"code": exc.code, "message": exc.message}),
        )
        return 2

    try:
        agent = build_simple_agent(
            provider,
            [],
            max_iter=10,
            temperature=request.temperature,
            tenant_id="",
            conversation_id=request.conversation_id or "",
            project_id=request.project_id or "",
        )
    except Exception:
        _emit(
            stdout,
            _format_sse_event(
                EVENT_ERROR,
                {
                    "code": "LOCAL_RUNTIME_INITIALIZATION_FAILED",
                    "message": "local desktop runtime could not be initialized",
                },
            ),
        )
        return 1

    state: dict[str, Any] = {
        "messages": _messages(request),
        "next_action": None,
        "tool_results": [],
        "iteration": 0,
        "max_iter": 10,
        "sensory": None,
        "working": None,
        "usage": {},
    }
    saw_error = False
    async for frame in sse_stream(agent, state):
        if _event_type(frame) == EVENT_ERROR:
            # Provider/runtime exceptions can contain endpoints, response bodies, or
            # credential fragments.  Keep the shared SSE helper useful to the web app,
            # but sanitize its process-bound error frame before it reaches Electron.
            frame = _format_sse_event(
                EVENT_ERROR,
                {
                    "code": "LOCAL_RUNTIME_FAILED",
                    "message": "local desktop runtime failed",
                },
            )
            saw_error = True
        _emit(stdout, frame)
    return 1 if saw_error else 0


__all__ = [
    "DesktopRunProtocolError",
    "run_desktop_stdio",
]
