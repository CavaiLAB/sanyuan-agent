"""The tool executor lifts a tool's structured block signal onto the ToolMessage.

A web tool that lands on an anti-bot page returns ``{..., "blocked": True,
"block_reason": ...}``; the executor copies those onto
``ToolMessage.additional_kwargs`` so the SSE layer can flag it (the call
succeeded with HTTP 200, so it is not an ``is_error``).
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from neuroagent.agents.tool_executor import tool_executor_node
from neuroagent.tools.registry import ToolRegistry, callable_to_schema


def _state_with_call(name: str, args: dict[str, Any], tc_id: str = "tc-1") -> dict[str, Any]:
    return {
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": tc_id,
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                ],
            }
        ]
    }


def _registry_with(tool: Any, name: str) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(name, tool, callable_to_schema(tool))
    return reg


@pytest.mark.asyncio
async def test_blocked_dict_lifts_to_additional_kwargs() -> None:
    def fetchy(url: str) -> dict[str, Any]:
        """Fetch a URL."""
        return {
            "url": url,
            "title": "访问已被拦截",
            "content": "访问已被拦截",
            "blocked": True,
            "block_reason": "目标站点返回了反爬/验证页",
        }

    out = await tool_executor_node(
        _state_with_call("fetchy", {"url": "https://x.com/forbidden"}),
        {"configurable": {"tool_registry": _registry_with(fetchy, "fetchy")}},
    )
    msg = out["messages"][0]
    assert msg.additional_kwargs.get("blocked") is True
    assert msg.additional_kwargs.get("block_reason") == "目标站点返回了反爬/验证页"


@pytest.mark.asyncio
async def test_blocked_without_reason_sets_only_flag() -> None:
    def fetchy(url: str) -> dict[str, Any]:
        """Fetch a URL."""
        return {"url": url, "blocked": True, "block_reason": None}

    out = await tool_executor_node(
        _state_with_call("fetchy", {"url": "https://x.com/sec-check"}),
        {"configurable": {"tool_registry": _registry_with(fetchy, "fetchy")}},
    )
    extra = out["messages"][0].additional_kwargs
    assert extra.get("blocked") is True
    assert "block_reason" not in extra


@pytest.mark.asyncio
async def test_execution_error_is_clean_with_no_traceback() -> None:
    # A tool that raises must surface a clean one-line reason to the chat — never
    # the raw Python traceback / stack frames (those go to the server log only).
    def boom(path: str) -> str:
        """A tool that always fails."""
        raise RuntimeError("kaboom")

    out = await tool_executor_node(
        _state_with_call("boom", {"path": "x"}),
        {"configurable": {"tool_registry": _registry_with(boom, "boom")}},
    )
    msg = out["messages"][0]
    content = msg.content
    assert "[tool-error]" in content
    assert "boom" in content
    assert "kaboom" in content
    assert "Traceback" not in content
    assert 'File "' not in content
    # The failure is marked status="error" so the SSE layer flags is_error and the
    # UI renders it red instead of a green success.
    assert msg.status == "error"


@pytest.mark.asyncio
async def test_unknown_tool_marks_status_error() -> None:
    # The model requested a tool that isn't registered: a structured error listing
    # the available tools, marked status="error" — never a crash, never a success.
    def echo(msg: str) -> str:
        """Echo."""
        return msg

    out = await tool_executor_node(
        _state_with_call("no_such_tool", {}),
        {"configurable": {"tool_registry": _registry_with(echo, "echo")}},
    )
    msg = out["messages"][0]
    assert "[tool-error]" in msg.content
    assert "not found" in msg.content
    assert msg.status == "error"


@pytest.mark.asyncio
async def test_successful_tool_has_no_error_status() -> None:
    def fetchy(url: str) -> dict[str, Any]:
        """Fetch a URL."""
        return {"url": url, "title": "ok", "content": "hello", "blocked": False}

    out = await tool_executor_node(
        _state_with_call("fetchy", {"url": "https://x.com/p"}),
        {"configurable": {"tool_registry": _registry_with(fetchy, "fetchy")}},
    )
    msg = out["messages"][0]
    assert not msg.additional_kwargs.get("blocked")
    # A successful tool must NOT be flagged as an error (default status).
    assert msg.status != "error"
