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
    content = out["messages"][0].content
    assert "[tool-error]" in content
    assert "boom" in content
    assert "kaboom" in content
    assert "Traceback" not in content
    assert 'File "' not in content


@pytest.mark.asyncio
async def test_normal_dict_carries_no_block_kwargs() -> None:
    def fetchy(url: str) -> dict[str, Any]:
        """Fetch a URL."""
        return {"url": url, "title": "ok", "content": "hello", "blocked": False}

    out = await tool_executor_node(
        _state_with_call("fetchy", {"url": "https://x.com/p"}),
        {"configurable": {"tool_registry": _registry_with(fetchy, "fetchy")}},
    )
    assert not out["messages"][0].additional_kwargs.get("blocked")
