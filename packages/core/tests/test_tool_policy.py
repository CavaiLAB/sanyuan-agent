"""Tool permission policy: glob rules, executor enforcement, spec wiring.

``ToolPolicy`` (014) decides per agent whether a tool may be called at all —
``deny`` blocks outright, ``confirm`` requires the call id in
``confirmed_tool_calls`` (the risk-gate flow), ``allow`` passes through. It is
enforced in ``tool_executor_node`` and declared on ``AgentSpec.tool_policy``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import litellm
import pytest
from pydantic import ValidationError
from sanyuan_agent.agents.tool_executor import tool_executor_node
from sanyuan_agent.config import AgentSpec, load_agent_from_spec
from sanyuan_agent.models import LLMProvider, LLMProviderRegistry
from sanyuan_agent.security import PolicyRule, ToolPolicy
from sanyuan_agent.tools.registry import ToolRegistry, callable_to_schema


class TestToolPolicyDecide:
    def test_first_matching_rule_wins(self) -> None:
        policy = ToolPolicy(
            [PolicyRule("bash", "deny"), PolicyRule("*", "confirm")],
        )
        assert policy.decide("bash") == "deny"
        assert policy.decide("web_search") == "confirm"

    def test_glob_pattern_scopes_tool_families(self) -> None:
        policy = ToolPolicy([PolicyRule("mcp__*", "confirm")], default="allow")
        assert policy.decide("mcp__docs__echo") == "confirm"
        assert policy.decide("web_search") == "allow"

    def test_default_applies_when_no_rule_matches(self) -> None:
        assert ToolPolicy().decide("anything") == "allow"
        assert ToolPolicy(default="deny").decide("anything") == "deny"

    def test_invalid_action_or_pattern_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown policy action"):
            PolicyRule("bash", "block")
        with pytest.raises(ValueError, match="non-empty"):
            PolicyRule("", "deny")
        with pytest.raises(ValueError, match="unknown default action"):
            ToolPolicy(default="ask")


def _tool_call(name: str, args: dict[str, Any], tc_id: str = "tc-1") -> dict[str, Any]:
    return {
        "id": tc_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _state_with_call(tc: dict[str, Any]) -> dict[str, Any]:
    return {"messages": [{"role": "assistant", "content": "", "tool_calls": [tc]}]}


def _registry_with_echo() -> tuple[ToolRegistry, list[str]]:
    calls: list[str] = []

    def echo(text: str) -> str:
        """Echo text back."""
        calls.append(text)
        return f"echo: {text}"

    registry = ToolRegistry()
    registry.register("echo", echo, callable_to_schema(echo))
    return registry, calls


class TestExecutorEnforcement:
    async def _run(
        self,
        policy: ToolPolicy | None,
        *,
        confirmed: frozenset[str] = frozenset(),
    ) -> tuple[str, list[str]]:
        registry, calls = _registry_with_echo()
        result = await tool_executor_node(
            _state_with_call(_tool_call("echo", {"text": "hi"})),
            {
                "configurable": {
                    "tool_registry": registry,
                    "tool_policy": policy,
                    "confirmed_tool_calls": confirmed,
                }
            },
        )
        return result["messages"][0].content, calls

    async def test_deny_blocks_execution(self) -> None:
        content, calls = await self._run(ToolPolicy([PolicyRule("echo", "deny")]))
        assert content.startswith("[policy-denied]")
        assert calls == []

    async def test_confirm_without_confirmation_blocks(self) -> None:
        content, calls = await self._run(ToolPolicy([PolicyRule("e*", "confirm")]))
        assert content.startswith("[confirmation-required]")
        assert calls == []

    async def test_confirm_with_confirmed_id_executes(self) -> None:
        content, calls = await self._run(
            ToolPolicy([PolicyRule("e*", "confirm")]),
            confirmed=frozenset({"tc-1"}),
        )
        assert content == "echo: hi"
        assert calls == ["hi"]

    async def test_allow_and_absent_policy_execute(self) -> None:
        for policy in (ToolPolicy([PolicyRule("echo", "allow")]), None):
            content, calls = await self._run(policy)
            assert content == "echo: hi"
            assert calls == ["hi"]

    async def test_default_deny_blocks_unlisted_tool(self) -> None:
        content, calls = await self._run(
            ToolPolicy([PolicyRule("web_*", "allow")], default="deny")
        )
        assert content.startswith("[policy-denied]")
        assert calls == []


class TestAgentSpecPolicy:
    def test_defaults_are_empty_allow(self) -> None:
        spec = AgentSpec(name="a", system_prompt="", llm_provider="higress")
        assert spec.tool_policy == []
        assert spec.tool_policy_default == "allow"

    def test_rules_parse_and_validate(self) -> None:
        spec = AgentSpec(
            name="a",
            system_prompt="",
            llm_provider="higress",
            tool_policy=[{"pattern": "mcp__*", "action": "confirm"}],
            tool_policy_default="deny",
        )
        assert spec.tool_policy[0].pattern == "mcp__*"
        with pytest.raises(ValidationError):
            AgentSpec(
                name="a",
                system_prompt="",
                llm_provider="higress",
                tool_policy=[{"pattern": "x", "action": "block"}],
            )
        with pytest.raises(ValidationError):
            AgentSpec(
                name="a",
                system_prompt="",
                llm_provider="higress",
                tool_policy=[{"pattern": "x", "action": "deny", "extra": 1}],
            )


def _llm_turns(*turns: Any) -> AsyncMock:
    return AsyncMock(side_effect=list(turns))


def _tool_call_response(name: str, args: dict[str, Any]) -> Any:
    tc = SimpleNamespace(
        id="tc-1",
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
        model_dump=lambda: _tool_call(name, args),
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=[tc]))],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
    )


def _final_response(content: str) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None))],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
    )


class TestLoaderWiring:
    async def test_spec_policy_denies_through_built_agent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end: an agent built from a spec with a deny rule never runs the tool."""
        llm_registry = LLMProviderRegistry()
        llm_registry.register(
            LLMProvider(
                name="higress",
                provider_kind="openai_compatible",
                default_model="openai/deepseek-v4-flash",
            )
        )
        registry, calls = _registry_with_echo()
        spec = AgentSpec(
            name="locked-down",
            system_prompt="",
            llm_provider="higress",
            tools=["echo"],
            tool_policy=[{"pattern": "echo", "action": "deny"}],
        )
        monkeypatch.setattr(
            litellm,
            "acompletion",
            _llm_turns(
                _tool_call_response("echo", {"text": "hi"}),
                _final_response("done"),
            ),
        )

        loaded = load_agent_from_spec(
            spec, tool_registry=registry, llm_registry=llm_registry
        )
        state = {
            "messages": loaded.initial_messages("call echo"),
            "next_action": None,
            "tool_results": [],
            "iteration": 0,
            "max_iter": spec.max_iterations,
            "sensory": None,
            "working": None,
            "usage": {},
        }
        result = await loaded.graph.ainvoke(state)

        assert calls == []  # the tool itself never ran
        contents = [
            getattr(m, "content", m.get("content", "") if isinstance(m, dict) else "")
            for m in result["messages"]
        ]
        assert any("[policy-denied]" in str(c) for c in contents)
