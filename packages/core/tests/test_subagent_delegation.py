"""Sub-agent delegation: an orchestrator delegates a task to a worker agent.

A worker is declared as an :class:`AgentSpec` (009) and run in an *isolated*
fresh state seeded only with the delegated task — no parent context leaks in,
only the worker's final answer comes out. ``make_subagent_tool`` wraps that as
an LLM-callable tool so an orchestrator agent can invoke it like any other tool.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import litellm
import pytest
from neuroagent.config import AgentSpec
from neuroagent.models import LLMProvider, LLMProviderRegistry
from neuroagent.orchestration import make_subagent_tool, run_subagent
from neuroagent.tools.registry import ToolRegistry


@pytest.fixture()
def llm_registry() -> LLMProviderRegistry:
    reg = LLMProviderRegistry()
    reg.register(
        LLMProvider(
            name="higress",
            provider_kind="openai_compatible",
            default_model="openai/deepseek-v4-flash",
        )
    )
    return reg


def _spec(name: str = "researcher") -> AgentSpec:
    return AgentSpec(
        name=name,
        system_prompt="You are a focused worker. Answer concisely.",
        llm_provider="higress",
    )


def _final(content: str) -> Any:
    """An LLM response with no tool_calls → the agent's final answer."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None))],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
    )


class TestRunSubagent:
    async def test_returns_worker_final_answer(
        self, llm_registry: LLMProviderRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        acompletion = AsyncMock(return_value=_final("the capital is Paris"))
        monkeypatch.setattr(litellm, "acompletion", acompletion)

        answer = await run_subagent(
            _spec(),
            "What is the capital of France?",
            tool_registry=ToolRegistry(),
            llm_registry=llm_registry,
        )

        assert answer == "the capital is Paris"

    async def test_isolated_state_seeded_only_with_task(
        self, llm_registry: LLMProviderRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        acompletion = AsyncMock(return_value=_final("ok"))
        monkeypatch.setattr(litellm, "acompletion", acompletion)

        await run_subagent(
            _spec(),
            "do the thing",
            tool_registry=ToolRegistry(),
            llm_registry=llm_registry,
        )

        # The worker only ever sees its own system prompt + the delegated task,
        # never any parent conversation history.
        sent = acompletion.await_args.kwargs["messages"]
        roles = [m["role"] for m in sent]
        assert roles[0] == "system"
        assert sent[0]["content"].startswith("You are a focused worker")
        user_msgs = [m for m in sent if m["role"] == "user"]
        assert user_msgs == [{"role": "user", "content": "do the thing"}]


class TestMakeSubagentTool:
    async def test_tool_name_is_slugged_delegate(self, llm_registry: LLMProviderRegistry) -> None:
        tool = make_subagent_tool(
            _spec("Research Bot"),
            tool_registry=ToolRegistry(),
            llm_registry=llm_registry,
        )
        assert tool.__name__ == "delegate_to_research_bot"
        assert "Research Bot" in (tool.__doc__ or "")

    async def test_tool_invocation_runs_worker(
        self, llm_registry: LLMProviderRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        acompletion = AsyncMock(return_value=_final("delegated result"))
        monkeypatch.setattr(litellm, "acompletion", acompletion)

        tool = make_subagent_tool(
            _spec(),
            tool_registry=ToolRegistry(),
            llm_registry=llm_registry,
        )
        result = await tool("summarise this")

        assert result == "delegated result"

    async def test_worker_failure_returns_error_string_not_raise(
        self, llm_registry: LLMProviderRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        acompletion = AsyncMock(side_effect=RuntimeError("worker boom"))
        monkeypatch.setattr(litellm, "acompletion", acompletion)

        tool = make_subagent_tool(
            _spec("researcher"),
            tool_registry=ToolRegistry(),
            llm_registry=llm_registry,
        )
        result = await tool("anything")

        # The orchestrator must not crash when a sub-agent fails.
        assert "[subagent-error]" in result
        assert "researcher" in result

    async def test_tool_is_registrable_for_an_orchestrator(
        self, llm_registry: LLMProviderRegistry
    ) -> None:
        # The delegate tool must look like any other harness tool (named callable
        # with a typed signature) so an orchestrator agent can list it.
        from neuroagent.agents.simple import _to_tool_def

        tool = make_subagent_tool(
            _spec("planner"),
            tool_registry=ToolRegistry(),
            llm_registry=llm_registry,
        )
        schema = _to_tool_def(tool)
        assert schema["function"]["name"] == "delegate_to_planner"
        assert "task" in schema["function"]["parameters"]["properties"]
