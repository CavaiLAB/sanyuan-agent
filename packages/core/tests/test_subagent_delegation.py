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
from sanyuan_agent.config import AgentSpec
from sanyuan_agent.models import LLMProvider, LLMProviderRegistry
from sanyuan_agent.orchestration import make_subagent_tool, run_subagent, run_subagent_outcome
from sanyuan_agent.tools.registry import ToolRegistry


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


class TestRunSubagentOutcome:
    async def test_returns_answer_and_usage(
        self, llm_registry: LLMProviderRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        acompletion = AsyncMock(return_value=_final("structured answer"))
        monkeypatch.setattr(litellm, "acompletion", acompletion)

        outcome = await run_subagent_outcome(
            _spec(),
            "quantify yourself",
            tool_registry=ToolRegistry(),
            llm_registry=llm_registry,
        )

        assert outcome.answer == "structured answer"
        # _final reports prompt_tokens=3 / completion_tokens=2; the outcome
        # carries the camelCase wire shape with totalTokens derived.
        assert outcome.usage == {
            "promptTokens": 3,
            "completionTokens": 2,
            "totalTokens": 5,
        }

    async def test_run_subagent_string_contract_unchanged(
        self, llm_registry: LLMProviderRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(litellm, "acompletion", AsyncMock(return_value=_final("plain")))

        answer = await run_subagent(
            _spec(),
            "keep it simple",
            tool_registry=ToolRegistry(),
            llm_registry=llm_registry,
        )

        assert isinstance(answer, str)
        assert answer == "plain"


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
        from sanyuan_agent.agents.simple import _to_tool_def

        tool = make_subagent_tool(
            _spec("planner"),
            tool_registry=ToolRegistry(),
            llm_registry=llm_registry,
        )
        schema = _to_tool_def(tool)
        assert schema["function"]["name"] == "delegate_to_planner"
        assert "task" in schema["function"]["parameters"]["properties"]


def _stream_chunk(content=None, usage=None):
    delta = SimpleNamespace(content=content, tool_calls=None)
    choices = [SimpleNamespace(delta=delta)] if content else []
    return SimpleNamespace(choices=choices, usage=usage)


def _fake_stream(chunks):
    async def acompletion(**kwargs):
        assert kwargs.get("stream") is True

        async def gen():
            for c in chunks:
                yield c

        return gen()

    return acompletion


class TestStreamFinalDeltas:
    """stream_final: live final-answer deltas ride the observer channel."""

    async def test_deltas_flow_through_real_graph_custom_channel(
        self, llm_registry: LLMProviderRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Content long enough to cross the plan node's 160-char hold window,
        # so deltas flow live (not just the end-of-stream flush).
        part_a, part_b, part_c = "a" * 100, "b" * 100, "c" * 10
        chunks = [
            _stream_chunk(content=part_a),
            _stream_chunk(content=part_b),
            _stream_chunk(content=part_c),
            _stream_chunk(usage=SimpleNamespace(prompt_tokens=7, completion_tokens=11)),
        ]
        monkeypatch.setattr(litellm, "acompletion", _fake_stream(chunks))

        events: list[dict[str, Any]] = []
        outcome = await run_subagent_outcome(
            _spec(),
            "write a long report",
            tool_registry=ToolRegistry(),
            llm_registry=llm_registry,
            tool_event_observer=events.append,
            stream_final=True,
        )

        deltas = [e for e in events if e["kind"] == "final_delta"]
        assert len(deltas) >= 2  # hold flush + live tail, not one end dump
        assert "".join(d["content"] for d in deltas) == outcome.answer
        assert outcome.answer == part_a + part_b + part_c
        # Streaming kept the token accounting (stream_options include_usage).
        assert outcome.usage == {
            "promptTokens": 7,
            "completionTokens": 11,
            "totalTokens": 18,
        }

    async def test_stream_final_without_observer_stays_non_streaming(
        self, llm_registry: LLMProviderRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # No observer = nobody to receive deltas: the run must keep the plain
        # ainvoke path and non-streaming LLM calls.
        acompletion = AsyncMock(return_value=_final("plain answer"))
        monkeypatch.setattr(litellm, "acompletion", acompletion)

        outcome = await run_subagent_outcome(
            _spec(),
            "task",
            tool_registry=ToolRegistry(),
            llm_registry=llm_registry,
            stream_final=True,
        )

        assert outcome.answer == "plain answer"
        assert acompletion.await_args.kwargs.get("stream") is not True

    async def test_observer_without_stream_final_emits_no_deltas(
        self, llm_registry: LLMProviderRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        acompletion = AsyncMock(return_value=_final("observed answer"))
        monkeypatch.setattr(litellm, "acompletion", acompletion)

        events: list[dict[str, Any]] = []
        outcome = await run_subagent_outcome(
            _spec(),
            "task",
            tool_registry=ToolRegistry(),
            llm_registry=llm_registry,
            tool_event_observer=events.append,
        )

        assert outcome.answer == "observed answer"
        assert [e for e in events if e["kind"] == "final_delta"] == []
        assert acompletion.await_args.kwargs.get("stream") is not True


class TestToolResultArtifactExtraction:
    """artifact_write-style results surface as artifacts on tool_result events."""

    def test_artifact_write_result_carries_artifacts(self) -> None:
        from sanyuan_agent.orchestration.subagent import _tool_events_from_update

        content = (
            '{"artifact_id": "art-1", "download_url": "https://oss/x", '
            '"filename": "report.html", "content_type": "text/html", "sha256": "h"}'
        )
        events = _tool_events_from_update(
            "tool_executor",
            {
                "messages": [
                    {
                        "role": "tool",
                        "tool_call_id": "tc-9",
                        "name": "artifact_write",
                        "content": content,
                    }
                ]
            },
        )
        assert len(events) == 1
        assert events[0]["artifacts"] == [
            {"id": "art-1", "filename": "report.html", "url": "https://oss/x"}
        ]

    def test_plain_tool_result_has_no_artifacts_key(self) -> None:
        from sanyuan_agent.orchestration.subagent import _tool_events_from_update

        events = _tool_events_from_update(
            "tool_executor",
            {
                "messages": [
                    {
                        "role": "tool",
                        "tool_call_id": "tc-1",
                        "name": "tool_bash",
                        "content": "not json at all",
                    }
                ]
            },
        )
        assert len(events) == 1
        assert "artifacts" not in events[0]
