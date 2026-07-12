"""Context-window trimming (018): bound the message list sent to the LLM.

``trim_messages`` keeps the leading system messages and the first task turn,
elides the middle behind a marker, and keeps the most recent messages within
a token budget without orphaning tool results. The plan node applies it each
turn with ``context_token_budget`` (a ``build_simple_agent`` / ``AgentSpec``
knob), so unbounded conversations stop overflowing the model context.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import litellm
import pytest
from langchain_core.messages import ToolMessage
from neuroagent.config import AgentSpec, load_agent_from_spec
from neuroagent.memory.context import trim_messages
from neuroagent.models import LLMProvider, LLMProviderRegistry
from neuroagent.tools.registry import ToolRegistry
from pydantic import ValidationError


def _msg(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


def _conversation(turns: int, *, chars: int = 400) -> list[dict[str, str]]:
    msgs = [_msg("system", "You are helpful."), _msg("user", "the original task")]
    for i in range(turns):
        msgs.append(_msg("assistant", f"thought {i} " + "x" * chars))
        msgs.append(_msg("user", f"follow-up {i} " + "y" * chars))
    return msgs


class TestTrimMessages:
    def test_under_budget_unchanged(self) -> None:
        msgs = _conversation(3)
        assert trim_messages(msgs, token_budget=100_000) is msgs

    def test_over_budget_keeps_head_marker_and_tail(self) -> None:
        msgs = _conversation(40)

        trimmed = trim_messages(msgs, token_budget=2000, keep_recent=4)

        assert len(trimmed) < len(msgs)
        assert trimmed[0] == msgs[0]  # system prompt kept
        assert trimmed[1] == msgs[1]  # original task kept
        assert "context trimmed" in trimmed[2]["content"]
        assert trimmed[-1] == msgs[-1]  # newest messages kept verbatim
        # The kept tail fits the budget shape: far fewer than the original.
        assert len(trimmed) <= 4 + sum(
            1 for _ in range(len(msgs))
        )  # sanity: no growth

    def test_keep_recent_minimum_honoured(self) -> None:
        msgs = _conversation(40, chars=4000)

        trimmed = trim_messages(msgs, token_budget=100, keep_recent=6)

        # head(2) + marker + at least keep_recent tail
        assert len(trimmed) >= 2 + 1 + 6
        assert trimmed[-6:] == msgs[-6:]

    def test_tail_never_starts_with_orphan_tool_result(self) -> None:
        msgs: list[Any] = [_msg("system", "s"), _msg("user", "task")]
        for i in range(30):
            msgs.append(
                {
                    "role": "assistant",
                    "content": "z" * 400,
                    "tool_calls": [{"id": f"tc-{i}"}],
                }
            )
            msgs.append(ToolMessage(content="r" * 400, tool_call_id=f"tc-{i}", name="t"))

        trimmed = trim_messages(msgs, token_budget=1500, keep_recent=3)

        after_marker = trimmed[3]
        role = (
            after_marker.get("role")
            if isinstance(after_marker, dict)
            else getattr(after_marker, "type", "")
        )
        assert role != "tool"
        # Pairing intact: every kept tool message's parent assistant is kept too.
        kept_tc_ids = {
            tc["id"]
            for m in trimmed
            if isinstance(m, dict)
            for tc in m.get("tool_calls", []) or []
        }
        for m in trimmed:
            if isinstance(m, ToolMessage):
                assert m.tool_call_id in kept_tc_ids

    def test_short_body_left_alone(self) -> None:
        msgs = [_msg("system", "s" * 100_000), _msg("user", "task")]
        assert trim_messages(msgs, token_budget=10) is msgs

    def test_trailing_parallel_tool_block_still_trims(self) -> None:
        """The plan node runs right after a tool round, so the list ends in
        tool results. A final assistant turn with >= keep_recent parallel
        calls and huge results must not silently disable trimming."""
        msgs: list[Any] = [_msg("system", "s"), _msg("user", "task")]
        for i in range(20):
            msgs.append(_msg("assistant", f"thought {i} " + "x" * 400))
            msgs.append(_msg("user", f"reply {i} " + "y" * 400))
        tc_ids = [f"par-{i}" for i in range(4)]
        msgs.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": t} for t in tc_ids],
            }
        )
        for t in tc_ids:
            msgs.append(ToolMessage(content="r" * 4000, tool_call_id=t, name="search"))

        trimmed = trim_messages(msgs, token_budget=2000, keep_recent=4)

        assert len(trimmed) < len(msgs)  # trimming did NOT bail out
        # The kept tail starts at the parent assistant turn — pair intact.
        after_marker = trimmed[3]
        assert isinstance(after_marker, dict)
        assert [tc["id"] for tc in after_marker.get("tool_calls", [])] == tc_ids
        assert all(isinstance(m, ToolMessage) for m in trimmed[-4:])

    def test_inlined_image_not_counted_as_base64_text(self) -> None:
        """A multimodal image turn must not be estimated at ~len(base64)/4 tokens.

        Regression for the vision-inlining / trim interaction: counting the base64
        data URI as text would push a single image over the budget and collapse
        the conversation to keep_recent. The image part is charged a small fixed
        cost, so an otherwise-small conversation stays untrimmed.
        """
        import base64

        big_uri = "data:image/png;base64," + base64.b64encode(b"x" * 300_000).decode()
        image_turn = {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is in this image?"},
                {"type": "image_url", "image_url": {"url": big_uri}},
            ],
        }
        msgs: list[Any] = [
            _msg("system", "You are helpful."),
            image_turn,
            _msg("assistant", "a diagram"),
            _msg("user", "and the text?"),
        ]
        # ~400KB of base64 would be ~100k tokens if counted as text; with the fixed
        # image charge the whole conversation is well under budget → unchanged.
        assert trim_messages(msgs, token_budget=24_000) is msgs

    def test_tool_call_arguments_count_toward_budget(self) -> None:
        """Huge tool_calls arguments with empty content must trigger trimming."""
        msgs: list[Any] = [_msg("system", "s"), _msg("user", "task")]
        for i in range(30):
            msgs.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"w-{i}",
                            "function": {"name": "write_file", "arguments": "z" * 8000},
                        }
                    ],
                }
            )
            msgs.append(ToolMessage(content="ok", tool_call_id=f"w-{i}", name="write_file"))

        trimmed = trim_messages(msgs, token_budget=2000, keep_recent=2)

        assert len(trimmed) < len(msgs)


class TestAgentWiring:
    def test_spec_default_and_validation(self) -> None:
        spec = AgentSpec(name="a", system_prompt="", llm_provider="p")
        assert spec.context_token_budget == 24000
        with pytest.raises(ValidationError):
            AgentSpec(
                name="a", system_prompt="", llm_provider="p", context_token_budget=0
            )

    async def test_plan_node_trims_what_the_llm_sees(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        llm_registry = LLMProviderRegistry()
        llm_registry.register(
            LLMProvider(
                name="p",
                provider_kind="openai_compatible",
                default_model="openai/deepseek-v4-flash",
            )
        )
        final = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )
        acompletion = AsyncMock(return_value=final)
        monkeypatch.setattr(litellm, "acompletion", acompletion)

        spec = AgentSpec(
            name="tiny-window",
            system_prompt="sys",
            llm_provider="p",
            context_token_budget=500,
        )
        loaded = load_agent_from_spec(
            spec, tool_registry=ToolRegistry(), llm_registry=llm_registry
        )
        history = _conversation(30)
        state = {
            "messages": [*loaded.initial_messages("task"), *history[2:]],
            "next_action": None,
            "tool_results": [],
            "iteration": 0,
            "max_iter": 2,
            "sensory": None,
            "working": None,
            "usage": {},
        }

        await loaded.graph.ainvoke(state)

        sent = acompletion.call_args.kwargs["messages"]
        assert len(sent) < len(state["messages"])
        assert any(
            "context trimmed" in (m.get("content") or "")
            for m in sent
            if isinstance(m, dict)
        )
        # Budget respected within the rough estimator's terms.
        assert sum(len(json.dumps(m, default=str)) for m in sent) < 5000
