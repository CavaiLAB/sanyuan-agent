"""Task-specific skill and memory context follows the current user turn."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from neuroagent.agents.simple import (
    _extract_user_query,
    _maybe_inject_longterm,
    _maybe_inject_recall,
    _maybe_inject_skill,
    build_simple_agent,
)
from neuroagent.models import LLMProvider
from neuroagent.skills import Skill, SkillRegistry


def _conversation(content: Any) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": "Research an old topic"},
        {"role": "assistant", "content": "The old task is complete."},
        {"role": "user", "content": content},
        {"role": "assistant", "content": "Checking the current task."},
        {"role": "tool", "content": "A recent tool result"},
    ]


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("Summarize the new topic", "Summarize the new topic"),
        (
            [
                {"type": "text", "text": "Summarize this picture"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,SECRET"}},
                {"type": "text", "text": "Use two sentences"},
            ],
            "Summarize this picture\nUse two sentences",
        ),
        ([{"type": "image_url", "image_url": {"url": "https://example.test/picture"}}], ""),
        ("", ""),
    ],
)
def test_query_uses_latest_user_text_without_falling_back_to_history(
    content: Any, expected: str
) -> None:
    assert _extract_user_query(_conversation(content)) == expected


@pytest.mark.asyncio
async def test_both_memory_retrievers_receive_current_user_query() -> None:
    messages = _conversation("Summarize the new topic")
    recall = Mock()
    recall.recall = AsyncMock(return_value=[])
    longterm = Mock()
    longterm.retrieve_semantic = AsyncMock(return_value=[])

    await _maybe_inject_recall(messages, recall, "tenant-a", "conversation-a")
    await _maybe_inject_longterm(
        messages, longterm, "tenant-a", "user-a", "project-a", "conversation-a"
    )

    assert recall.recall.await_args.kwargs["query"] == "Summarize the new topic"
    assert longterm.retrieve_semantic.await_args.kwargs["query"] == "Summarize the new topic"
    assert recall.recall.await_args.kwargs["tenant_id"] == "tenant-a"
    context = longterm.retrieve_semantic.await_args.kwargs["context"]
    assert (context.project_id, context.session_id) == ("project-a", "conversation-a")


def test_standalone_skill_injection_follows_current_task(tmp_path: Path) -> None:
    registry = SkillRegistry()
    for name, keyword in (("research", "research"), ("summary", "summarize")):
        registry.register(
            Skill(
                name=name,
                version="1",
                description=f"Help with {name}",
                tools=[],
                prompt_template=f"Instructions for {name}",
                source_path=tmp_path / name / "SKILL.md",
                trigger_keywords=[keyword],
            )
        )

    result = _maybe_inject_skill(_conversation("Summarize the new topic"), registry)
    assert result[0]["content"] == "[Skill: summary] Instructions for summary"

    unrelated = _conversation("Hello")
    assert _maybe_inject_skill(unrelated, registry) is unrelated


@pytest.mark.asyncio
async def test_plan_keeps_current_task_and_retrieval_aligned_when_history_is_trimmed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    current_query = "Summarize the new topic"
    messages = _conversation(current_query)[:3]
    for index in range(6):
        messages.extend([
            {
                "role": "assistant", "content": "Reading the results.",
                "tool_calls": [{
                    "id": f"read-{index}", "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"notes"}'},
                }],
            },
            {"role": "tool", "tool_call_id": f"read-{index}", "content": "detail " * 500},
        ])
    skills = SkillRegistry()
    skills.register(Skill(
        name="summary", version="1", description="Summarize findings", tools=[],
        prompt_template="Current summary guidance", source_path=tmp_path / "SKILL.md",
        trigger_keywords=["summarize"],
    ))
    recall = Mock(recall=AsyncMock(return_value=[]))
    longterm = Mock(retrieve_semantic=AsyncMock(return_value=[]))
    llm = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Done", tool_calls=None))],
        usage=None,
    ))
    monkeypatch.setattr("neuroagent.agents.simple._call_llm", llm)
    agent = build_simple_agent(
        LLMProvider(name="test", provider_kind="openai", default_model="test-model"),
        [],
        skill_registry=skills,
        episodic_recall=recall,
        longterm_memory=longterm,
        tenant_id="tenant-a",
        context_token_budget=100,
    )

    await agent.ainvoke({"messages": messages, "iteration": 0, "max_iter": 1})

    sent = llm.await_args.args[1]
    assert any("[context trimmed:" in message.get("content", "") for message in sent)
    assert any(message.get("content") == current_query for message in sent)
    assert sent[0]["content"] == "[Skill: summary] Current summary guidance"
    assert recall.recall.await_args.kwargs["query"] == current_query
    assert longterm.retrieve_semantic.await_args.kwargs["query"] == current_query
