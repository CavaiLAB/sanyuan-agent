"""_call_llm streaming branch: delta forwarding gate + response reassembly.

The gate rules (see ``_stream_completion``):
- a tool_call delta permanently stops emission (tool-turn narration must not
  double-render in the live body);
- the first ``hold_chars`` of content buffer before emitting, so short
  narration-then-tools turns emit nothing at all;
- everything still accumulates into the returned message, so the terminal
  final_answer equals the deltas' concatenation plus whatever was held back.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from neuroagent.agents.simple import _call_llm
from neuroagent.models import LLMProvider

pytestmark = pytest.mark.asyncio

_PROVIDER = LLMProvider(name="t", provider_kind="openai", default_model="m-1")


def _chunk(content=None, tool_calls=None, usage=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    choices = [SimpleNamespace(delta=delta)] if (content or tool_calls) else []
    return SimpleNamespace(choices=choices, usage=usage)


def _tc(index=0, tc_id=None, name=None, arguments=None):
    fn = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=tc_id, function=fn)


def _fake_stream(chunks):
    async def acompletion(**kwargs):
        assert kwargs.get("stream") is True
        assert kwargs.get("stream_options") == {"include_usage": True}

        async def gen():
            for c in chunks:
                yield c

        return gen()

    return acompletion


async def test_final_answer_streams_past_hold_window():
    chunks = [
        _chunk(content="abcd"),
        _chunk(content="efgh"),
        _chunk(content="ij"),
        _chunk(usage=SimpleNamespace(prompt_tokens=5, completion_tokens=7)),
    ]
    got: list[str] = []
    with patch("litellm.acompletion", new=_fake_stream(chunks)):
        resp = await _call_llm(
            _PROVIDER, [], [], on_content_delta=got.append, hold_chars=6
        )
    # First two pieces buffer until the 6-char hold is crossed, then flow live.
    assert got == ["abcdefgh", "ij"]
    assert resp.choices[0].message.content == "abcdefghij"
    assert resp.choices[0].message.tool_calls is None
    assert resp.usage.prompt_tokens == 5
    assert resp.usage.completion_tokens == 7


async def test_short_final_answer_flushes_once_at_stream_end():
    chunks = [_chunk(content="ok")]
    got: list[str] = []
    with patch("litellm.acompletion", new=_fake_stream(chunks)):
        resp = await _call_llm(
            _PROVIDER, [], [], on_content_delta=got.append, hold_chars=160
        )
    assert got == ["ok"]
    assert resp.choices[0].message.content == "ok"


async def test_narration_then_tool_call_emits_nothing():
    chunks = [
        _chunk(content="我先查一下资料"),
        _chunk(tool_calls=[_tc(0, "tc-1", "web_search", '{"q":')]),
        _chunk(tool_calls=[_tc(0, None, None, '"iclaw"}')]),
    ]
    got: list[str] = []
    with patch("litellm.acompletion", new=_fake_stream(chunks)):
        resp = await _call_llm(
            _PROVIDER, [], [], on_content_delta=got.append, hold_chars=160
        )
    assert got == []
    # Narration is preserved on the message (it becomes the `step` frame)…
    assert resp.choices[0].message.content == "我先查一下资料"
    # …and fragmented tool-call arguments reassemble.
    tcs = resp.choices[0].message.tool_calls
    assert len(tcs) == 1
    assert tcs[0].id == "tc-1"
    assert tcs[0].function.name == "web_search"
    assert tcs[0].function.arguments == '{"q":"iclaw"}'


async def test_tool_call_after_flush_stops_further_emission():
    chunks = [
        _chunk(content="long narration that crosses the hold window"),
        _chunk(tool_calls=[_tc(0, "tc-1", "web_search", "{}")]),
        _chunk(content="post-tool content"),
    ]
    got: list[str] = []
    with patch("litellm.acompletion", new=_fake_stream(chunks)):
        resp = await _call_llm(
            _PROVIDER, [], [], on_content_delta=got.append, hold_chars=8
        )
    assert got == ["long narration that crosses the hold window"]
    assert "post-tool content" in resp.choices[0].message.content


async def test_hold_zero_streams_immediately():
    chunks = [_chunk(content="a"), _chunk(content="b")]
    got: list[str] = []
    with patch("litellm.acompletion", new=_fake_stream(chunks)):
        await _call_llm(_PROVIDER, [], [], on_content_delta=got.append, hold_chars=0)
    assert got == ["a", "b"]


async def test_no_callback_keeps_plain_call():
    async def plain_acompletion(**kwargs):
        assert "stream" not in kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="plain", tool_calls=None, role="assistant"
                    )
                )
            ],
            usage=None,
        )

    with patch("litellm.acompletion", new=plain_acompletion):
        resp = await _call_llm(_PROVIDER, [], [])
    assert resp.choices[0].message.content == "plain"
