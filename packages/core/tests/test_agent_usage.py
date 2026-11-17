"""Tests for token-usage capture in the reasoning agent (fix: real reflow)."""

from __future__ import annotations

from types import SimpleNamespace

from sanyuan_agent.agents.simple import _get_usage, _merge_usage


class TestGetUsage:
    def test_extracts_from_litellm_response(self) -> None:
        resp = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=120, completion_tokens=34))
        assert _get_usage(resp) == {"promptTokens": 120, "completionTokens": 34}

    def test_missing_usage_is_zeros(self) -> None:
        assert _get_usage(SimpleNamespace()) == {"promptTokens": 0, "completionTokens": 0}

    def test_none_fields_coerced_to_zero(self) -> None:
        resp = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=None, completion_tokens=None))
        assert _get_usage(resp) == {"promptTokens": 0, "completionTokens": 0}


class TestMergeUsage:
    def test_sums_turns(self) -> None:
        a = {"promptTokens": 10, "completionTokens": 5}
        b = {"promptTokens": 7, "completionTokens": 3}
        assert _merge_usage(a, b) == {"promptTokens": 17, "completionTokens": 8}

    def test_none_safe(self) -> None:
        assert _merge_usage(None, {"promptTokens": 4, "completionTokens": 1}) == {
            "promptTokens": 4,
            "completionTokens": 1,
        }
        assert _merge_usage({"promptTokens": 2, "completionTokens": 2}, None) == {
            "promptTokens": 2,
            "completionTokens": 2,
        }

    def test_both_empty(self) -> None:
        assert _merge_usage({}, {}) == {"promptTokens": 0, "completionTokens": 0}
