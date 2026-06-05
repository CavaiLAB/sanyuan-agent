"""Tests for LiteLLM token-usage tracking — tracker, callback, OTel metric."""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock

import pytest
from neuroagent.costs.callback import (
    _infer_provider,
    _on_success,
    install_usage_callback,
)
from neuroagent.costs.tracker import (
    LLMCallUsage,
    UsageTracker,
)

# -- Fixtures -------------------------------------------------------------


@pytest.fixture
def flushed() -> list[list[LLMCallUsage]]:
    """Collects flushed batches."""
    return []


@pytest.fixture
def tracker(flushed: list[list[LLMCallUsage]]) -> UsageTracker:
    async def _flush(batch: list[LLMCallUsage]) -> None:
        flushed.append(batch)

    return UsageTracker(flush_callback=_flush, flush_interval=0.05)


def _make_record(**overrides: Any) -> LLMCallUsage:
    defaults = dict(
        run_id="r-001",
        model="openai/gpt-4",
        provider="openai",
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        timestamp=time.time(),
        tenant_id="t-001",
    )
    defaults.update(overrides)
    return LLMCallUsage(**defaults)


# -- Tracker enqueue ------------------------------------------------------


class TestTrackerEnqueue:
    def test_enqueue_adds_record(self, tracker: UsageTracker) -> None:
        record = _make_record()
        tracker.enqueue(record)
        assert tracker.pending_count == 1

    def test_enqueue_thread_safe(self, tracker: UsageTracker) -> None:
        for i in range(100):
            tracker.enqueue(_make_record(total_tokens=i))
        assert tracker.pending_count == 100

    def test_queue_full_warns_but_does_not_block(
        self, tracker: UsageTracker
    ) -> None:
        small_tracker = UsageTracker(flush_callback=None, max_pending=5)
        for i in range(20):
            small_tracker.enqueue(_make_record(total_tokens=i))
        assert small_tracker.pending_count == 20  # nothing dropped


# -- Tracker flush --------------------------------------------------------


class TestTrackerFlush:
    async def test_flush_invokes_callback(
        self, tracker: UsageTracker, flushed: list[list[LLMCallUsage]]
    ) -> None:
        tracker.enqueue(_make_record())
        tracker.enqueue(_make_record(model="anthropic/claude-3"))
        count = await tracker.flush()
        assert count == 2
        assert len(flushed) == 1
        assert len(flushed[0]) == 2

    async def test_flush_empty_is_noop(
        self, tracker: UsageTracker, flushed: list[list[LLMCallUsage]]
    ) -> None:
        count = await tracker.flush()
        assert count == 0
        assert len(flushed) == 0

    async def test_re_enqueue_on_flush_failure(self) -> None:
        call_count = 0

        async def _bad_flush(batch: list[LLMCallUsage]) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("db down")

        t = UsageTracker(flush_callback=_bad_flush)
        t.enqueue(_make_record())
        await t.flush()
        # First flush failed — record re-enqueued
        assert t.pending_count == 1
        # Second flush succeeds
        await t.flush()
        assert t.pending_count == 0


# -- Tracker lifecycle ----------------------------------------------------


class TestTrackerLifecycle:
    async def test_start_stop(self, tracker: UsageTracker) -> None:
        tracker.start()
        assert tracker._running
        await tracker.stop()
        assert not tracker._running

    async def test_background_flush(
        self, flushed: list[list[LLMCallUsage]]
    ) -> None:
        async def _flush(batch: list[LLMCallUsage]) -> None:
            flushed.append(batch)

        t = UsageTracker(flush_callback=_flush, flush_interval=0.05)
        t.enqueue(_make_record())
        t.start()
        await asyncio.sleep(0.15)
        await t.stop()
        assert len(flushed) >= 1


# -- Callback -------------------------------------------------------------


class TestCallback:
    def test_infer_provider(self) -> None:
        assert _infer_provider("openai/gpt-4") == "openai"
        assert _infer_provider("anthropic/claude-3") == "anthropic"
        assert _infer_provider("gpt-4") == "unknown"

    def test_callback_enqueues_token_usage(
        self, tracker: UsageTracker
    ) -> None:
        install_usage_callback(tracker)
        # Simulate LiteLLM callback invocation
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 200
        mock_usage.completion_tokens = 100
        mock_usage.total_tokens = 300

        mock_response = MagicMock()
        mock_response.model = "openai/gpt-4o"
        mock_response.usage = mock_usage

        _on_success(
            kwargs={"model": "openai/gpt-4o", "run_id": "r-1", "tenant_id": "t-1"},
            completion_response=mock_response,
            start_time=None,
            end_time=None,
        )
        assert tracker.pending_count == 1
        record = tracker._pending[0]
        assert record.input_tokens == 200
        assert record.output_tokens == 100
        assert record.total_tokens == 300

    def test_callback_handles_missing_usage(
        self, tracker: UsageTracker
    ) -> None:
        install_usage_callback(tracker)
        mock_response = MagicMock()
        mock_response.model = "openai/gpt-4o"
        mock_response.usage = None

        _on_success(
            kwargs={"model": "openai/gpt-4o"},
            completion_response=mock_response,
            start_time=None,
            end_time=None,
        )
        assert tracker.pending_count == 1
        # Peek at the record
        record = tracker._pending[0]
        assert record.input_tokens == 0
        assert record.output_tokens == 0
        assert record.total_tokens == 0

    def test_callback_no_tracker_is_noop(self) -> None:
        """If no tracker is installed, callback must not crash."""
        from neuroagent.costs import callback as cb
        old = cb._tracker
        cb._tracker = None
        try:
            _on_success(
                kwargs={},
                completion_response=MagicMock(),
                start_time=None,
                end_time=None,
            )
        finally:
            cb._tracker = old


# -- Multi-model aggregation ----------------------------------------------


class TestMultiModelAggregation:
    async def test_multiple_models_single_run(
        self, tracker: UsageTracker, flushed: list[list[LLMCallUsage]]
    ) -> None:
        tracker.enqueue(_make_record(model="openai/gpt-4", total_tokens=100))
        tracker.enqueue(_make_record(model="anthropic/claude-3", total_tokens=200))
        tracker.enqueue(_make_record(model="openai/gpt-4", total_tokens=50))

        await tracker.flush()
        batch = flushed[0]
        assert len(batch) == 3
        total = sum(r.total_tokens for r in batch)
        assert total == 350
