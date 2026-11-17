"""Per-call run/tenant context for the LiteLLM usage callback.

The callback previously relied on module-level defaults set via
``set_run_context`` — racy under concurrent runs. Context now travels with
each call through ``litellm`` metadata.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from sanyuan_agent.agents.simple import _call_llm
from sanyuan_agent.costs import callback as cb
from sanyuan_agent.costs.tracker import UsageTracker
from sanyuan_agent.models import LLMProvider


def _response(prompt: int = 10, completion: int = 5) -> Any:
    return SimpleNamespace(
        model="openai/deepseek-v4-flash",
        usage=SimpleNamespace(
            prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion
        ),
    )


@pytest.fixture()
def tracker(monkeypatch: pytest.MonkeyPatch) -> UsageTracker:
    t = UsageTracker()
    monkeypatch.setattr(cb, "_tracker", t)
    monkeypatch.setattr(cb, "_default_run_id", "default-run")
    monkeypatch.setattr(cb, "_default_tenant_id", "default-tenant")
    return t


class TestCallbackMetadataContext:
    def test_prefers_litellm_metadata(self, tracker: UsageTracker) -> None:
        kwargs = {
            "model": "openai/deepseek-v4-flash",
            "litellm_params": {"metadata": {"run_id": "run-meta", "tenant_id": "tenant-meta"}},
        }

        cb._on_success(kwargs, _response(), None, None)

        record = tracker._pending[0]
        assert record.run_id == "run-meta"
        assert record.tenant_id == "tenant-meta"

    def test_falls_back_to_defaults_without_metadata(self, tracker: UsageTracker) -> None:
        cb._on_success({"model": "openai/m"}, _response(), None, None)

        record = tracker._pending[0]
        assert record.run_id == "default-run"
        assert record.tenant_id == "default-tenant"

    def test_partial_metadata_mixes_with_defaults(self, tracker: UsageTracker) -> None:
        kwargs = {
            "model": "openai/m",
            "litellm_params": {"metadata": {"run_id": "run-meta"}},
        }

        cb._on_success(kwargs, _response(), None, None)

        record = tracker._pending[0]
        assert record.run_id == "run-meta"
        assert record.tenant_id == "default-tenant"


class TestCallLlmMetadata:
    async def test_passes_metadata_to_litellm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import litellm

        acompletion = AsyncMock(return_value=_response())
        monkeypatch.setattr(litellm, "acompletion", acompletion)
        provider = LLMProvider(
            name="m",
            provider_kind="openai_compatible",
            api_base="http://higress.test/v1",
            api_key_env="",
            default_model="openai/m",
        )

        await _call_llm(
            provider,
            [{"role": "user", "content": "hi"}],
            [],
            metadata={"run_id": "r1", "tenant_id": "t1"},
        )

        assert acompletion.await_args.kwargs["metadata"] == {
            "run_id": "r1",
            "tenant_id": "t1",
        }

    async def test_omits_metadata_when_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import litellm

        acompletion = AsyncMock(return_value=_response())
        monkeypatch.setattr(litellm, "acompletion", acompletion)
        provider = LLMProvider(
            name="m",
            provider_kind="openai_compatible",
            api_base="http://higress.test/v1",
            api_key_env="",
            default_model="openai/m",
        )

        await _call_llm(provider, [{"role": "user", "content": "hi"}], [])

        assert "metadata" not in acompletion.await_args.kwargs
