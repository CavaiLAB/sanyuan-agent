"""Model fallback chain: when the primary model is unavailable, try the next.

``LLMProvider.fallback_models`` declares an ordered list of alternative model
ids. ``_call_llm`` tries the primary (default_model), and on a *transient*
litellm error (rate-limit / connection / timeout / 5xx) falls through to each
fallback in turn. Empty fallback list = today's single-attempt behaviour.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import litellm
import pytest
from sanyuan_agent.agents.simple import _call_llm
from sanyuan_agent.config import AgentSpec, load_agent_from_spec
from sanyuan_agent.models import LLMProvider, LLMProviderRegistry
from sanyuan_agent.tools.registry import ToolRegistry


def _ok(model: str = "primary") -> Any:
    return SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )


def _rate_limit() -> litellm.RateLimitError:
    return litellm.RateLimitError("rate limited", llm_provider="openai", model="m")


def _provider(default: str, *fallbacks: str) -> LLMProvider:
    return LLMProvider(
        name="higress",
        provider_kind="openai_compatible",
        api_base="http://higress.test/v1",
        api_key_env="",
        default_model=default,
        fallback_models=tuple(fallbacks),
    )


MSGS = [{"role": "user", "content": "hi"}]


class TestCallLlmFallback:
    async def test_primary_success_no_fallback_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        acompletion = AsyncMock(return_value=_ok("primary"))
        monkeypatch.setattr(litellm, "acompletion", acompletion)

        resp = await _call_llm(_provider("primary", "backup"), MSGS, [])

        assert resp.model == "primary"
        assert acompletion.await_count == 1
        assert acompletion.await_args.kwargs["model"] == "primary"

    async def test_primary_fails_fallback_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        acompletion = AsyncMock(side_effect=[_rate_limit(), _ok("backup")])
        monkeypatch.setattr(litellm, "acompletion", acompletion)

        resp = await _call_llm(_provider("primary", "backup"), MSGS, [])

        assert resp.model == "backup"
        assert acompletion.await_count == 2
        models = [c.kwargs["model"] for c in acompletion.await_args_list]
        assert models == ["primary", "backup"]

    async def test_fallbacks_tried_in_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        acompletion = AsyncMock(side_effect=[_rate_limit(), _rate_limit(), _ok("third")])
        monkeypatch.setattr(litellm, "acompletion", acompletion)

        resp = await _call_llm(_provider("primary", "second", "third"), MSGS, [])

        assert resp.model == "third"
        models = [c.kwargs["model"] for c in acompletion.await_args_list]
        assert models == ["primary", "second", "third"]

    async def test_all_models_fail_raises_last(self, monkeypatch: pytest.MonkeyPatch) -> None:
        acompletion = AsyncMock(side_effect=[_rate_limit(), _rate_limit()])
        monkeypatch.setattr(litellm, "acompletion", acompletion)

        with pytest.raises(litellm.RateLimitError):
            await _call_llm(_provider("primary", "backup"), MSGS, [])
        assert acompletion.await_count == 2

    async def test_non_transient_error_does_not_fall_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A bad-request / auth error won't be fixed by another model — bubble immediately.
        bad = litellm.BadRequestError("bad", model="m", llm_provider="openai")
        acompletion = AsyncMock(side_effect=bad)
        monkeypatch.setattr(litellm, "acompletion", acompletion)

        with pytest.raises(litellm.BadRequestError):
            await _call_llm(_provider("primary", "backup"), MSGS, [])
        assert acompletion.await_count == 1  # no fallback attempt

    async def test_empty_fallback_single_attempt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        acompletion = AsyncMock(side_effect=_rate_limit())
        monkeypatch.setattr(litellm, "acompletion", acompletion)

        with pytest.raises(litellm.RateLimitError):
            await _call_llm(_provider("primary"), MSGS, [])
        assert acompletion.await_count == 1


class TestAgentSpecFallback:
    def test_spec_has_fallback_models_default_empty(self) -> None:
        spec = AgentSpec(name="a", system_prompt="p", llm_provider="x")
        assert spec.fallback_models == []

    def test_loader_wires_fallback_models_onto_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reg = LLMProviderRegistry()
        reg.register(
            LLMProvider(
                name="higress",
                provider_kind="openai_compatible",
                default_model="openai/deepseek-v4-flash",
            )
        )
        captured: dict[str, Any] = {}
        import sanyuan_agent.config.loader as loader_mod

        real_build = loader_mod.build_simple_agent

        def spy_build(llm_provider, tools, **kw):  # type: ignore[no-untyped-def]
            captured["provider"] = llm_provider
            return real_build(llm_provider, tools, **kw)

        monkeypatch.setattr(loader_mod, "build_simple_agent", spy_build)

        spec = AgentSpec(
            name="a",
            system_prompt="p",
            llm_provider="higress",
            fallback_models=["openai/deepseek-v4-pro", "openai/kimi-k2.6"],
        )
        load_agent_from_spec(spec, tool_registry=ToolRegistry(), llm_registry=reg)

        assert captured["provider"].fallback_models == (
            "openai/deepseek-v4-pro",
            "openai/kimi-k2.6",
        )
