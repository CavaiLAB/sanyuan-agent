"""Tests for LLM provider registry and acompletion wrapper."""

from unittest.mock import AsyncMock, patch

import pytest
from neuroagent.models import LLMProvider, LLMProviderRegistry, acompletion


class TestLLMProviderRegistry:
    def test_register_and_get(self):
        reg = LLMProviderRegistry()
        p = LLMProvider(name="openai", provider_kind="openai")
        reg.register(p)
        assert reg.get("openai") == p

    def test_list_sorted(self):
        reg = LLMProviderRegistry()
        reg.register(LLMProvider(name="z", provider_kind="x"))
        reg.register(LLMProvider(name="a", provider_kind="x"))
        assert reg.list() == ["a", "z"]

    def test_duplicate_raises(self):
        reg = LLMProviderRegistry()
        reg.register(LLMProvider(name="dup", provider_kind="x"))
        with pytest.raises(ValueError, match="already registered"):
            reg.register(LLMProvider(name="dup", provider_kind="x"))

    def test_get_missing_raises(self):
        reg = LLMProviderRegistry()
        with pytest.raises(KeyError, match="not found"):
            reg.get("missing")

    def test_len_and_iter(self):
        reg = LLMProviderRegistry()
        reg.register(LLMProvider(name="a", provider_kind="x"))
        reg.register(LLMProvider(name="b", provider_kind="x"))
        assert len(reg) == 2
        assert {p.name for p in reg} == {"a", "b"}

    def test_from_config_dict(self):
        reg = LLMProviderRegistry.from_config({
            "openai": {
                "provider_kind": "openai",
                "api_base": "https://api.openai.com/v1",
                "api_key_env": "OPENAI_API_KEY",
                "default_model": "gpt-4o",
            }
        })
        p = reg.get("openai")
        assert p.provider_kind == "openai"
        assert p.api_base == "https://api.openai.com/v1"
        assert p.api_key_env == "OPENAI_API_KEY"
        assert p.default_model == "gpt-4o"

    def test_from_config_provider_object(self):
        p = LLMProvider(name="anthropic", provider_kind="anthropic")
        reg = LLMProviderRegistry.from_config({"anthropic": p})
        assert reg.get("anthropic") == p

    def test_from_config_invalid_type_raises(self):
        with pytest.raises(TypeError):
            LLMProviderRegistry.from_config({"bad": 123})


class TestAcompletion:
    @pytest.mark.asyncio
    async def test_calls_litellm_with_correct_args(self):
        reg = LLMProviderRegistry()
        reg.register(LLMProvider(
            name="test",
            provider_kind="openai",
            api_base="https://test.example",
            api_key_env="TEST_KEY",
            default_model="gpt-4o",
        ))

        mock_response = AsyncMock()
        with patch("neuroagent.models.registry.litellm.acompletion", new=mock_response):
            with patch.dict("os.environ", {"TEST_KEY": "secret123"}):
                await acompletion(
                    "test",
                    messages=[{"role": "user", "content": "hi"}],
                    registry=reg,
                )

        mock_response.assert_awaited_once()
        call_kwargs = mock_response.await_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["messages"] == [{"role": "user", "content": "hi"}]
        assert call_kwargs["api_key"] == "secret123"
        assert call_kwargs["api_base"] == "https://test.example"

    @pytest.mark.asyncio
    async def test_acompletion_override_model(self):
        reg = LLMProviderRegistry()
        reg.register(LLMProvider(
            name="test",
            provider_kind="openai",
            default_model="gpt-4o",
        ))

        mock_response = AsyncMock()
        with patch("neuroagent.models.registry.litellm.acompletion", new=mock_response):
            await acompletion(
                "test",
                messages=[],
                registry=reg,
                model="gpt-3.5-turbo",
            )

        assert mock_response.await_args.kwargs["model"] == "gpt-3.5-turbo"

    @pytest.mark.asyncio
    async def test_acompletion_no_default_model_raises(self):
        reg = LLMProviderRegistry()
        reg.register(LLMProvider(name="test", provider_kind="openai"))

        with pytest.raises(ValueError, match="No model"):
            await acompletion("test", messages=[], registry=reg)

    @pytest.mark.asyncio
    async def test_acompletion_missing_provider_raises(self):
        reg = LLMProviderRegistry()
        with pytest.raises(KeyError, match="not found"):
            await acompletion("missing", messages=[], registry=reg)
