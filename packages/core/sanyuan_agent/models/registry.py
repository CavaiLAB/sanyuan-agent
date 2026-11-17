"""LLM provider registry and async completion wrapper."""

import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import litellm


@dataclass(frozen=True, slots=True)
class LLMProvider:
    """Configuration for a single LLM backend."""

    name: str
    provider_kind: str
    api_base: str = ""
    api_key_env: str = ""
    default_model: str = ""
    # Ordered alternative model ids tried (in order) when the primary model
    # hits a *transient* failure (rate-limit / connection / timeout / 5xx).
    # Empty = single-attempt, today's behaviour.
    fallback_models: tuple[str, ...] = ()


class LLMProviderRegistry:
    """In-memory registry for LLM providers."""

    def __init__(self) -> None:
        self._providers: dict[str, LLMProvider] = {}

    def register(self, provider: LLMProvider) -> None:
        if provider.name in self._providers:
            raise ValueError(
                f"Provider {provider.name!r} is already registered"
            )
        self._providers[provider.name] = provider

    def get(self, name: str) -> LLMProvider:
        try:
            return self._providers[name]
        except KeyError as exc:
            raise KeyError(f"Provider {name!r} not found") from exc

    def list(self) -> list[str]:
        return sorted(self._providers.keys())

    def __len__(self) -> int:
        return len(self._providers)

    def __iter__(self) -> Iterator[LLMProvider]:
        return iter(self._providers.values())

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LLMProviderRegistry":
        """Build a registry from a pydantic/dict config mapping."""
        registry = cls()
        for key, value in config.items():
            if isinstance(value, LLMProvider):
                registry.register(value)
            elif isinstance(value, dict):
                registry.register(LLMProvider(name=key, **value))
            else:
                raise TypeError(f"Invalid provider config for {key!r}")
        return registry


async def acompletion(
    provider_name: str,
    messages: list[dict[str, str]],
    registry: LLMProviderRegistry,
    **kwargs: Any,
) -> Any:
    """Async completion via the named provider."""
    provider = registry.get(provider_name)

    api_key = os.environ.get(provider.api_key_env) if provider.api_key_env else None
    model = kwargs.pop("model", provider.default_model)
    if not model:
        raise ValueError(f"No model specified and provider {provider.name!r} has no default_model")

    # litellm expects model in format "provider/model" or just "model"
    return await litellm.acompletion(
        model=model,
        messages=messages,
        api_key=api_key,
        api_base=provider.api_base or None,
        **kwargs,
    )
