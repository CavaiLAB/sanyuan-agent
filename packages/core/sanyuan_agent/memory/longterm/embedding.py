"""Embedding provider abstraction — OpenAI or local sentence-transformers."""

from __future__ import annotations

import logging
import os
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class IEmbeddingProvider(Protocol):
    """Protocol for text embedding providers."""

    async def embed(self, text: str) -> list[float]:
        """Return a dense vector for *text*."""
        ...

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...


class OpenAIEmbeddingProvider:
    """Embed via OpenAI-compatible API (default: text-embedding-3-small, 1536d)."""

    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        api_base: str | None = None,
        dimension: int = 1536,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._api_base = api_base
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> list[float]:
        import httpx

        api_key = self._api_key or os.environ.get("OPENAI_API_KEY", "")
        api_base = self._api_base or "https://api.openai.com/v1"

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{api_base}/embeddings",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"input": text, "model": self._model},
            )
            resp.raise_for_status()
            data = resp.json()
            embedding: list[float] = data["data"][0]["embedding"]
            return embedding


class LocalEmbeddingProvider:
    """Embed via sentence-transformers (default: all-MiniLM-L6-v2, 384d)."""

    def __init__(
        self,
        *,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimension: int = 384,
    ) -> None:
        self._model_name = model_name
        self._dimension = dimension
        self._model: Any = None

    @property
    def dimension(self) -> int:
        return self._dimension

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

        self._model = SentenceTransformer(self._model_name)
        return self._model

    async def embed(self, text: str) -> list[float]:
        model = self._load_model()
        vec = model.encode(text, normalize_embeddings=True)
        result: list[float] = vec.tolist()
        return result


def create_embedding_provider(
    *,
    provider: str | None = None,
) -> IEmbeddingProvider:
    """Factory: select embedding provider from env or explicit argument.

    Env ``SANYUAN_AGENT_EMBEDDING_PROVIDER``: ``"openai"`` (default) or ``"local"``.
    """
    name = provider or os.environ.get("SANYUAN_AGENT_EMBEDDING_PROVIDER", "openai")
    name = name.lower().strip()

    if name == "local":
        logger.info("Using local embedding provider")
        return LocalEmbeddingProvider()
    logger.info("Using OpenAI embedding provider")
    return OpenAIEmbeddingProvider()
