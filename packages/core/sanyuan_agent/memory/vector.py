"""Vector-based episodic recall using sentence-transformers embeddings."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from sanyuan_agent.memory.abc import IEpisodicStore
from sanyuan_agent.memory.recall import EpisodicRecall

logger = logging.getLogger(__name__)


class VectorRecall(EpisodicRecall):
    """Retrieve episodes via dense vector similarity.

    Wraps an :class:`IEpisodicStore` and lazily loads a
    ``sentence_transformers.SentenceTransformer`` model to embed episode
    text and queries.  Results are ranked by cosine similarity.

    Falls back gracefully if *sentence-transformers* is not installed.
    """

    def __init__(
        self,
        store: IEpisodicStore,
        *,
        embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 5,
        similarity_threshold: float = 0.3,
    ) -> None:
        super().__init__(store=store, top_k=top_k, strategy="keyword")
        self._embedder_name = embedder_name
        self._similarity_threshold = similarity_threshold
        self._model: Any = None
        self._embeddings: dict[str, np.ndarray] = {}

    def _load_model(self) -> Any:
        """Lazily load the SentenceTransformer model."""
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import (  # type: ignore[import-not-found]
                SentenceTransformer,
            )
        except ImportError:
            logger.warning("sentence-transformers not installed; vector recall unavailable")
            return None
        self._model = SentenceTransformer(self._embedder_name)
        return self._model

    async def recall(
        self,
        tenant_id: str,
        query: str,
        conversation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the most relevant episodes via cosine similarity.

        If the embedder is unavailable, falls back to the parent keyword
        (TF-IDF) strategy.
        """
        if not tenant_id or not query.strip():
            return []

        model = self._load_model()
        if model is None:
            return await super().recall(tenant_id, query, conversation_id)

        # Fetch recent episodes and index any new ones
        episodes = await self._store.recent_episodes(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            limit=100,
        )
        if not episodes:
            return []

        self._ensure_embeddings(episodes, model)

        # Embed the query
        query_vec = model.encode(query, normalize_embeddings=True)
        if isinstance(query_vec, np.ndarray) and query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        # Compute cosine similarities
        scored: list[tuple[float, dict[str, Any]]] = []
        for ep in episodes:
            ep_id = ep.get("id", "")
            vec = self._embeddings.get(ep_id)
            if vec is None:
                continue
            sim = float(np.dot(query_vec, vec.T).flatten()[0])
            if sim >= self._similarity_threshold:
                scored.append((sim, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[: self._top_k]]

    def _ensure_embeddings(
        self,
        episodes: list[dict[str, Any]],
        model: Any,
    ) -> None:
        """Batch-embed episodes that are not yet indexed."""
        to_embed: list[tuple[str, str]] = []
        for ep in episodes:
            ep_id = ep.get("id", "")
            if ep_id and ep_id not in self._embeddings:
                text = f"{ep['user_query']} {ep['final_answer']}"
                to_embed.append((ep_id, text))

        if not to_embed:
            return

        texts = [t for _, t in to_embed]
        vectors = model.encode(texts, normalize_embeddings=True)
        for (ep_id, _), vec in zip(to_embed, vectors, strict=True):
            self._embeddings[ep_id] = np.asarray(vec)

    def format_prompt_block(
        self,
        episodes: list[dict[str, Any]],
        *,
        max_chars: int = 2000,
    ) -> str:
        return super().format_prompt_block(episodes, max_chars=max_chars)
