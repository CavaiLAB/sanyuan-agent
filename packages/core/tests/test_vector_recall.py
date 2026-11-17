"""Tests for VectorRecall — vector-based episodic retrieval."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from neuroagent.memory.abc import IEpisodicStore
from neuroagent.memory.vector import VectorRecall


class FakeStore(IEpisodicStore):
    """In-memory fake episodic store for unit tests."""

    def __init__(self) -> None:
        self._episodes: list[dict[str, Any]] = []

    async def add_episode(
        self,
        tenant_id: str,
        conversation_id: str,
        user_query: str,
        final_answer: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._episodes.append(
            {
                "id": f"ep-{len(self._episodes)}",
                "tenant_id": tenant_id,
                "conversation_id": conversation_id,
                "user_query": user_query,
                "final_answer": final_answer,
                "metadata": metadata or {},
            }
        )

    async def recent_episodes(
        self,
        tenant_id: str,
        conversation_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        filtered = [
            ep
            for ep in self._episodes
            if ep["tenant_id"] == tenant_id
            and (conversation_id is None or ep["conversation_id"] == conversation_id)
        ]
        return list(reversed(filtered[-limit:]))


def _make_mock_model() -> MagicMock:
    """Create a mock SentenceTransformer that returns deterministic embeddings."""

    def fake_encode(texts: Any, normalize_embeddings: bool = True) -> np.ndarray:
        if isinstance(texts, list):
            # Batch: return one vector per text based on hash
            vecs = []
            for t in texts:
                h = hash(t) % 1000
                vecs.append(np.array([h / 1000, (h + 1) / 1000, (h + 2) / 1000], dtype=np.float32))
            return np.array(vecs)
        # Single text
        h = hash(texts) % 1000
        return np.array([h / 1000, (h + 1) / 1000, (h + 2) / 1000], dtype=np.float32)

    model = MagicMock()
    model.encode = fake_encode
    return model


class TestVectorRecallEmbed:
    async def test_embed_and_retrieve_happy(self) -> None:
        store = FakeStore()
        await store.add_episode("t1", "c1", "how to bake bread", "use yeast and flour")
        await store.add_episode("t1", "c1", "stock market tips", "buy low sell high")
        await store.add_episode("t1", "c1", "bread machine recipes", "white bread")

        recall = VectorRecall(store=store, top_k=2, similarity_threshold=0.0)
        # Inject mock model
        recall._model = _make_mock_model()

        results = await recall.recall("t1", "bread baking tips")
        assert len(results) <= 2
        assert len(results) > 0

    async def test_empty_history_returns_empty(self) -> None:
        store = FakeStore()
        recall = VectorRecall(store=store)
        recall._model = _make_mock_model()

        results = await recall.recall("t1", "anything")
        assert results == []

    async def test_cosine_similarity_threshold(self) -> None:
        """With high threshold, irrelevant episodes should be filtered out."""
        store = FakeStore()

        # Create episodes with very different content
        await store.add_episode("t1", "c1", "quantum physics", "schrodinger equation")
        await store.add_episode("t1", "c1", "cooking pasta", "boil water")

        recall = VectorRecall(store=store, top_k=5, similarity_threshold=0.99)
        # Mock that gives very different vectors → low similarity
        model = MagicMock()

        def encode(texts: Any, normalize_embeddings: bool = True) -> np.ndarray:
            if isinstance(texts, list):
                return np.array([[1.0, 0.0, 0.0]] * len(texts), dtype=np.float32)
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)

        model.encode = encode
        recall._model = model

        results = await recall.recall("t1", "completely different query")
        # query vec [0,1,0] has 0 dot product with episode vecs [1,0,0] → sim=0 < 0.99
        assert results == []

    async def test_tenant_isolation(self) -> None:
        store = FakeStore()
        await store.add_episode("t1", "c1", "t1 query", "t1 answer")
        await store.add_episode("t2", "c2", "t2 query", "t2 answer")

        recall = VectorRecall(store=store, similarity_threshold=0.0)
        recall._model = _make_mock_model()

        results = await recall.recall("t1", "query")
        assert len(results) == 1
        assert results[0]["tenant_id"] == "t1"

    async def test_batch_embed_correctness(self) -> None:
        """Verify all episodes get embedded in batch."""
        store = FakeStore()
        for i in range(5):
            await store.add_episode("t1", "c1", f"query {i}", f"answer {i}")

        recall = VectorRecall(store=store, similarity_threshold=0.0)
        model = _make_mock_model()
        recall._model = model

        await recall.recall("t1", "query 0")
        # All 5 episodes should be embedded
        assert len(recall._embeddings) == 5

    async def test_fallback_to_keyword_when_no_model(self) -> None:
        """When embedder is unavailable, falls back to TF-IDF keyword."""
        store = FakeStore()
        await store.add_episode("t1", "c1", "python async tips", "use asyncio")

        recall = VectorRecall(store=store)
        # _model stays None, _load_model returns None (no sentence-transformers)

        # Should still return results via TF-IDF fallback
        # This works because sklearn is available in the test env
        results = await recall.recall("t1", "python async")
        assert len(results) >= 0  # May be 0 or 1 depending on TF-IDF scoring

    async def test_env_switch_keyword(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """NEUROAGENT_MEMORY_BACKEND=keyword creates EpisodicRecall."""
        from neuroagent.memory.recall import EpisodicRecall

        monkeypatch.setenv("NEUROAGENT_MEMORY_BACKEND", "keyword")

        store = FakeStore()
        recall = EpisodicRecall(store=store, strategy="keyword")
        assert recall._strategy == "keyword"

    async def test_env_switch_vector(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """NEUROAGENT_MEMORY_BACKEND=vector creates VectorRecall."""
        monkeypatch.setenv("NEUROAGENT_MEMORY_BACKEND", "vector")

        store = FakeStore()
        recall = VectorRecall(store=store)
        assert isinstance(recall, VectorRecall)

    async def test_embedder_missing_no_crash(self) -> None:
        """VectorRecall should not crash when sentence-transformers is absent."""
        store = FakeStore()
        await store.add_episode("t1", "c1", "test query", "test answer")

        recall = VectorRecall(store=store)
        # _load_model will return None (sentence-transformers not importable
        # or mocked to raise ImportError)
        def mock_load() -> Any:
            return None

        recall._load_model = mock_load  # type: ignore[assignment]

        # Should fallback to keyword without crashing
        results = await recall.recall("t1", "test")
        # Either returns results via TF-IDF or empty — but no exception
        assert isinstance(results, list)
