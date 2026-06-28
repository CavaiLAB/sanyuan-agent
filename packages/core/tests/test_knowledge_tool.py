"""knowledge_search tool / provider: agent-callable retrieval over the RAG store.

The knowledge core already retrieves and injects context; this pins the *tool*
that lets the model search the base on demand — schema, metadata, clamping, and
end-to-end execution through the registry.
"""

from __future__ import annotations

import json

import pytest
from neuroagent.knowledge import InMemoryKnowledgeBase, ingest_document
from neuroagent.tools.knowledge import KnowledgeBaseToolProvider
from neuroagent.tools.provider import register_provider
from neuroagent.tools.registry import ToolRegistry


class _FakeEmbedder:
    """Maps known phrases to fixed vectors; unknown text → zero vector."""

    def __init__(self, table: dict[str, list[float]], dim: int = 3) -> None:
        self._table = table
        self._dim = dim

    async def embed(self, text: str) -> list[float]:
        return self._table.get(text, [0.0] * self._dim)


async def _seed_kb() -> tuple[InMemoryKnowledgeBase, _FakeEmbedder]:
    embedder = _FakeEmbedder(
        {
            "cats": [1.0, 0.0, 0.0],
            "dogs": [0.0, 1.0, 0.0],
            "feline pets": [1.0, 0.0, 0.0],
        }
    )
    kb = InMemoryKnowledgeBase()
    await ingest_document(kb, "col1", "cats", embedder, doc_id="d1")
    await ingest_document(kb, "col1", "dogs", embedder, doc_id="d2")
    return kb, embedder


class TestSpec:
    async def test_single_read_only_spec_with_metadata(self) -> None:
        kb, embedder = await _seed_kb()
        (spec,) = KnowledgeBaseToolProvider(kb, embedder, collection_id="col1").provide()
        assert spec.name == "knowledge_search"
        assert spec.category == "knowledge"
        assert spec.capabilities == ("knowledge", "search", "rag")
        assert spec.side_effect == "read"
        assert spec.policy_hint is None  # retrieval is not gated by default

    async def test_schema_non_degenerate(self) -> None:
        kb, embedder = await _seed_kb()
        (spec,) = KnowledgeBaseToolProvider(kb, embedder, collection_id="col1").provide()
        props = spec.schema["parameters"]["properties"]
        assert set(props) == {"query", "top_k"}
        assert spec.schema["parameters"]["required"] == ["query"]
        assert props["top_k"] == {
            "type": "integer",
            "description": "Maximum number of passages to return.",
            "default": 5,
        }


class TestExecution:
    async def test_search_returns_ranked_results(self) -> None:
        kb, embedder = await _seed_kb()
        registry = ToolRegistry()
        names = register_provider(
            registry, KnowledgeBaseToolProvider(kb, embedder, collection_id="col1")
        )
        assert names == ["knowledge_search"]
        result = await registry.execute("knowledge_search", json.dumps({"query": "feline pets"}))
        assert result["collection"] == "col1"
        assert result["count"] == 2
        # "cats" (cosine 1.0) outranks "dogs" (cosine 0.0).
        assert result["results"][0]["text"] == "cats"
        assert result["results"][0]["score"] == pytest.approx(1.0)
        assert result["results"][0]["metadata"]["doc_id"] == "d1"

    async def test_top_k_clamped_to_at_least_one(self) -> None:
        kb, embedder = await _seed_kb()
        registry = ToolRegistry()
        register_provider(
            registry, KnowledgeBaseToolProvider(kb, embedder, collection_id="col1")
        )
        result = await registry.execute(
            "knowledge_search", json.dumps({"query": "feline pets", "top_k": 0})
        )
        assert result["count"] == 1


class TestConstruction:
    async def test_empty_collection_id_rejected(self) -> None:
        kb, embedder = await _seed_kb()
        with pytest.raises(ValueError, match="collection_id"):
            KnowledgeBaseToolProvider(kb, embedder, collection_id="")

    async def test_non_positive_max_top_k_rejected(self) -> None:
        kb, embedder = await _seed_kb()
        with pytest.raises(ValueError, match="max_top_k"):
            KnowledgeBaseToolProvider(kb, embedder, collection_id="col1", max_top_k=0)
