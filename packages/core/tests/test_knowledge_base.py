"""KnowledgeBase / RAG foundation: chunk → embed → store → retrieve → inject.

The explicit knowledge base (user/org document collections) is distinct from
the memory layers (episodic / semantic): it ingests documents, chunks them,
embeds the chunks, and retrieves relevant context to inject into an agent's
prompt. This is the pure-backend core — embedding is injected (IEmbeddingProvider),
persistence is an in-memory store here (a pgvector-backed store is a follow-up).
"""

from __future__ import annotations

import math

import pytest
from sanyuan_agent.knowledge import (
    InMemoryKnowledgeBase,
    KnowledgeChunk,
    chunk_text,
    ingest_document,
    knowledge_context_message,
    retrieve,
)

# ---------------------------------------------------------------------------
# DocumentChunker
# ---------------------------------------------------------------------------


class TestChunkText:
    def test_empty_text_yields_no_chunks(self) -> None:
        assert chunk_text("") == []
        assert chunk_text("   \n  ") == []

    def test_short_text_is_single_chunk(self) -> None:
        chunks = chunk_text("hello world", chunk_size=100, overlap=10)
        assert len(chunks) == 1
        assert chunks[0].text == "hello world"
        assert chunks[0].index == 0
        assert chunks[0].start == 0
        assert chunks[0].end == 11

    def test_long_text_splits_with_overlap(self) -> None:
        text = "".join(str(i % 10) for i in range(250))  # 250 chars
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        # step = 80 → chunks start at 0, 80, 160; the 160:250 window reaches the
        # end (90 chars) so there is no further chunk.
        assert [c.start for c in chunks] == [0, 80, 160]
        assert [c.end for c in chunks] == [100, 180, 250]
        assert chunks[0].text == text[0:100]
        assert chunks[1].text == text[80:180]
        # consecutive chunks share `overlap` chars
        assert chunks[0].text[-20:] == chunks[1].text[:20]
        assert chunks[1].text[-20:] == chunks[2].text[:20]
        # each chunk.text matches its offsets
        for c in chunks:
            assert c.text == text[c.start : c.end]

    def test_indices_are_sequential(self) -> None:
        chunks = chunk_text("a" * 300, chunk_size=100, overlap=0)
        assert [c.index for c in chunks] == [0, 1, 2]

    def test_overlap_must_be_less_than_chunk_size(self) -> None:
        with pytest.raises(ValueError):
            chunk_text("abc", chunk_size=10, overlap=10)

    def test_cjk_text_chunks_by_codepoint(self) -> None:
        text = "数据" * 100  # 200 codepoints
        chunks = chunk_text(text, chunk_size=50, overlap=0)
        assert len(chunks) == 4
        assert "".join(c.text for c in chunks) == text


# ---------------------------------------------------------------------------
# Fake embedder (deterministic, controlled vectors)
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    """Maps known phrases to fixed vectors; unknown text → zero vector."""

    def __init__(self, table: dict[str, list[float]], dim: int = 3) -> None:
        self._table = table
        self._dim = dim

    async def embed(self, text: str) -> list[float]:
        return self._table.get(text, [0.0] * self._dim)

    @property
    def dimension(self) -> int:
        return self._dim


def _unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


# ---------------------------------------------------------------------------
# InMemoryKnowledgeBase
# ---------------------------------------------------------------------------


class TestInMemoryKnowledgeBase:
    def _chunk(self, cid: str, col: str, text: str, emb: list[float]) -> KnowledgeChunk:
        return KnowledgeChunk(id=cid, collection_id=col, text=text, embedding=emb)

    def test_search_ranks_by_cosine_similarity(self) -> None:
        kb = InMemoryKnowledgeBase()
        kb.add(
            [
                self._chunk("a", "col1", "apples", _unit([1.0, 0.0, 0.0])),
                self._chunk("b", "col1", "bananas", _unit([0.0, 1.0, 0.0])),
                self._chunk("c", "col1", "cherries", _unit([0.0, 0.0, 1.0])),
            ]
        )
        results = kb.search("col1", _unit([0.9, 0.1, 0.0]), top_k=2)
        assert [chunk.id for chunk, _ in results] == ["a", "b"]
        assert results[0][1] > results[1][1]  # scores descending

    def test_top_k_limits_results(self) -> None:
        kb = InMemoryKnowledgeBase()
        kb.add([self._chunk(str(i), "c", f"t{i}", _unit([float(i), 1.0, 0.0])) for i in range(5)])
        assert len(kb.search("c", _unit([1.0, 1.0, 0.0]), top_k=3)) == 3

    def test_collection_isolation(self) -> None:
        kb = InMemoryKnowledgeBase()
        kb.add([self._chunk("a", "col1", "x", _unit([1.0, 0.0, 0.0]))])
        kb.add([self._chunk("b", "col2", "y", _unit([1.0, 0.0, 0.0]))])
        results = kb.search("col1", _unit([1.0, 0.0, 0.0]), top_k=10)
        assert [chunk.id for chunk, _ in results] == ["a"]

    def test_delete_collection(self) -> None:
        kb = InMemoryKnowledgeBase()
        kb.add([self._chunk("a", "col1", "x", _unit([1.0, 0.0, 0.0]))])
        kb.delete_collection("col1")
        assert kb.search("col1", _unit([1.0, 0.0, 0.0]), top_k=10) == []


# ---------------------------------------------------------------------------
# Ingestion + retrieval + injection
# ---------------------------------------------------------------------------


class TestIngestionAndRetrieval:
    async def test_ingest_document_chunks_embeds_stores(self) -> None:
        text = "alpha section. " * 20  # ~300 chars → multiple chunks
        embedder = _FakeEmbedder({}, dim=3)  # all-zero ok for store test
        kb = InMemoryKnowledgeBase()

        ids = await ingest_document(
            kb, "col1", text, embedder, doc_id="doc1", chunk_size=100, overlap=10
        )

        assert len(ids) >= 2
        assert all(i.startswith("doc1#") for i in ids)

    async def test_retrieve_returns_relevant_chunk(self) -> None:
        kb = InMemoryKnowledgeBase()
        embedder = _FakeEmbedder(
            {
                "the cat sat": _unit([1.0, 0.0, 0.0]),
                "the dog ran": _unit([0.0, 1.0, 0.0]),
                "cat": _unit([1.0, 0.0, 0.0]),
            }
        )
        await ingest_document(
            kb, "c", "the cat sat", embedder, doc_id="d1", chunk_size=100, overlap=0
        )
        await ingest_document(
            kb, "c", "the dog ran", embedder, doc_id="d2", chunk_size=100, overlap=0
        )

        hits = await retrieve(kb, "c", "cat", embedder, top_k=1)

        assert len(hits) == 1
        assert hits[0].text == "the cat sat"

    async def test_knowledge_context_message_formats_hits(self) -> None:
        chunks = [
            KnowledgeChunk(id="1", collection_id="c", text="Fact A.", embedding=[0.0]),
            KnowledgeChunk(id="2", collection_id="c", text="Fact B.", embedding=[0.0]),
        ]
        msg = knowledge_context_message(chunks)
        assert msg is not None
        assert msg["role"] == "system"
        assert "Fact A." in msg["content"]
        assert "Fact B." in msg["content"]

    def test_knowledge_context_message_empty_is_none(self) -> None:
        assert knowledge_context_message([]) is None
