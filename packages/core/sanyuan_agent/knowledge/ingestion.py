"""Document ingestion + retrieval + prompt injection for the knowledge base."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .chunker import chunk_text
from .store import KnowledgeBase, KnowledgeChunk


@runtime_checkable
class Embedder(Protocol):
    """Subset of ``IEmbeddingProvider`` this module needs."""

    async def embed(self, text: str) -> list[float]: ...


async def ingest_document(
    kb: KnowledgeBase,
    collection_id: str,
    text: str,
    embedder: Embedder,
    *,
    doc_id: str,
    chunk_size: int = 1024,
    overlap: int = 128,
    metadata: dict[str, Any] | None = None,
) -> list[str]:
    """Chunk *text*, embed each chunk, store them, and return the chunk ids.

    Chunk ids are ``{doc_id}#{index}`` so a document's chunks are addressable
    and a re-ingest of the same doc_id produces stable ids.
    """
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    stored: list[KnowledgeChunk] = []
    for chunk in chunks:
        embedding = await embedder.embed(chunk.text)
        chunk_id = f"{doc_id}#{chunk.index}"
        stored.append(
            KnowledgeChunk(
                id=chunk_id,
                collection_id=collection_id,
                text=chunk.text,
                embedding=embedding,
                metadata={"doc_id": doc_id, "index": chunk.index, **(metadata or {})},
            )
        )
    kb.add(stored)
    return [c.id for c in stored]


async def retrieve(
    kb: KnowledgeBase,
    collection_id: str,
    query: str,
    embedder: Embedder,
    *,
    top_k: int = 5,
) -> list[KnowledgeChunk]:
    """Embed *query* and return the *top_k* most relevant chunks from the collection."""
    query_embedding = await embedder.embed(query)
    return [chunk for chunk, _score in kb.search(collection_id, query_embedding, top_k=top_k)]


def knowledge_context_message(chunks: list[KnowledgeChunk]) -> dict[str, str] | None:
    """Format retrieved chunks into a system message for RAG injection, or None.

    Mirrors the profile/memory injection pattern: a single system message the
    caller prepends to the agent's messages.
    """
    if not chunks:
        return None
    lines = [f"- {chunk.text.strip()}" for chunk in chunks]
    return {
        "role": "system",
        "content": (
            "Knowledge base context (use when relevant; cite nothing you can't support):\n"
            + "\n".join(lines)
        ),
    }
