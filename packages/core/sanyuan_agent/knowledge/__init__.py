"""KnowledgeBase / RAG — explicit document collections, distinct from memory.

Pipeline: ``chunk_text`` splits a document into overlapping chunks;
``ingest_document`` embeds and stores them in a :class:`KnowledgeBase`;
``retrieve`` finds the chunks most relevant to a query; and
``knowledge_context_message`` formats them into a system message to inject into
an agent's prompt (RAG). Embedding is injected (any ``IEmbeddingProvider``);
the in-memory store here is for tests/dev — a pgvector-backed store is a
follow-up.
"""

from .chunker import Chunk, chunk_text
from .ingestion import ingest_document, knowledge_context_message, retrieve
from .store import InMemoryKnowledgeBase, KnowledgeBase, KnowledgeChunk

__all__ = [
    "Chunk",
    "InMemoryKnowledgeBase",
    "KnowledgeBase",
    "KnowledgeChunk",
    "chunk_text",
    "ingest_document",
    "knowledge_context_message",
    "retrieve",
]
