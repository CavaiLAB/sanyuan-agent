"""Knowledge-base retrieval tool (``knowledge_search``) over the RAG store.

The knowledge core (:mod:`neuroagent.knowledge`) already chunks, embeds, and
retrieves, and can *inject* context into a prompt
(``knowledge_context_message``). What was missing is an **agent-callable** tool
so the model can search the knowledge base on demand mid-reasoning rather than
relying solely on pre-injected context. This provider binds an injected
:class:`~neuroagent.knowledge.store.KnowledgeBase` + ``Embedder`` and a default
collection, exposing one read-only ``knowledge_search`` tool.

Boundaries mirror the rest of the tool layer: the embedder is injected (a fake
in tests), the harness imports nothing from ``neuroagent_app`` and reads no
secret, and the result is a plain JSON-serializable dict (id / text / score /
metadata) — never a store handle.
"""

from __future__ import annotations

from typing import Any

from neuroagent.knowledge.ingestion import Embedder
from neuroagent.knowledge.store import KnowledgeBase
from neuroagent.tools.provider import ToolSpec, spec_from_callable

_DEFAULT_TOP_K = 5
_MAX_TOP_K = 20


class KnowledgeBaseToolProvider:
    """Yields the read-only ``knowledge_search`` tool bound to a KB + embedder.

    *collection_id* scopes every search to one collection (the agent does not
    choose the collection — that's a deployment concern). *top_k* requested by
    the model is clamped to ``[1, max_top_k]`` so a tool call can't ask for an
    unbounded result set.

    Selectable through the provider path: reference the provider in
    ``AgentSpec.providers`` and enable ``"knowledge_search"`` via
    ``AgentSpec.tools`` / ``tool_sets``.
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        embedder: Embedder,
        *,
        collection_id: str,
        max_top_k: int = _MAX_TOP_K,
    ) -> None:
        if not collection_id:
            raise ValueError("KnowledgeBaseToolProvider requires a non-empty collection_id")
        if max_top_k < 1:
            raise ValueError(f"max_top_k must be >= 1, got {max_top_k}")
        self._kb = kb
        self._embedder = embedder
        self._collection_id = collection_id
        self._max_top_k = max_top_k

    def provide(self) -> list[ToolSpec]:
        kb = self._kb
        embedder = self._embedder
        collection_id = self._collection_id
        max_top_k = self._max_top_k

        async def knowledge_search(query: str, top_k: int = _DEFAULT_TOP_K) -> dict[str, Any]:
            """Search the knowledge base for passages relevant to a query.

            Args:
                query: Natural-language search query.
                top_k: Maximum number of passages to return.
            """
            k = max(1, min(top_k, max_top_k))
            embedding = await embedder.embed(query)
            pairs = kb.search(collection_id, embedding, top_k=k)
            return {
                "collection": collection_id,
                "count": len(pairs),
                "results": [
                    {
                        "id": chunk.id,
                        "text": chunk.text,
                        "score": score,
                        "metadata": chunk.metadata,
                    }
                    for chunk, score in pairs
                ],
            }

        return [
            spec_from_callable(
                knowledge_search,
                category="knowledge",
                capabilities=("knowledge", "search", "rag"),
                side_effect="read",
            )
        ]
