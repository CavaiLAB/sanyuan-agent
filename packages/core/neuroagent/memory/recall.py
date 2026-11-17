"""Episodic recall — keyword TF-IDF and recent-strategy retrieval."""

from __future__ import annotations

import logging
from typing import Any

from neuroagent.memory.abc import IEpisodicStore

logger = logging.getLogger(__name__)

_MAX_PROMPT_CHARS = 2000


class EpisodicRecall:
    """Retrieve relevant past episodes for prompt injection.

    *store* must implement :class:`IEpisodicStore`.
    *strategy* controls the ranking method:

    - ``"keyword"`` — TF-IDF cosine similarity over
      ``user_query + " " + final_answer``.
    - ``"recent"``  — fall back to :meth:`IEpisodicStore.recent_episodes`.

    *top_k* limits the number of returned episodes.
    """

    def __init__(
        self,
        store: IEpisodicStore,
        *,
        top_k: int = 5,
        strategy: str = "keyword",
    ) -> None:
        self._store = store
        self._top_k = top_k
        self._strategy = strategy

    async def recall(
        self,
        tenant_id: str,
        query: str,
        conversation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the most relevant episodes for *query*.

        If *conversation_id* is provided the search is scoped to that
        conversation; otherwise all tenant episodes are considered.
        """
        if not tenant_id or not query.strip():
            return []

        if self._strategy == "recent":
            return await self._store.recent_episodes(
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                limit=self._top_k,
            )

        # keyword strategy — TF-IDF over all tenant (or conversation) episodes
        episodes = await self._store.recent_episodes(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            limit=100,  # fetch a reasonable window for TF-IDF ranking
        )
        if not episodes:
            return []

        return _rank_by_tfidf(query, episodes, top_k=self._top_k)

    def format_prompt_block(
        self,
        episodes: list[dict[str, Any]],
        *,
        max_chars: int = _MAX_PROMPT_CHARS,
    ) -> str:
        """Format episodes as a single prompt block, truncated to *max_chars*."""
        if not episodes:
            return ""

        lines: list[str] = ["相关历史:"]
        for ep in episodes:
            lines.append(
                f"- Query: {ep['user_query']}\n  Answer: {ep['final_answer']}"
            )

        block = "\n".join(lines)
        if len(block) > max_chars:
            # Truncate and add ellipsis
            truncated = block[: max_chars - 3]
            # Try to cut at a newline to avoid mid-line breakage
            last_nl = truncated.rfind("\n")
            if last_nl > max_chars * 0.5:
                truncated = truncated[:last_nl]
            block = truncated + "..."
        return block


def _rank_by_tfidf(
    query: str,
    episodes: list[dict[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    """Rank *episodes* by TF-IDF cosine similarity to *query*."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        logger.warning("sklearn not available for TF-IDF: %s", exc)
        return episodes[:top_k]

    corpus = [f"{ep['user_query']} {ep['final_answer']}" for ep in episodes]
    # query is appended as the last document so the vectoriser sees the
    # same vocabulary
    all_docs = [*corpus, query]

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform(all_docs)
    except ValueError:
        # Empty vocabulary (e.g. all docs are empty or only stop-words)
        return episodes[:top_k]

    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import-untyped]

    query_vec = tfidf_matrix[-1]
    episode_vecs = tfidf_matrix[:-1]
    scores = cosine_similarity(query_vec, episode_vecs).flatten()

    scored = list(zip(scores, episodes, strict=True))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ep for _score, ep in scored[:top_k]]
