"""Tests for EpisodicRecall — TF-IDF keyword + recent fallback."""

from __future__ import annotations

from typing import Any

from sanyuan_agent.memory.abc import IEpisodicStore
from sanyuan_agent.memory.recall import EpisodicRecall


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


class TestEpisodicRecallKeyword:
    async def test_recall_returns_relevant_episodes(self) -> None:
        store = FakeStore()
        await store.add_episode("t1", "c1", "how to bake bread", "use yeast and flour", {})
        await store.add_episode("t1", "c1", "best hiking trails", "try Yosemite", {})
        await store.add_episode("t1", "c1", "bread machine recipes", "white bread", {})

        recall = EpisodicRecall(store=store, top_k=2, strategy="keyword")
        results = await recall.recall("t1", "bread baking tips")

        assert len(results) == 2
        # The bread-related episodes should rank highest
        queries = [r["user_query"] for r in results]
        assert "how to bake bread" in queries
        assert "bread machine recipes" in queries

    async def test_recall_excludes_irrelevant(self) -> None:
        store = FakeStore()
        await store.add_episode("t1", "c1", "how to bake bread", "use yeast", {})
        await store.add_episode("t1", "c1", "stock market trends", "buy low sell high", {})

        recall = EpisodicRecall(store=store, top_k=1, strategy="keyword")
        results = await recall.recall("t1", "bread")

        assert len(results) == 1
        assert results[0]["user_query"] == "how to bake bread"

    async def test_empty_history_returns_empty(self) -> None:
        store = FakeStore()
        recall = EpisodicRecall(store=store, top_k=3, strategy="keyword")
        results = await recall.recall("t1", "anything")
        assert results == []

    async def test_tenant_isolation(self) -> None:
        store = FakeStore()
        await store.add_episode("t1", "c1", "t1 query", "t1 answer", {})
        await store.add_episode("t2", "c2", "t2 query", "t2 answer", {})

        recall = EpisodicRecall(store=store, top_k=5, strategy="keyword")
        results = await recall.recall("t1", "query")
        assert len(results) == 1
        assert results[0]["tenant_id"] == "t1"

    async def test_top_k_limits_results(self) -> None:
        store = FakeStore()
        for i in range(10):
            await store.add_episode("t1", "c1", f"query {i}", f"answer {i}", {})

        recall = EpisodicRecall(store=store, top_k=3, strategy="keyword")
        results = await recall.recall("t1", "query")
        assert len(results) == 3

    async def test_long_query_handled(self) -> None:
        store = FakeStore()
        await store.add_episode("t1", "c1", "python async", "use asyncio", {})

        recall = EpisodicRecall(store=store, top_k=1, strategy="keyword")
        long_query = "python " * 50 + "async programming patterns"
        results = await recall.recall("t1", long_query)
        assert len(results) == 1

    async def test_conversation_filter(self) -> None:
        store = FakeStore()
        await store.add_episode("t1", "c1", "c1 query", "c1 answer", {})
        await store.add_episode("t1", "c2", "c2 query", "c2 answer", {})

        recall = EpisodicRecall(store=store, top_k=5, strategy="keyword")
        results = await recall.recall("t1", "query", conversation_id="c1")
        assert len(results) == 1
        assert results[0]["conversation_id"] == "c1"


class TestEpisodicRecallRecent:
    async def test_recent_strategy_falls_back(self) -> None:
        store = FakeStore()
        await store.add_episode("t1", "c1", "first", "a1", {})
        await store.add_episode("t1", "c1", "second", "a2", {})

        recall = EpisodicRecall(store=store, top_k=2, strategy="recent")
        results = await recall.recall("t1", "irrelevant")

        # recent strategy ignores query and returns newest episodes
        assert len(results) == 2
        assert results[0]["user_query"] == "second"


class TestFormatPromptBlock:
    async def test_format_prompt_block_basic(self) -> None:
        store = FakeStore()
        recall = EpisodicRecall(store=store, top_k=2)
        episodes = [
            {"user_query": "q1", "final_answer": "a1"},
            {"user_query": "q2", "final_answer": "a2"},
        ]
        block = recall.format_prompt_block(episodes)
        assert "相关历史:" in block
        assert "q1" in block
        assert "a2" in block

    async def test_format_prompt_block_empty(self) -> None:
        store = FakeStore()
        recall = EpisodicRecall(store=store)
        assert recall.format_prompt_block([]) == ""

    async def test_format_prompt_block_truncates(self) -> None:
        store = FakeStore()
        recall = EpisodicRecall(store=store)
        episodes = [
            {"user_query": "x" * 500, "final_answer": "y" * 500},
        ]
        block = recall.format_prompt_block(episodes, max_chars=100)
        assert len(block) <= 100
        assert block.endswith("...")


class TestAgentIntegration:
    async def test_maybe_inject_recall_with_recall(self) -> None:
        from sanyuan_agent.agents.simple import _maybe_inject_recall

        store = FakeStore()
        await store.add_episode("t1", "c1", "python async tips", "use asyncio", {})
        recall = EpisodicRecall(store=store, top_k=1, strategy="keyword")

        messages = [{"role": "user", "content": "how to do async in python"}]
        result = await _maybe_inject_recall(messages, recall, "t1", "c1")

        assert len(result) == 2  # system msg + original user msg
        assert result[0]["role"] == "system"
        assert "相关历史" in result[0]["content"]

    async def test_maybe_inject_recall_no_op_without_recall(self) -> None:
        from sanyuan_agent.agents.simple import _maybe_inject_recall

        messages = [{"role": "user", "content": "hello"}]
        result = await _maybe_inject_recall(messages, None, "t1", "c1")
        assert result == messages

    async def test_maybe_inject_recall_no_op_without_tenant(self) -> None:
        from sanyuan_agent.agents.simple import _maybe_inject_recall

        store = FakeStore()
        recall = EpisodicRecall(store=store)
        messages = [{"role": "user", "content": "hello"}]
        result = await _maybe_inject_recall(messages, recall, "", "c1")
        assert result == messages
