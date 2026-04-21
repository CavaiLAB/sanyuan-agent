"""Tests for web_search tool — DuckDuckGo + Brave fallback."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from neuroagent.tools.registry import ToolRegistry
from neuroagent.tools.web_search import SearchResult, build_web_search_tool


class TestWebSearchRegistry:
    def test_tool_registered(self):
        reg = ToolRegistry()
        build_web_search_tool(reg)
        assert reg.has_tool("web_search")

    def test_schema_has_params(self):
        reg = ToolRegistry()
        build_web_search_tool(reg)
        schema = reg.get_schema("web_search")
        params = schema["parameters"]["properties"]
        assert "query" in params
        assert "max_results" in params
        required = schema["parameters"]["required"]
        assert "query" in required
        assert "max_results" not in required


@pytest.mark.asyncio
class TestWebSearchDuckDuckGo:
    async def test_happy_path(self):
        reg = ToolRegistry()
        tool = build_web_search_tool(reg)

        mock_results = [
            SearchResult(title="T1", url="https://example.com/1", snippet="B1"),
            SearchResult(title="T2", url="https://example.com/2", snippet="B2"),
        ]

        with patch(
            "neuroagent.tools.web_search._ddg_search", autospec=True
        ) as mock_ddg:
            mock_ddg.return_value = mock_results
            results = await tool("python asyncio", max_results=5)

        assert len(results) == 2
        assert results[0].title == "T1"
        assert results[0].url == "https://example.com/1"
        assert results[0].snippet == "B1"

    async def test_max_results_capped(self):
        """DuckDuckGo backend caps at 10 inside _ddg_search."""
        reg = ToolRegistry()
        tool = build_web_search_tool(reg)

        with patch(
            "neuroagent.tools.web_search._ddg_search", autospec=True
        ) as mock_ddg:
            mock_ddg.return_value = []
            await tool("query", max_results=100)
            # _ddg_search is called with the raw max_results; capping happens inside
            assert mock_ddg.call_args[0][1] == 100

    async def test_empty_results(self):
        reg = ToolRegistry()
        tool = build_web_search_tool(reg)

        with patch(
            "neuroagent.tools.web_search._ddg_search", autospec=True
        ) as mock_ddg:
            mock_ddg.return_value = []
            results = await tool("xyz12345nonsense")
            assert results == []

    async def test_ddg_integration_mock(self):
        """Mock DDGS.text directly to verify translation layer."""
        reg = ToolRegistry()
        tool = build_web_search_tool(reg)

        ddg_items = [
            {"title": "A", "href": "http://a.com", "body": "body A"},
        ]

        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = ddg_items

        with patch(
            "duckduckgo_search.DDGS", return_value=mock_ddgs
        ):
            results = await tool("hello", max_results=1)

        assert len(results) == 1
        assert results[0].title == "A"
        assert results[0].url == "http://a.com"
        assert results[0].snippet == "body A"


@pytest.mark.asyncio
class TestWebSearchBrave:
    async def test_brave_happy_path(self, monkeypatch):
        monkeypatch.setenv("NEUROAGENT_WEB_SEARCH_API_KEY", "fake-key")
        reg = ToolRegistry()
        tool = build_web_search_tool(reg)

        mock_resp = MagicMock()
        mock_resp.json = MagicMock(return_value={
            "web": {
                "results": [
                    {"title": "Brave", "url": "https://brave.com", "description": "desc"}
                ]
            }
        })
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("httpx.AsyncClient", return_value=mock_client):
            results = await tool("brave", max_results=3)

        assert len(results) == 1
        assert results[0].title == "Brave"
        assert results[0].url == "https://brave.com"
        assert results[0].snippet == "desc"

    async def test_brave_empty_response(self, monkeypatch):
        monkeypatch.setenv("NEUROAGENT_WEB_SEARCH_API_KEY", "fake-key")
        reg = ToolRegistry()
        tool = build_web_search_tool(reg)

        mock_resp = MagicMock()
        mock_resp.json = MagicMock(return_value={"web": {"results": []}})
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("httpx.AsyncClient", return_value=mock_client):
            results = await tool("nonsense", max_results=5)

        assert results == []
