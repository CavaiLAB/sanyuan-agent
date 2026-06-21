"""Tests for web_fetch tool — httpx GET with SSRF guard."""

import httpx
import pytest
import respx
from httpx import Response
from neuroagent.tools.registry import ToolRegistry
from neuroagent.tools.web_fetch import build_web_fetch_tool


class TestWebFetchRegistry:
    def test_tool_registered(self):
        reg = ToolRegistry()
        build_web_fetch_tool(reg)
        assert reg.has_tool("web_fetch")

    def test_schema_has_url_param(self):
        reg = ToolRegistry()
        build_web_fetch_tool(reg)
        schema = reg.get_schema("web_fetch")
        params = schema["parameters"]["properties"]
        assert "url" in params
        required = schema["parameters"]["required"]
        assert "url" in required


@pytest.mark.asyncio
class TestWebFetchHappyPath:
    @respx.mock
    async def test_fetch_html(self):
        reg = ToolRegistry()
        tool = build_web_fetch_tool(reg)

        html = """<!DOCTYPE html>
<html><head><title>Hello</title></head>
<body><h1>Title</h1><p>paragraph text</p></body></html>
"""
        route = respx.get("https://example.com/page").mock(
            return_value=Response(200, text=html, headers={"content-type": "text/html"})
        )

        result = await tool("https://example.com/page")

        assert route.called
        assert result["status_code"] == 200
        assert result["url"] == "https://example.com/page"
        assert "Title" in result["title"]
        assert "paragraph text" in result["content"]

    @respx.mock
    async def test_fetch_plain_text(self):
        reg = ToolRegistry()
        tool = build_web_fetch_tool(reg)

        route = respx.get("https://example.com/raw.txt").mock(
            return_value=Response(200, text="raw content", headers={"content-type": "text/plain"})
        )

        result = await tool("https://example.com/raw.txt")

        assert route.called
        assert result["status_code"] == 200
        assert "raw content" in result["content"]


@pytest.mark.asyncio
class TestWebFetchSSRF:
    async def test_localhost_ipv4_rejected(self):
        reg = ToolRegistry()
        tool = build_web_fetch_tool(reg)

        with pytest.raises(ValueError, match="SSRF"):
            await tool("http://127.0.0.1/secret")

    async def test_localhost_name_rejected(self):
        reg = ToolRegistry()
        tool = build_web_fetch_tool(reg)

        with pytest.raises(ValueError, match="SSRF"):
            await tool("http://localhost/admin")

    async def test_192_168_rejected(self):
        reg = ToolRegistry()
        tool = build_web_fetch_tool(reg)

        with pytest.raises(ValueError, match="SSRF"):
            await tool("http://192.168.1.1/")

    async def test_10_x_rejected(self):
        reg = ToolRegistry()
        tool = build_web_fetch_tool(reg)

        with pytest.raises(ValueError, match="SSRF"):
            await tool("http://10.0.0.1/")

    async def test_172_16_rejected(self):
        reg = ToolRegistry()
        tool = build_web_fetch_tool(reg)

        with pytest.raises(ValueError, match="SSRF"):
            await tool("http://172.16.0.1/")

    async def test_public_ip_allowed(self):
        reg = ToolRegistry()
        tool = build_web_fetch_tool(reg)

        with respx.mock:
            route = respx.get("https://8.8.8.8/").mock(
                return_value=Response(
                    200, text="ok", headers={"content-type": "text/plain"}
                )
            )
            result = await tool("https://8.8.8.8/")
            assert route.called
            assert result["status_code"] == 200


@pytest.mark.asyncio
class TestWebFetchGuards:
    @respx.mock
    async def test_non_text_content_type_rejected(self):
        reg = ToolRegistry()
        tool = build_web_fetch_tool(reg)

        respx.get("https://example.com/file.pdf").mock(
            return_value=Response(
                200,
                content=b"%PDF",
                headers={"content-type": "application/pdf"},
            )
        )

        with pytest.raises(ValueError, match="Non-text content-type"):
            await tool("https://example.com/file.pdf")

    @respx.mock
    async def test_timeout(self):
        reg = ToolRegistry()
        tool = build_web_fetch_tool(reg, timeout=1)

        respx.get("https://example.com/slow").mock(
            side_effect=httpx.TimeoutException("timed out")
        )

        with pytest.raises(httpx.TimeoutException):
            await tool("https://example.com/slow")

    @respx.mock
    async def test_http_error_propagated(self):
        reg = ToolRegistry()
        tool = build_web_fetch_tool(reg)

        respx.get("https://example.com/404").mock(
            return_value=Response(404, text="not found", headers={"content-type": "text/html"})
        )

        with pytest.raises(httpx.HTTPStatusError):
            await tool("https://example.com/404")


@pytest.mark.asyncio
class TestWebFetchBlockSignal:
    """A 200 anti-bot page is flagged `blocked` (raise_for_status can't see it)."""

    @respx.mock
    async def test_soft_block_page_flagged(self):
        reg = ToolRegistry()
        tool = build_web_fetch_tool(reg)
        respx.get("https://bj.zu.anjuke.com/x").mock(
            return_value=Response(
                200,
                text="访问已被拦截\n请完成安全验证后重试",
                headers={"content-type": "text/plain"},
            )
        )
        result = await tool("https://bj.zu.anjuke.com/x")
        assert result["status_code"] == 200
        assert result["blocked"] is True
        assert result["block_reason"]

    @respx.mock
    async def test_normal_page_not_flagged(self):
        reg = ToolRegistry()
        tool = build_web_fetch_tool(reg)
        respx.get("https://example.com/post").mock(
            return_value=Response(
                200,
                text="北京二居室租金趋势\n2024 年安贞西里整租均价约 8500 元 环比小幅上涨",
                headers={"content-type": "text/plain"},
            )
        )
        result = await tool("https://example.com/post")
        assert result["blocked"] is False
        assert result["block_reason"] is None
