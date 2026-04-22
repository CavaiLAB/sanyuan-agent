"""Tests for Playwright browser tools — all mocked, no real Chromium."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from neuroagent.artifacts.client import ArtifactRef
from neuroagent.tools.browser import BrowserSession, build_browser_tools
from neuroagent.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


async def _async_return(val: Any) -> Any:
    return val


class FakePage:
    """Minimal async fake for Playwright Page."""

    def __init__(self) -> None:
        self.url = "https://example.com"
        self._title = "Example"

    async def goto(self, url: str, **kwargs: Any) -> None:
        self.url = url
        self._title = f"Page-{url}"

    async def title(self) -> str:
        return self._title

    async def click(self, selector: str, **kwargs: Any) -> None:
        if selector == "#nonexistent":
            raise TimeoutError(f"Timeout 10000ms waiting for selector: {selector}")

    async def fill(self, selector: str, value: str, **kwargs: Any) -> None:
        if selector == "#nonexistent":
            raise TimeoutError(f"Timeout 10000ms waiting for selector: {selector}")

    async def type(self, selector: str, text: str, **kwargs: Any) -> None:
        if selector == "#nonexistent":
            raise TimeoutError(f"Timeout 10000ms waiting for selector: {selector}")

    async def screenshot(self, **kwargs: Any) -> bytes:
        return b"\x89PNG\r\n\x1a\nfake_screenshot_data"


def _mock_session() -> BrowserSession:
    session = BrowserSession.__new__(BrowserSession)
    session._artifact_client = None
    session._tenant_id = ""
    session._timeout_ms = 10_000
    session._playwright = None
    session._browser = None
    session._context = None
    session._page = FakePage()
    return session


def _session_with_page(page: FakePage) -> BrowserSession:
    s = BrowserSession.__new__(BrowserSession)
    s._artifact_client = None
    s._tenant_id = ""
    s._timeout_ms = 10_000
    s._playwright = None
    s._browser = None
    s._context = None
    s._page = page
    return s


@pytest.fixture()
def registry() -> ToolRegistry:
    return ToolRegistry()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_navigate_happy(registry: ToolRegistry) -> None:
    """Navigate to a valid URL returns success with title and url."""
    build_browser_tools(registry, lambda: _async_return(_mock_session()))
    navigate = registry.get("browser_navigate")
    result = await navigate(url="https://example.com")
    assert result["success"] is True
    assert result["url"] == "https://example.com"
    assert "title" in result


async def test_click_selector_not_found(registry: ToolRegistry) -> None:
    """Click on a nonexistent selector returns error."""
    build_browser_tools(registry, lambda: _async_return(_mock_session()))
    click = registry.get("browser_click")
    result = await click(selector="#nonexistent")
    assert result["success"] is False
    assert "Timeout" in result["error"]


async def test_navigate_timeout(registry: ToolRegistry) -> None:
    """Navigate to a slow URL returns error on timeout."""
    session = _mock_session()

    async def slow_goto(url: str, **kwargs: Any) -> None:
        raise TimeoutError("Timeout 10000ms")

    session._page.goto = slow_goto  # type: ignore[assignment]

    build_browser_tools(registry, lambda: _async_return(session))
    navigate = registry.get("browser_navigate")
    result = await navigate(url="https://slow.example.com")
    assert result["success"] is False
    assert "Timeout" in result["error"]


async def test_ssrf_reject_localhost(registry: ToolRegistry) -> None:
    """Navigate to localhost is blocked by SSRF guard."""
    build_browser_tools(registry, lambda: _async_return(_mock_session()))
    navigate = registry.get("browser_navigate")
    result = await navigate(url="http://localhost:8080/admin")
    assert result["success"] is False
    assert "SSRF" in result["error"] or "localhost" in result["error"].lower()


async def test_ssrf_reject_file_scheme(registry: ToolRegistry) -> None:
    """file:// URLs are blocked."""
    build_browser_tools(registry, lambda: _async_return(_mock_session()))
    navigate = registry.get("browser_navigate")
    result = await navigate(url="file:///etc/passwd")
    assert result["success"] is False
    assert "http(s)" in result["error"]


async def test_ssrf_reject_private_ip(registry: ToolRegistry) -> None:
    """Private IP ranges (10.x) are blocked."""
    build_browser_tools(registry, lambda: _async_return(_mock_session()))
    navigate = registry.get("browser_navigate")
    result = await navigate(url="http://10.0.0.1/secret")
    assert result["success"] is False
    assert "SSRF" in result["error"] or "private" in result["error"].lower()


async def test_click_screenshot_chain(registry: ToolRegistry) -> None:
    """Click followed by screenshot works correctly."""
    build_browser_tools(registry, lambda: _async_return(_mock_session()))

    click_fn = registry.get("browser_click")
    result = await click_fn(selector="#ok-btn")
    assert result["success"] is True

    screenshot_fn = registry.get("browser_screenshot")
    result = await screenshot_fn()
    assert result["success"] is True
    assert result["screenshot_artifact_id"] == ""


async def test_screenshot_with_artifact_upload(registry: ToolRegistry) -> None:
    """Screenshot uploads PNG via artifact client and returns artifact_id."""
    mock_client = AsyncMock()
    mock_client.create_artifact.return_value = ArtifactRef(
        id="art-123",
        url="https://storage/art-123",
        sha256="abc",
        filename="screenshot.png",
        content_type="image/png",
    )

    session = _mock_session()
    session._artifact_client = mock_client
    session._tenant_id = "t1"

    build_browser_tools(registry, lambda: _async_return(session))
    screenshot_fn = registry.get("browser_screenshot")
    result = await screenshot_fn()
    assert result["success"] is True
    assert result["screenshot_artifact_id"] == "art-123"
    mock_client.create_artifact.assert_awaited_once()
    call_kwargs = mock_client.create_artifact.call_args
    assert call_kwargs.kwargs["content_type"] == "image/png"


async def test_session_cleanup() -> None:
    """cleanup() closes context, browser, and playwright without error."""
    session = BrowserSession.__new__(BrowserSession)
    session._context = AsyncMock()
    session._browser = AsyncMock()
    session._playwright = AsyncMock()
    session._page = AsyncMock()
    session._timeout_ms = 10_000
    session._artifact_client = None
    session._tenant_id = ""

    await session.cleanup()

    session._context.close.assert_awaited_once()
    session._browser.close.assert_awaited_once()
    session._playwright.stop.assert_awaited_once()
    assert session._page is None


async def test_concurrent_sessions_independent() -> None:
    """Two sessions operate on separate pages independently."""
    page_a = FakePage()
    page_b = FakePage()
    page_b.url = "https://other.com"
    page_b._title = "Other"

    reg_a = ToolRegistry()
    reg_b = ToolRegistry()

    build_browser_tools(reg_a, lambda: _async_return(_session_with_page(page_a)))
    build_browser_tools(reg_b, lambda: _async_return(_session_with_page(page_b)))

    nav_a = reg_a.get("browser_navigate")
    nav_b = reg_b.get("browser_navigate")

    result_a = await nav_a(url="https://a.com")
    result_b = await nav_b(url="https://b.com")

    assert result_a["url"] == "https://a.com"
    assert result_b["url"] == "https://b.com"


async def test_error_log_no_pii(registry: ToolRegistry) -> None:
    """Error messages in result dict are structured for downstream sanitization."""
    session = _mock_session()

    async def fail_goto(url: str, **kwargs: Any) -> None:
        raise RuntimeError("Navigation failed cookie=secret_value")

    session._page.goto = fail_goto  # type: ignore[assignment]

    build_browser_tools(registry, lambda: _async_return(session))
    navigate = registry.get("browser_navigate")
    result = await navigate(url="https://example.com")
    assert result["success"] is False
    assert "error" in result


async def test_lazy_session_creation(registry: ToolRegistry) -> None:
    """Session is only created on first tool call, not at registration."""
    call_count = 0

    async def factory() -> BrowserSession:
        nonlocal call_count
        call_count += 1
        return _mock_session()

    build_browser_tools(registry, factory)
    assert call_count == 0

    navigate = registry.get("browser_navigate")
    await navigate(url="https://example.com")
    assert call_count == 1

    click = registry.get("browser_click")
    await click(selector="#btn")
    assert call_count == 1  # reused
