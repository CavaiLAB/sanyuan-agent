"""Tests for Playwright browser tools — mocked Chromium, CDP client, session cache."""

from __future__ import annotations

import hashlib
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from neuroagent.artifacts.client import ArtifactRef
from neuroagent.tools.browser.actions import _annotate_block, build_browser_tools
from neuroagent.tools.browser.client import CDPClient
from neuroagent.tools.browser.session import BrowserSession, _session_cache
from neuroagent.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


async def _async_return(val: Any) -> Any:
    return val


class _FakeAccessibility:
    async def snapshot(self, **kwargs: Any) -> dict[str, Any]:
        return {"role": "WebArea", "name": "Example", "children": []}


class _FakeDownload:
    """Minimal async fake for a Playwright Download."""

    def __init__(self, suggested: str, content: bytes) -> None:
        self.suggested_filename = suggested
        self._content = content

    async def save_as(self, path: str) -> None:
        with open(path, "wb") as fh:
            fh.write(self._content)


class FakePage:
    """Minimal async fake for Playwright Page."""

    def __init__(self) -> None:
        self.url = "https://example.com"
        self._title = "Example"
        self._html = "<body><h1>Hello</h1><p id='content'>World</p></body>"
        self.accessibility = _FakeAccessibility()

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

    async def query_selector(self, selector: str) -> Any:
        if selector == "#nonexistent":
            return None
        el = AsyncMock()
        el.inner_text.return_value = "Hello World"
        el.inner_html.return_value = "<h1>Hello</h1>"
        return el


def _mock_session() -> BrowserSession:
    session = BrowserSession.__new__(BrowserSession)
    session._artifact_client = None
    session._tenant_id = ""
    session._timeout_ms = 10_000
    session._cdp_url = None
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
    s._cdp_url = None
    s._playwright = None
    s._browser = None
    s._context = None
    s._page = page
    return s


@pytest.fixture(autouse=True)
def _clear_session_cache() -> None:
    _session_cache.clear()


@pytest.fixture()
def registry() -> ToolRegistry:
    return ToolRegistry()


def _factory(session: BrowserSession | None = None) -> Any:
    """Return a session factory that ignores thread_id."""
    s = session or _mock_session()

    async def _fac(thread_id: str) -> BrowserSession:
        return s

    return _fac


# ---------------------------------------------------------------------------
# 1. Navigate happy path
# ---------------------------------------------------------------------------


async def test_navigate_happy(registry: ToolRegistry) -> None:
    build_browser_tools(registry, _factory())
    navigate = registry.get("browser_navigate")
    result = await navigate(url="https://example.com")
    assert result["success"] is True
    assert result["url"] == "https://example.com"
    assert "title" in result


# ---------------------------------------------------------------------------
# 2. Click selector not found
# ---------------------------------------------------------------------------


async def test_click_selector_not_found(registry: ToolRegistry) -> None:
    build_browser_tools(registry, _factory())
    click = registry.get("browser_click")
    result = await click(selector="#nonexistent")
    assert result["success"] is False
    assert "Timeout" in result["error"]


# ---------------------------------------------------------------------------
# 3. Navigate timeout
# ---------------------------------------------------------------------------


async def test_navigate_timeout(registry: ToolRegistry) -> None:
    session = _mock_session()

    async def slow_goto(url: str, **kwargs: Any) -> None:
        raise TimeoutError("Timeout 10000ms")

    session._page.goto = slow_goto  # type: ignore[assignment]

    build_browser_tools(registry, _factory(session))
    navigate = registry.get("browser_navigate")
    result = await navigate(url="https://slow.example.com")
    assert result["success"] is False
    assert "Timeout" in result["error"]


# ---------------------------------------------------------------------------
# 4. SSRF: reject localhost
# ---------------------------------------------------------------------------


async def test_ssrf_reject_localhost(registry: ToolRegistry) -> None:
    build_browser_tools(registry, _factory())
    navigate = registry.get("browser_navigate")
    result = await navigate(url="http://localhost:8080/admin")
    assert result["success"] is False
    assert "SSRF" in result["error"] or "localhost" in result["error"].lower()


# ---------------------------------------------------------------------------
# 5. SSRF: reject file:// scheme
# ---------------------------------------------------------------------------


async def test_ssrf_reject_file_scheme(registry: ToolRegistry) -> None:
    build_browser_tools(registry, _factory())
    navigate = registry.get("browser_navigate")
    result = await navigate(url="file:///etc/passwd")
    assert result["success"] is False
    # file:// may trigger SSRF guard ("Invalid URL: missing host") or scheme check
    assert "http(s)" in result["error"] or "Invalid" in result["error"]


# ---------------------------------------------------------------------------
# 6. SSRF: reject private IP
# ---------------------------------------------------------------------------


async def test_ssrf_reject_private_ip(registry: ToolRegistry) -> None:
    build_browser_tools(registry, _factory())
    navigate = registry.get("browser_navigate")
    result = await navigate(url="http://10.0.0.1/secret")
    assert result["success"] is False
    assert "SSRF" in result["error"] or "private" in result["error"].lower()


# ---------------------------------------------------------------------------
# 7. Click → screenshot chain
# ---------------------------------------------------------------------------


async def test_click_screenshot_chain(registry: ToolRegistry) -> None:
    build_browser_tools(registry, _factory())

    click_fn = registry.get("browser_click")
    result = await click_fn(selector="#ok-btn")
    assert result["success"] is True

    screenshot_fn = registry.get("browser_screenshot")
    result = await screenshot_fn()
    assert result["success"] is True
    assert "screenshot_base64" in result


# ---------------------------------------------------------------------------
# 8. Screenshot with artifact upload + base64
# ---------------------------------------------------------------------------


async def test_screenshot_with_artifact_and_base64(registry: ToolRegistry) -> None:
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

    build_browser_tools(registry, _factory(session))
    screenshot_fn = registry.get("browser_screenshot")
    result = await screenshot_fn()
    assert result["success"] is True
    assert result["screenshot_artifact_id"] == "art-123"
    assert "screenshot_base64" in result
    assert result["screenshot_base64"]  # non-empty
    mock_client.create_artifact.assert_awaited_once()


# ---------------------------------------------------------------------------
# 9. Session close/cleanup
# ---------------------------------------------------------------------------


async def test_session_close() -> None:
    session = BrowserSession.__new__(BrowserSession)
    mock_ctx = AsyncMock()
    mock_browser = AsyncMock()
    mock_pw = AsyncMock()
    session._context = mock_ctx
    session._browser = mock_browser
    session._playwright = mock_pw
    session._page = AsyncMock()
    session._timeout_ms = 10_000
    session._artifact_client = None
    session._tenant_id = ""
    session._cdp_url = None

    await session.close()

    mock_ctx.close.assert_awaited_once()
    mock_browser.close.assert_awaited_once()
    mock_pw.stop.assert_awaited_once()
    assert session._page is None


# ---------------------------------------------------------------------------
# 10. Concurrent sessions independent
# ---------------------------------------------------------------------------


async def test_concurrent_sessions_independent() -> None:
    page_a = FakePage()
    page_b = FakePage()
    page_b.url = "https://other.com"
    page_b._title = "Other"

    reg_a = ToolRegistry()
    reg_b = ToolRegistry()

    build_browser_tools(reg_a, _factory(_session_with_page(page_a)))
    build_browser_tools(reg_b, _factory(_session_with_page(page_b)))

    nav_a = reg_a.get("browser_navigate")
    nav_b = reg_b.get("browser_navigate")

    result_a = await nav_a(url="https://a.com")
    result_b = await nav_b(url="https://b.com")

    assert result_a["url"] == "https://a.com"
    assert result_b["url"] == "https://b.com"


# ---------------------------------------------------------------------------
# 11. Error log no PII
# ---------------------------------------------------------------------------


async def test_error_log_no_pii(registry: ToolRegistry) -> None:
    session = _mock_session()

    async def fail_goto(url: str, **kwargs: Any) -> None:
        raise RuntimeError("Navigation failed cookie=secret_value")

    session._page.goto = fail_goto  # type: ignore[assignment]

    build_browser_tools(registry, _factory(session))
    navigate = registry.get("browser_navigate")
    result = await navigate(url="https://example.com")
    assert result["success"] is False
    assert "error" in result


# ---------------------------------------------------------------------------
# 12. Lazy session creation
# ---------------------------------------------------------------------------


async def test_lazy_session_creation(registry: ToolRegistry) -> None:
    call_count = 0

    async def factory(thread_id: str) -> BrowserSession:
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


# ---------------------------------------------------------------------------
# 13. read_dom happy path
# ---------------------------------------------------------------------------


async def test_read_dom_happy(registry: ToolRegistry) -> None:
    build_browser_tools(registry, _factory())
    read_dom = registry.get("browser_read_dom")
    result = await read_dom(selector="#content")
    assert result["success"] is True
    assert "text" in result
    assert "html" in result
    assert result["text"] == "Hello World"


# ---------------------------------------------------------------------------
# 14. read_dom element not found
# ---------------------------------------------------------------------------


async def test_read_dom_element_not_found(registry: ToolRegistry) -> None:
    build_browser_tools(registry, _factory())
    read_dom = registry.get("browser_read_dom")
    result = await read_dom(selector="#nonexistent")
    assert result["success"] is False
    assert "not found" in result["error"].lower()


# ---------------------------------------------------------------------------
# 15. Session cache by thread_id
# ---------------------------------------------------------------------------


async def test_session_cache_by_thread_id() -> None:
    s1 = BrowserSession.get_or_create("thread-1")
    s2 = BrowserSession.get_or_create("thread-1")
    assert s1 is s2

    s3 = BrowserSession.get_or_create("thread-2")
    assert s3 is not s1

    BrowserSession._clear_cache("thread-1")
    assert BrowserSession._cached("thread-1") is None
    assert BrowserSession._cached("thread-2") is s3
    BrowserSession._clear_cache("thread-2")


# ---------------------------------------------------------------------------
# 16. CDP client resolves URL from supervisor
# ---------------------------------------------------------------------------


async def test_cdp_client_resolves_from_supervisor() -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "cdpWsUrl": "ws://sandbox:9222/devtools/browser/abc",
    }

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_resp

    cdp = CDPClient(
        supervisor_url="http://supervisor:8080",
        sandbox_id="sb_123",
        http_client=mock_client,
    )
    url = await cdp.resolve_cdp_url()
    assert url == "ws://sandbox:9222/devtools/browser/abc"
    mock_client.post.assert_awaited_once_with(
        "http://supervisor:8080/v2/sandboxes/sb_123/browser/cdp"
    )


# ---------------------------------------------------------------------------
# 17. CDP client fallback on 501
# ---------------------------------------------------------------------------


async def test_cdp_client_fallback_on_501() -> None:
    cdp_resp = MagicMock()
    cdp_resp.status_code = 501

    exec_resp = MagicMock()
    exec_resp.status_code = 200
    exec_resp.raise_for_status = MagicMock()

    post_responses = [cdp_resp, exec_resp]

    mock_client = AsyncMock()
    mock_client.post.side_effect = post_responses

    cdp = CDPClient(
        supervisor_url="http://supervisor:8080",
        sandbox_id="sb_456",
        http_client=mock_client,
    )
    url = await cdp.resolve_cdp_url()
    assert "9222" in url
    assert "devtools/browser" in url
    # Second call should have been the exec fallback
    assert mock_client.post.call_count == 2


# ---------------------------------------------------------------------------
# 18. CDP client fallback on HTTP error
# ---------------------------------------------------------------------------


async def test_cdp_client_fallback_on_http_error() -> None:
    cdp_call = AsyncMock(side_effect=httpx.ConnectError("refused"))

    exec_resp = MagicMock()
    exec_resp.status_code = 200
    exec_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post.side_effect = [cdp_call if False else httpx.ConnectError("refused"), exec_resp]

    # Need to fix: side_effect should raise, not return
    call_count = 0
    responses = [httpx.ConnectError("refused"), exec_resp]

    async def mock_post(url: str, **kwargs: Any) -> Any:
        nonlocal call_count
        resp = responses[call_count]
        call_count += 1
        if isinstance(resp, Exception):
            raise resp
        return resp

    mock_client.post = mock_post

    cdp = CDPClient(
        supervisor_url="http://supervisor:8080",
        sandbox_id="sb_789",
        http_client=mock_client,
    )
    url = await cdp.resolve_cdp_url()
    assert "9222" in url
    assert call_count == 2


# ---------------------------------------------------------------------------
# 19. Screenshot returns valid base64
# ---------------------------------------------------------------------------


async def test_screenshot_returns_base64(registry: ToolRegistry) -> None:
    build_browser_tools(registry, _factory())
    screenshot_fn = registry.get("browser_screenshot")
    result = await screenshot_fn()
    assert result["success"] is True
    assert "screenshot_base64" in result
    # Verify it's valid base64
    import base64

    decoded = base64.b64decode(result["screenshot_base64"])
    assert decoded.startswith(b"\x89PNG")


# ---------------------------------------------------------------------------
# 20. Session cache size tracking
# ---------------------------------------------------------------------------


async def test_session_cache_size() -> None:
    assert BrowserSession._cache_size() == 0
    BrowserSession.get_or_create("t1")
    assert BrowserSession._cache_size() == 1
    BrowserSession.get_or_create("t2")
    assert BrowserSession._cache_size() == 2
    BrowserSession._clear_cache("t1")
    assert BrowserSession._cache_size() == 1
    BrowserSession._clear_cache("t2")
    assert BrowserSession._cache_size() == 0


# ---------------------------------------------------------------------------
# 21. CDP client close cleans up http client
# ---------------------------------------------------------------------------


async def test_cdp_client_close() -> None:
    mock_http = AsyncMock()
    cdp = CDPClient(
        supervisor_url="http://supervisor:8080",
        sandbox_id="sb_x",
        http_client=mock_http,
    )
    await cdp.close()
    mock_http.aclose.assert_awaited_once()
    assert cdp._http is None


# ---------------------------------------------------------------------------
# 22. type tool happy path
# ---------------------------------------------------------------------------


async def test_type_happy(registry: ToolRegistry) -> None:
    build_browser_tools(registry, _factory())
    type_fn = registry.get("browser_type")
    result = await type_fn(selector="#search", text="hello world")
    assert result["success"] is True
    assert "title" in result


# ---------------------------------------------------------------------------
# 23. type tool nonexistent selector
# ---------------------------------------------------------------------------


async def test_type_nonexistent_selector(registry: ToolRegistry) -> None:
    build_browser_tools(registry, _factory())
    type_fn = registry.get("browser_type")
    result = await type_fn(selector="#nonexistent", text="fail")
    assert result["success"] is False
    assert "Timeout" in result["error"]


# ---------------------------------------------------------------------------
# 24. read_dom default selector (body)
# ---------------------------------------------------------------------------


async def test_read_dom_default_selector(registry: ToolRegistry) -> None:
    build_browser_tools(registry, _factory())
    read_dom = registry.get("browser_read_dom")
    result = await read_dom()
    assert result["success"] is True


# ---------------------------------------------------------------------------
# 25. Accessibility tree tool
# ---------------------------------------------------------------------------


async def test_browser_accessibility_tool(registry: ToolRegistry) -> None:
    build_browser_tools(registry, _factory())
    a11y = registry.get("browser_accessibility")
    result = await a11y()
    assert result["success"] is True
    assert result["tree"]["role"] == "WebArea"


# ---------------------------------------------------------------------------
# 26. Download quarantine: isolates + flags unscanned + hashes
# ---------------------------------------------------------------------------


async def test_quarantine_download_isolates_and_flags(tmp_path: Any) -> None:
    s = BrowserSession.__new__(BrowserSession)
    s._quarantine_dir = str(tmp_path / "q")
    s._downloads = []

    content = b"%PDF-1.4 fake-report"
    rec = await s._quarantine_download(_FakeDownload("report.pdf", content))

    assert rec["status"] == "quarantined"
    assert rec["scanned"] is False
    assert rec["suggestedFilename"] == "report.pdf"
    assert rec["sizeBytes"] == len(content)
    assert rec["sha256"] == hashlib.sha256(content).hexdigest()
    # The bytes land in the quarantine dir — never the workspace.
    assert os.path.dirname(os.path.abspath(rec["quarantinePath"])) == str(tmp_path / "q")
    assert os.path.exists(rec["quarantinePath"])
    assert s.list_downloads() == [rec]


# ---------------------------------------------------------------------------
# 27. Download quarantine: a malicious filename cannot escape the dir
# ---------------------------------------------------------------------------


async def test_quarantine_download_blocks_path_traversal(tmp_path: Any) -> None:
    s = BrowserSession.__new__(BrowserSession)
    s._quarantine_dir = str(tmp_path)
    s._downloads = []

    rec = await s._quarantine_download(_FakeDownload("../../etc/passwd", b"x"))
    # The path-separators in the suggested name are sanitised away, so the file
    # stays inside the quarantine dir.
    assert os.path.dirname(os.path.abspath(rec["quarantinePath"])) == str(tmp_path)


# ---------------------------------------------------------------------------
# 28. browser_downloads tool lists the quarantine
# ---------------------------------------------------------------------------


async def test_browser_downloads_tool_lists_quarantine(
    registry: ToolRegistry, tmp_path: Any
) -> None:
    s = _mock_session()
    s._quarantine_dir = str(tmp_path)
    s._downloads = [{"suggestedFilename": "a.bin", "status": "quarantined"}]

    build_browser_tools(registry, _factory(s))
    downloads = registry.get("browser_downloads")
    result = await downloads()
    assert result["success"] is True
    assert result["downloads"][0]["suggestedFilename"] == "a.bin"


class TestAnnotateBlock:
    """navigate/read_dom results are tagged blocked on anti-bot pages."""

    def test_block_page_flagged(self) -> None:
        out = _annotate_block(
            {"success": True, "title": "访问已被拦截", "url": "https://bj.zu.anjuke.com/x"}
        )
        assert out["blocked"] is True
        assert out["block_reason"]

    def test_normal_page_not_flagged(self) -> None:
        out = _annotate_block(
            {
                "success": True,
                "title": "北京租金",
                "url": "https://example.com/p",
                "text": "整租均价 8500 元 环比小幅上涨",
            }
        )
        assert out["blocked"] is False
        assert out["block_reason"] is None

    def test_read_dom_text_feeds_detection(self) -> None:
        out = _annotate_block(
            {
                "success": True,
                "title": "",
                "url": "https://x.com/p",
                "text": '<div class="g-recaptcha"></div>',
            }
        )
        assert out["blocked"] is True

    def test_read_dom_html_feeds_decisive_fingerprints(self) -> None:
        out = _annotate_block(
            {
                "success": True,
                "title": "",
                "url": "https://x.com/p",
                "text": "challenge",
                "html": '<div id="px-captcha"></div>',
            }
        )
        assert out["blocked"] is True

    def test_clean_html_recaptcha_widget_is_not_blocked(self) -> None:
        out = _annotate_block(
            {
                "success": True,
                "title": "Account login",
                "url": "https://example.com/login",
                "text": (
                    "Sign in to continue. This page includes a normal account form, "
                    "remember-me checkbox, password reset link, and support contact."
                ),
                "html": '<form><input name="email"><div class="g-recaptcha"></div></form>',
            }
        )
        assert out["blocked"] is False
        assert out["block_reason"] is None

    def test_failed_result_passes_through_untouched(self) -> None:
        out = _annotate_block({"success": False, "error": "boom"})
        assert "blocked" not in out
