"""Playwright browser tools — navigate, click, type, screenshot, read_dom via CDP."""

from sanyuan_agent.tools.browser.actions import BrowserToolProvider, build_browser_tools
from sanyuan_agent.tools.browser.client import CDPClient
from sanyuan_agent.tools.browser.session import BrowserSession

__all__ = [
    "BrowserSession",
    "BrowserToolProvider",
    "CDPClient",
    "build_browser_tools",
]
