"""Web tool provider: a thin wrapper over the existing web_search + web_fetch builders.

`WebToolProvider` yields the two existing web tools as specs with
``category="web"`` / ``side_effect="read"`` metadata. The default constructor
changes **no behavior and no schema**. An app may inject a ``SearchRouter``;
provider selection remains server-side and the public tool schema stays the same.
Per PR1 metadata never gates execution (``ToolPolicy`` / ``RiskGate`` remain
authoritative).

**No secret values in the harness.** Without an injected Router, web_search's
optional Brave key (``SANYUAN_AGENT_WEB_SEARCH_API_KEY``) continues to be read by
the builder exactly as before. This module reads no env itself.
"""

from __future__ import annotations

from sanyuan_agent.tools.provider import ToolSpec, spec_from_callable
from sanyuan_agent.tools.registry import ToolRegistry
from sanyuan_agent.tools.web_fetch import build_web_fetch_tool
from sanyuan_agent.tools.web_reader import ReaderRouter
from sanyuan_agent.tools.web_search import SearchRouter, build_web_search_tool


class WebToolProvider:
    """Thin ToolProvider wrapping the existing web_search + web_fetch builders.

    Selectable through the provider path: reference the provider in
    ``AgentSpec.providers`` and enable ``"web_search"`` / ``"web_fetch"`` via
    ``AgentSpec.tools`` / ``tool_sets``.
    """

    def __init__(
        self,
        search_router: SearchRouter | None = None,
        reader_router: ReaderRouter | None = None,
        *,
        search_required_secrets: tuple[str, ...] = (),
        fetch_required_secrets: tuple[str, ...] = (),
    ) -> None:
        self._search_router = search_router
        self._reader_router = reader_router
        self._search_required_secrets = search_required_secrets
        self._fetch_required_secrets = fetch_required_secrets

    def provide(self) -> list[ToolSpec]:
        # Build into a scratch registry, then re-emit the registered callables +
        # schemas verbatim as specs (schema passed through, never re-derived).
        scratch = ToolRegistry()
        search = build_web_search_tool(scratch, router=self._search_router)
        fetch = build_web_fetch_tool(scratch, reader_router=self._reader_router)
        return [
            spec_from_callable(
                search,
                schema=scratch.get_schema("web_search"),
                category="web",
                capabilities=("web", "search"),
                side_effect="read",
                required_secrets=self._search_required_secrets,
            ),
            spec_from_callable(
                fetch,
                schema=scratch.get_schema("web_fetch"),
                category="web",
                capabilities=("web", "fetch"),
                side_effect="read",
                required_secrets=self._fetch_required_secrets,
            ),
        ]
