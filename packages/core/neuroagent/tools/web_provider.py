"""Web tool provider: a thin wrapper over the existing web_search + web_fetch builders.

`WebToolProvider` yields the two existing web tools as specs with
``category="web"`` / ``side_effect="read"`` metadata. It changes **no behavior
and no schema**: each tool's callable and its reflected schema come straight from
``build_web_search_tool`` / ``build_web_fetch_tool`` (called with their defaults),
and only descriptive metadata is attached. Per PR1 that metadata never gates
execution (``ToolPolicy`` / ``RiskGate`` remain authoritative).

**No new secret or env behavior.** The builders are called with their defaults,
so web_search's optional Brave key (``NEUROAGENT_WEB_SEARCH_API_KEY``) continues
to be read by the builder exactly as before; this module reads no env itself and
declares no ``required_secrets``.
"""

from __future__ import annotations

from neuroagent.tools.provider import ToolSpec, spec_from_callable
from neuroagent.tools.registry import ToolRegistry
from neuroagent.tools.web_fetch import build_web_fetch_tool
from neuroagent.tools.web_search import build_web_search_tool


class WebToolProvider:
    """Thin ToolProvider wrapping the existing web_search + web_fetch builders.

    Selectable through the provider path: reference the provider in
    ``AgentSpec.providers`` and enable ``"web_search"`` / ``"web_fetch"`` via
    ``AgentSpec.tools`` / ``tool_sets``.
    """

    def provide(self) -> list[ToolSpec]:
        # Build into a scratch registry, then re-emit the registered callables +
        # schemas verbatim as specs (schema passed through, never re-derived).
        scratch = ToolRegistry()
        search = build_web_search_tool(scratch)
        fetch = build_web_fetch_tool(scratch)
        return [
            spec_from_callable(
                search,
                schema=scratch.get_schema("web_search"),
                category="web",
                capabilities=("web", "search"),
                side_effect="read",
            ),
            spec_from_callable(
                fetch,
                schema=scratch.get_schema("web_fetch"),
                category="web",
                capabilities=("web", "fetch"),
                side_effect="read",
            ),
        ]
