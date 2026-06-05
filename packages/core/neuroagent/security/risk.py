"""High-risk action classification + confirmation gate (PRD §9.1 安全底线).

Certain tool calls (destructive shell, disk writes, force-push, shutdown) are
flagged *high* risk and must be explicitly confirmed before the Tool Executor
runs them. The gate is consulted by the executor: a high-risk call that the
user has not confirmed is returned as ``[confirmation-required]`` instead of
executing. This is pure policy logic — no sandbox/infra needed — so it is unit
tested in isolation.
"""

from __future__ import annotations

import re

# Tools whose arguments can carry destructive operations.
HIGH_RISK_TOOLS: frozenset[str] = frozenset({"bash", "shell", "exec", "file_ops"})

_HIGH_RISK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\brm\s+-[a-z]*[rf]"),          # rm -rf / rm -fr
    re.compile(r"\bmkfs\b"),                      # format filesystem
    re.compile(r"\bdd\b.*\bof=/dev/"),           # raw disk write
    re.compile(r">\s*/dev/sd[a-z]"),             # overwrite a block device
    re.compile(r":\s*\(\s*\)\s*\{.*\};\s*:"),    # fork bomb
    re.compile(r"\bchmod\s+-R\s+0*777\s+/"),     # world-writable root
    re.compile(r"\b(shutdown|reboot|halt|poweroff)\b"),
    re.compile(r"\bgit\s+push\b.*--force"),      # force push
    re.compile(r"\bdrop\s+database\b", re.IGNORECASE),
)


def classify_tool_risk(tool_name: str, args_raw: str | None) -> str:
    """Return ``"high"`` if the call matches a destructive pattern, else ``"low"``."""
    if tool_name in HIGH_RISK_TOOLS and args_raw:
        for pat in _HIGH_RISK_PATTERNS:
            if pat.search(args_raw):
                return "high"
    return "low"


class RiskGate:
    """Decides whether a tool call needs explicit confirmation before running."""

    def requires_confirmation(
        self,
        tool_name: str,
        args_raw: str | None,
        *,
        tc_id: str | None = None,
        confirmed_ids: frozenset[str] | set[str] | None = None,
    ) -> bool:
        """True iff the call is high-risk and not already user-confirmed."""
        if classify_tool_risk(tool_name, args_raw) != "high":
            return False
        return tc_id not in (confirmed_ids or frozenset())
