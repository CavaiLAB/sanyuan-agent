"""Per-agent tool permission policy: ordered glob rules → allow/deny/confirm.

Complements the content-based :class:`RiskGate` (which inspects *arguments*
for destructive patterns): the policy decides, per agent, whether a tool may
be called *at all* — ``deny`` blocks it outright, ``confirm`` requires an
explicit user confirmation (same ``confirmed_tool_calls`` flow as the risk
gate), ``allow`` lets it through. Glob patterns make scoping whole tool
families one rule (e.g. ``mcp__*`` for every bridged MCP tool).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase

POLICY_ACTIONS: frozenset[str] = frozenset({"allow", "deny", "confirm"})


@dataclass(frozen=True)
class PolicyRule:
    """One ordered rule: a glob *pattern* over tool names plus its action."""

    pattern: str
    action: str

    def __post_init__(self) -> None:
        if not self.pattern:
            raise ValueError("policy rule pattern must be non-empty")
        if self.action not in POLICY_ACTIONS:
            raise ValueError(
                f"unknown policy action {self.action!r}; "
                f"expected one of {sorted(POLICY_ACTIONS)}"
            )


class ToolPolicy:
    """First matching rule wins; *default* applies when nothing matches."""

    def __init__(
        self,
        rules: Sequence[PolicyRule] = (),
        *,
        default: str = "allow",
    ) -> None:
        if default not in POLICY_ACTIONS:
            raise ValueError(
                f"unknown default action {default!r}; "
                f"expected one of {sorted(POLICY_ACTIONS)}"
            )
        self.rules = tuple(rules)
        self.default = default

    def decide(self, tool_name: str) -> str:
        """Return ``"allow"`` / ``"deny"`` / ``"confirm"`` for *tool_name*."""
        for rule in self.rules:
            if fnmatchcase(tool_name, rule.pattern):
                return rule.action
        return self.default
