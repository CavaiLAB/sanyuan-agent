"""Regex-based prompt-injection detector with confidence scoring."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_RULES_BUILTIN = "rules.yaml"


@dataclass(frozen=True, slots=True)
class InjectionMatch:
    """A single rule match against input text."""

    rule_name: str
    severity: str  # "low" | "medium" | "high" | "critical"
    confidence: float
    matched_span: str


@dataclass(frozen=True, slots=True)
class _RuleDef:
    """Compiled rule from the YAML rule library."""

    name: str
    pattern: re.Pattern[str]
    severity: str
    base_confidence: float
    tags: list[str]


class InjectionDetector:
    """Loads the rule library and runs regex-based detection.

    Parameters
    ----------
    rules_path:
        Path to a custom YAML rule file.  When *None*, the built-in
        ``rules.yaml`` shipped with the package is used.
    extra_rules_path:
        Optional second YAML file; rules are merged (later file wins on
        name collisions) for easy local customisation.
    """

    def __init__(
        self,
        rules_path: str | Path | None = None,
        extra_rules_path: str | Path | None = None,
    ) -> None:
        self._rules: list[_RuleDef] = []
        self._load_rules(rules_path)
        if extra_rules_path is not None:
            self._load_rules(extra_rules_path)
        logger.info("loaded %d injection detection rules", len(self._rules))

    # -- public API -------------------------------------------------------

    def detect(self, text: str) -> list[InjectionMatch]:
        """Return all rule matches for *text*, sorted by confidence desc."""
        if not text:
            return []
        matches: list[InjectionMatch] = []
        for rule in self._rules:
            for m in rule.pattern.finditer(text):
                span = m.group(0)
                # Boost confidence when the match is longer relative to the
                # original pattern (reduces false positives from short hits).
                boost = min(0.05, len(span) / max(len(m.re.pattern), 1) * 0.02)
                conf = min(rule.base_confidence + boost, 1.0)
                matches.append(
                    InjectionMatch(
                        rule_name=rule.name,
                        severity=rule.severity,
                        confidence=round(conf, 4),
                        matched_span=span,
                    )
                )
        matches.sort(key=lambda m: m.confidence, reverse=True)
        return matches

    @property
    def rule_count(self) -> int:
        return len(self._rules)

    @property
    def rule_names(self) -> list[str]:
        return [r.name for r in self._rules]

    # -- loading ----------------------------------------------------------

    def _load_rules(self, path: str | Path | None) -> None:
        """Load rules from *path*, or the built-in YAML if *path* is None."""
        if path is None:
            raw = self._read_builtin_rules()
        else:
            raw = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
        rule_list = data.get("rules", [])
        seen: set[str] = set()
        for entry in rule_list:
            name: str = entry.get("name", "")
            pattern_str: str = entry.get("pattern", "")
            severity: str = entry.get("severity", "low")
            confidence: float = float(entry.get("confidence", 0.5))
            tags: list[str] = entry.get("tags", [])
            if not name or not pattern_str:
                continue
            try:
                compiled = re.compile(pattern_str, re.IGNORECASE | re.MULTILINE)
            except re.error:
                logger.warning("invalid regex in rule %r: %s", name, pattern_str)
                continue
            if name in seen:
                # Replace existing rule (for merge support)
                self._rules = [r for r in self._rules if r.name != name]
            seen.add(name)
            self._rules.append(
                _RuleDef(
                    name=name,
                    pattern=compiled,
                    severity=severity,
                    base_confidence=confidence,
                    tags=list(tags),
                )
            )

    @staticmethod
    def _read_builtin_rules() -> str:
        pkg = resources.files("sanyuan_agent.security.prompt_injection")
        return pkg.joinpath(_RULES_BUILTIN).read_text(encoding="utf-8")
