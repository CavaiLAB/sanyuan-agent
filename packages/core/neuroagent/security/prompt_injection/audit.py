"""In-memory audit store for prompt-injection detections (Phase 1)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class InjectionAuditEntry:
    """A single recorded injection-detection event."""

    timestamp: float
    tenant_id: str
    user_id: str
    thread_id: str
    source: str  # "user_input" | "tool_output"
    rule_names: list[str]
    severity_max: str
    confidence_max: float
    matched_count: int
    matched_spans: list[str]


class InjectionAuditStore:
    """Thread-safe in-memory audit log with simple query support.

    Phase 2 should replace this with a database-backed implementation.
    """

    def __init__(self, max_entries: int = 10000) -> None:
        self._entries: list[InjectionAuditEntry] = []
        self._max = max_entries
        self._lock = threading.Lock()

    def record(self, entry: InjectionAuditEntry) -> None:
        with self._lock:
            if len(self._entries) >= self._max:
                self._entries = self._entries[len(self._entries) // 2 :]
            self._entries.append(entry)

    def query(
        self,
        *,
        severity: str | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[InjectionAuditEntry], int]:
        """Return (matching_entries, total_count)."""
        with self._lock:
            pool = list(self._entries)
        if severity is not None:
            pool = [e for e in pool if e.severity_max == severity]
        pool.sort(key=lambda e: e.timestamp, reverse=True)
        total = len(pool)
        page = pool[offset : offset + limit]
        return page, total

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._entries)


def make_audit_entry(
    *,
    tenant_id: str,
    user_id: str,
    thread_id: str,
    source: str,
    matches: list[Any],
) -> InjectionAuditEntry:
    """Convenience factory from detector matches."""
    sev_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    max_sev = "low"
    max_conf = 0.0
    names: list[str] = []
    spans: list[str] = []
    for m in matches:
        names.append(m.rule_name)
        spans.append(m.matched_span)
        if sev_order.get(m.severity, 0) > sev_order.get(max_sev, 0):
            max_sev = m.severity
        if m.confidence > max_conf:
            max_conf = m.confidence
    return InjectionAuditEntry(
        timestamp=time.time(),
        tenant_id=tenant_id,
        user_id=user_id,
        thread_id=thread_id,
        source=source,
        rule_names=names,
        severity_max=max_sev,
        confidence_max=round(max_conf, 4),
        matched_count=len(matches),
        matched_spans=spans,
    )
