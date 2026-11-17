"""LangGraph middleware for prompt-injection detection (observation mode).

Adds a ``detect_injection`` node before the ``plan`` node in the agent
graph.  When matches are found at severity >= *medium*, the middleware:

1. Creates an OTel span ``prompt_injection_detected`` with attributes.
2. Records an entry in the :class:`InjectionAuditStore`.

The request is **never rejected** (Phase 1 = observation only).
"""

from __future__ import annotations

import logging
from typing import Any

from opentelemetry import trace

from sanyuan_agent.security.prompt_injection.audit import (
    InjectionAuditStore,
    make_audit_entry,
)
from sanyuan_agent.security.prompt_injection.detector import InjectionDetector

logger = logging.getLogger(__name__)

tracer = trace.get_tracer("sanyuan_agent.security.prompt_injection")

_MIN_SEVERITY_FOR_SPAN = "medium"
_SEV_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def _severity_gte(a: str, b: str) -> bool:
    return _SEV_ORDER.get(a, 0) >= _SEV_ORDER.get(b, 0)


class InjectionDetectionMiddleware:
    """Configurable middleware that holds references to the detector and audit store.

    Usage::

        mw = InjectionDetectionMiddleware(detector, audit_store)
        # Then call mw.detect_node(state) as a LangGraph node.
    """

    def __init__(
        self,
        detector: InjectionDetector,
        audit: InjectionAuditStore,
        *,
        tenant_id: str = "",
        user_id: str = "",
        thread_id: str = "",
    ) -> None:
        self._detector = detector
        self._audit = audit
        self._tenant_id = tenant_id
        self._user_id = user_id
        self._thread_id = thread_id

    def update_context(
        self,
        *,
        tenant_id: str | None = None,
        user_id: str | None = None,
        thread_id: str | None = None,
    ) -> None:
        if tenant_id is not None:
            self._tenant_id = tenant_id
        if user_id is not None:
            self._user_id = user_id
        if thread_id is not None:
            self._thread_id = thread_id

    async def detect_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """LangGraph node: scan messages and record detections.

        Returns the state dict unchanged (observation-only mode).
        """
        messages: list[Any] = state.get("messages", [])

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            if role not in ("user", "tool"):
                continue

            content = msg.get("content", "")
            if not isinstance(content, str) or not content:
                continue

            matches = self._detector.detect(content)
            if not matches:
                continue

            source = "user_input" if role == "user" else "tool_output"

            # Only log + span when severity >= medium
            any_medium = any(
                _severity_gte(m.severity, _MIN_SEVERITY_FOR_SPAN) for m in matches
            )
            if any_medium:
                self._write_otel_span(source, matches)
                entry = make_audit_entry(
                    tenant_id=self._tenant_id,
                    user_id=self._user_id,
                    thread_id=self._thread_id,
                    source=source,
                    matches=matches,
                )
                self._audit.record(entry)
                logger.info(
                    "injection detected: source=%s rules=%s conf=%.2f",
                    source,
                    [m.rule_name for m in matches],
                    max(m.confidence for m in matches),
                )

        return state

    def _write_otel_span(
        self, source: str, matches: list[Any]
    ) -> None:
        with tracer.start_as_current_span("prompt_injection_detected") as span:
            span.set_attribute(
                "injection.rules", ",".join(m.rule_name for m in matches)
            )
            span.set_attribute(
                "injection.confidence_max",
                max(m.confidence for m in matches),
            )
            span.set_attribute("injection.matched_count", len(matches))
            span.set_attribute("injection.source", source)
            span.set_attribute("injection.severity_max", matches[0].severity)


def add_injection_detection(
    graph_builder: Any,
    middleware: InjectionDetectionMiddleware,
    *,
    before_node: str = "plan",
) -> None:
    """Add the injection-detection node to a ``StateGraph`` builder.

    The new node is set as the entry point and edges into *before_node*.
    """
    graph_builder.add_node("injection_detect", middleware.detect_node)
    graph_builder.set_entry_point("injection_detect")
    graph_builder.add_edge("injection_detect", before_node)
