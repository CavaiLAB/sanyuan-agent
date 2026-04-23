"""Prompt-injection detection for NeuroAgent (Phase 1: observation only)."""

from neuroagent.security.prompt_injection.audit import (
    InjectionAuditEntry,
    InjectionAuditStore,
    make_audit_entry,
)
from neuroagent.security.prompt_injection.detector import (
    InjectionDetector,
    InjectionMatch,
)
from neuroagent.security.prompt_injection.middleware import (
    InjectionDetectionMiddleware,
    add_injection_detection,
)

__all__ = [
    "InjectionAuditEntry",
    "InjectionAuditStore",
    "InjectionDetectionMiddleware",
    "InjectionDetector",
    "InjectionMatch",
    "add_injection_detection",
    "make_audit_entry",
]
