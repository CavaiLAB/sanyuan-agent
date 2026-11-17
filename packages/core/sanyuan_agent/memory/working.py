"""Working memory — plan-act-observe history with optional LLM compression.

Per PRD §7.6 the working set is bounded (~20 steps / ~8k tokens); beyond that
the oldest steps are compressed into a running summary while the most recent
steps stay verbatim. Compression uses an injected *summarizer* callable, so the
budget logic is unit-testable without an LLM. When no summarizer is supplied the
behaviour is identical to the previous ring-buffer (fully backward compatible).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from sanyuan_agent.memory.abc import IMemoryStore, InMemoryStore

# Compresses a block of step text into a shorter summary.
Summarizer = Callable[[str], str]


@dataclass(frozen=True, slots=True)
class WorkingStep:
    """One plan-act-observe triplet."""

    plan: str
    action: str
    result: str
    ts: float = field(default_factory=lambda: time.time())


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) — good enough for budgeting."""
    return max(1, len(text) // 4)


class WorkingMemory:
    """Tracks plan-act-observe triplets, compressing old ones past a budget."""

    _STORE_KEY = "_working_steps"
    _SUMMARY_KEY = "_working_summary"

    def __init__(
        self,
        *,
        store: IMemoryStore | None = None,
        summarizer: Summarizer | None = None,
        max_recent: int = 5,
        max_steps: int = 20,
        token_budget: int = 8000,
    ) -> None:
        self._store = store or InMemoryStore()
        self._summarizer = summarizer
        self._max_recent = max_recent
        self._max_steps = max_steps
        self._token_budget = token_budget
        raw = self._store.load(self._STORE_KEY, [])
        self._steps: list[WorkingStep] = [WorkingStep(**s) for s in raw]
        self._summary: str = self._store.load(self._SUMMARY_KEY, "") or ""

    def add_step(self, plan: str, action: str, result: str) -> None:
        """Record a new step, compressing older history if over budget."""
        self._steps.append(WorkingStep(plan=plan, action=action, result=result))
        self._maybe_compress()
        self._persist()

    @property
    def is_summarized(self) -> bool:
        """True once at least one compression has run."""
        return bool(self._summary)

    def get_summary(self, max_steps: int = 5) -> str:
        """Return the running summary (if any) plus the most recent steps."""
        recent = self._steps[-max_steps:]
        parts: list[str] = []
        if self._summary:
            parts.append(f"Summary of earlier steps:\n{self._summary}")
        if recent:
            parts.append(
                "\n\n".join(self._format_step(i, s) for i, s in enumerate(recent, 1))
            )
        if not parts:
            return "(no previous steps)"
        return "\n\n".join(parts)

    def clear(self) -> None:
        """Clear step history, summary, and backing store."""
        self._steps.clear()
        self._summary = ""
        self._store.clear()

    # -- internals ------------------------------------------------------------

    @staticmethod
    def _format_step(i: int, step: WorkingStep) -> str:
        return (
            f"Step {i}:\n"
            f"  Plan: {step.plan}\n"
            f"  Action: {step.action}\n"
            f"  Result: {step.result}"
        )

    def _total_tokens(self) -> int:
        body = sum(
            _estimate_tokens(f"{s.plan} {s.action} {s.result}") for s in self._steps
        )
        return body + _estimate_tokens(self._summary)

    def _maybe_compress(self) -> None:
        if self._summarizer is None:
            return
        over_budget = (
            len(self._steps) > self._max_steps
            or self._total_tokens() > self._token_budget
        )
        if not over_budget or len(self._steps) <= self._max_recent:
            return
        to_compress = self._steps[: -self._max_recent]
        keep = self._steps[-self._max_recent :]
        block = "\n\n".join(
            self._format_step(i, s) for i, s in enumerate(to_compress, 1)
        )
        prior = f"Earlier summary:\n{self._summary}\n\n" if self._summary else ""
        self._summary = self._summarizer(prior + block).strip()
        self._steps = keep

    def _persist(self) -> None:
        self._store.save(
            self._STORE_KEY,
            [
                {"plan": s.plan, "action": s.action, "result": s.result, "ts": s.ts}
                for s in self._steps
            ],
        )
        self._store.save(self._SUMMARY_KEY, self._summary)
