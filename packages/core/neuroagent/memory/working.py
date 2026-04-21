"""Working memory — structured plan-act-observe step history."""

from __future__ import annotations

from dataclasses import dataclass, field

from neuroagent.memory.abc import IMemoryStore, InMemoryStore


@dataclass(frozen=True, slots=True)
class WorkingStep:
    """One plan-act-observe triplet."""

    plan: str
    action: str
    result: str
    ts: float = field(default_factory=lambda: __import__("time").time())


class WorkingMemory:
    """Tracks plan-act-observe triplets for context summarisation."""

    _STORE_KEY = "_working_steps"

    def __init__(
        self,
        *,
        store: IMemoryStore | None = None,
    ) -> None:
        self._store = store or InMemoryStore()
        raw = self._store.load(self._STORE_KEY, [])
        self._steps: list[WorkingStep] = [WorkingStep(**s) for s in raw]

    def add_step(self, plan: str, action: str, result: str) -> None:
        """Record a new step."""
        step = WorkingStep(plan=plan, action=action, result=result)
        self._steps.append(step)
        self._store.save(
            self._STORE_KEY,
            [
                {"plan": s.plan, "action": s.action, "result": s.result, "ts": s.ts}
                for s in self._steps
            ],
        )

    def get_summary(self, max_steps: int = 5) -> str:
        """Return a text summary of the most recent steps for plan prompts.

        TODO(P1-PY-020): Add LLM-based compression when steps exceed a token budget.
        """
        recent = self._steps[-max_steps:]
        if not recent:
            return "(no previous steps)"
        lines: list[str] = []
        for i, step in enumerate(recent, 1):
            lines.append(
                f"Step {i}:\n"
                f"  Plan: {step.plan}\n"
                f"  Action: {step.action}\n"
                f"  Result: {step.result}"
            )
        return "\n\n".join(lines)

    def clear(self) -> None:
        """Clear step history and store."""
        self._steps.clear()
        self._store.clear()
