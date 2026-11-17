"""Workflow orchestration model — specs, run state machines, and events.

A *workflow* is a one-shot orchestration topology executed inside a single
run: ordered phases, each fanning out to concurrent work items, each item
delegated to a real sub-agent (an :class:`~sanyuan_agent.config.AgentSpec`
worker), followed by an optional synthesis step that reads only the workers'
results. This is deliberately NOT the goal-driven outer loop (Goal → LoopRun →
Iteration, see docs/specs/loop-engineering/) — a workflow run is a single
composite inner execution whose durable metadata can be projected elsewhere.

Terminology: a **sub-agent is the execution unit** (one work item → one agent
run); a **multi-agent workflow is the orchestration topology** over those
units. The Python runtime owns these semantics.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sanyuan_agent.config import AgentSpec

# ── Statuses ─────────────────────────────────────────────────────────────────
# WorkItem statuses align with the v0 WorkItemStatus contract vocabulary
# (planned | running | succeeded | failed; `denied` is reserved for a future
# tool-policy gate and never emitted by this engine). Workflow/phase statuses
# align with the AgentRun FSM terminal semantics, plus `partial` for an honest
# "some workers failed but the workflow completed under policy" outcome.

WorkItemStatus = Literal["planned", "running", "succeeded", "failed"]
PhaseStatus = Literal["pending", "running", "succeeded", "partial", "failed"]
# `cancelled` = an explicit stop (user cancel / TTL reaper) settled by the run
# REGISTRY, not by the engine: an async generator cannot yield on its close
# path, so the engine never emits the cancelled terminal itself (ADR
# di-workflow-background-runs D1/D5).
WorkflowStatus = Literal["queued", "running", "succeeded", "partial", "failed", "cancelled"]

WORKFLOW_TERMINAL_STATUSES: frozenset[str] = frozenset(
    {"succeeded", "partial", "failed", "cancelled"}
)


# ── Specs (declarative, immutable inputs) ────────────────────────────────────


class WorkItemSpec(BaseModel):
    """One unit of delegated work: a *worker* (AgentSpec name) plus its *brief*.

    The brief is the full instruction handed to the sub-agent — it is recorded
    verbatim on the run and in events so a UI can show exactly how each
    sub-agent was directed.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    worker: str = Field(min_length=1, description="key into WorkflowSpec.workers")
    brief: str = Field(min_length=1)


class PhaseSpec(BaseModel):
    """An ordered stage whose work items run concurrently (fan-out + join).

    ``on_failure`` decides what a failed item does to the run: ``continue``
    isolates the failure (phase joins, reports partial), ``abort`` fails the
    workflow at the phase join. Either way a phase where *every* item failed
    fails the workflow — there is nothing left to synthesize from.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    items: list[WorkItemSpec] = Field(min_length=1)
    on_failure: Literal["continue", "abort"] = "continue"


class SynthesisSpec(BaseModel):
    """The join step: one worker reads the collected results and synthesizes.

    The synthesis brief is built by the engine from the *instruction* plus each
    work item's worker/brief/status/result-or-error — never from tool logs or
    other execution noise.
    """

    model_config = ConfigDict(extra="forbid")

    worker: str = Field(min_length=1, description="key into WorkflowSpec.workers")
    instruction: str = Field(min_length=1)


class WorkflowSpec(BaseModel):
    """Declarative definition of a multi-agent workflow.

    ``workers`` declares the sub-agent roster; phases and synthesis reference
    workers by name. Dangling references fail validation loudly — a workflow
    must never silently "delegate" to a worker that does not exist.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    workers: dict[str, AgentSpec] = Field(min_length=1)
    phases: list[PhaseSpec] = Field(min_length=1)
    synthesis: SynthesisSpec | None = None
    max_concurrency: int = Field(default=4, ge=1)

    @model_validator(mode="after")
    def _check_references(self) -> WorkflowSpec:
        seen_items: set[str] = set()
        for phase in self.phases:
            for item in phase.items:
                if item.worker not in self.workers:
                    raise ValueError(
                        f"workflow {self.name!r} phase {phase.name!r} item {item.id!r} "
                        f"references unknown worker {item.worker!r}"
                    )
                if item.id in seen_items:
                    raise ValueError(
                        f"workflow {self.name!r} has duplicate work item id {item.id!r}"
                    )
                seen_items.add(item.id)
        if self.synthesis is not None and self.synthesis.worker not in self.workers:
            raise ValueError(
                f"workflow {self.name!r} synthesis references unknown worker "
                f"{self.synthesis.worker!r}"
            )
        return self


# ── Run state (auditable execution records) ──────────────────────────────────


class WorkItemRun(BaseModel):
    """Execution record of one work item — one sub-agent run.

    Carries the full audit trail for "how was this sub-agent directed and what
    came back": worker id, verbatim brief, minted agent run id, result, error,
    artifacts and evidence. Never stores secrets — briefs and results are
    user-facing content by construction.
    """

    item_id: str
    phase: str
    worker: str
    brief: str
    status: WorkItemStatus = "planned"
    run_id: str = ""
    result: str = ""
    error: str = ""
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    # Token accounting for this item's agent run (promptTokens /
    # completionTokens / totalTokens). Empty = unknown, e.g. the worker
    # raised before reporting — an honest gap, never fabricated zeros.
    usage: dict[str, int] = Field(default_factory=dict)
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def duration_ms(self) -> int | None:
        if self.started_at is None or self.finished_at is None:
            return None
        return int((self.finished_at - self.started_at).total_seconds() * 1000)


class PhaseRun(BaseModel):
    """Execution record of one phase: its items and the joined outcome."""

    name: str
    status: PhaseStatus = "pending"
    items: list[WorkItemRun] = Field(default_factory=list)

    @property
    def succeeded_items(self) -> list[WorkItemRun]:
        return [item for item in self.items if item.status == "succeeded"]

    @property
    def failed_items(self) -> list[WorkItemRun]:
        return [item for item in self.items if item.status == "failed"]


class WorkflowRun(BaseModel):
    """Execution record of a whole workflow run (the audit snapshot)."""

    id: str
    workflow: str
    status: WorkflowStatus = "queued"
    phases: list[PhaseRun] = Field(default_factory=list)
    synthesis_result: str = ""
    synthesis_run_id: str = ""
    synthesis_usage: dict[str, int] = Field(default_factory=dict)
    error: str = ""
    started_at: datetime | None = None
    finished_at: datetime | None = None

    def usage_total(self) -> dict[str, int]:
        """Workflow-level token total: every item's usage plus synthesis.

        Items with unknown usage (empty dict) contribute nothing — the total
        is a floor over what was actually reported, not an estimate.
        """
        prompt = 0
        completion = 0
        for phase in self.phases:
            for item in phase.items:
                prompt += int(item.usage.get("promptTokens", 0) or 0)
                completion += int(item.usage.get("completionTokens", 0) or 0)
        prompt += int(self.synthesis_usage.get("promptTokens", 0) or 0)
        completion += int(self.synthesis_usage.get("completionTokens", 0) or 0)
        return {
            "promptTokens": prompt,
            "completionTokens": completion,
            "totalTokens": prompt + completion,
        }

    def summary(self) -> dict[str, Any]:
        """Projection-friendly snapshot: statuses and audit fields, no noise."""
        return {
            "workflow_run_id": self.id,
            "workflow": self.workflow,
            "status": self.status,
            "error": self.error,
            "usage": self.usage_total(),
            "phases": [
                {
                    "name": phase.name,
                    "status": phase.status,
                    "items": [
                        {
                            "item_id": item.item_id,
                            "worker": item.worker,
                            "status": item.status,
                            "run_id": item.run_id,
                            "error": item.error,
                            "usage": item.usage,
                            "duration_ms": item.duration_ms,
                        }
                        for item in phase.items
                    ],
                }
                for phase in self.phases
            ],
        }


# ── Events (the streamable wire vocabulary) ──────────────────────────────────

EVENT_WORKFLOW_STARTED = "workflow_started"
EVENT_PHASE_STARTED = "phase_started"
EVENT_WORK_ITEM_STARTED = "work_item_started"
EVENT_AGENT_RUN_STARTED = "agent_run_started"
EVENT_AGENT_TOOL_CALL = "agent_tool_call"
EVENT_AGENT_TOOL_RESULT = "agent_tool_result"
EVENT_AGENT_RUN_PROGRESS = "agent_run_progress"
# Live final-answer tokens from a streaming-enabled agent run ({scope, worker,
# run_id, content}). Additive paint only: the terminal frame stays the
# authoritative text and the deltas' concatenation equals it. Today only the
# synthesis run streams (WorkflowEngine stream_synthesis_deltas opt-in).
EVENT_AGENT_RUN_DELTA = "agent_run_delta"
EVENT_AGENT_RUN_COMPLETED = "agent_run_completed"
EVENT_PHASE_COMPLETED = "phase_completed"
EVENT_SYNTHESIS_STARTED = "synthesis_started"
EVENT_WORKFLOW_COMPLETED = "workflow_completed"
EVENT_WORKFLOW_FAILED = "workflow_failed"
# Synthesized by the run registry (never the engine — see WorkflowStatus note):
# {reason, usage, summary}. usage is a FLOOR: in-flight items killed by the
# cancel never reported theirs (the ledger truth stays in llm_call_costs).
EVENT_WORKFLOW_CANCELLED = "workflow_cancelled"

WORKFLOW_EVENT_TYPES: tuple[str, ...] = (
    EVENT_WORKFLOW_STARTED,
    EVENT_PHASE_STARTED,
    EVENT_WORK_ITEM_STARTED,
    EVENT_AGENT_RUN_STARTED,
    EVENT_AGENT_TOOL_CALL,
    EVENT_AGENT_TOOL_RESULT,
    EVENT_AGENT_RUN_PROGRESS,
    EVENT_AGENT_RUN_DELTA,
    EVENT_AGENT_RUN_COMPLETED,
    EVENT_PHASE_COMPLETED,
    EVENT_SYNTHESIS_STARTED,
    EVENT_WORKFLOW_COMPLETED,
    EVENT_WORKFLOW_FAILED,
    EVENT_WORKFLOW_CANCELLED,
)

# Ordering invariants (stable across runs, even with concurrent items):
#   workflow_started precedes everything; phase_started precedes that phase's
#   item events; each item emits work_item_started → agent_run_started →
#   ((agent_tool_call → agent_tool_result) | agent_run_progress)* →
#   agent_run_completed in that order; phase_completed follows all of its
#   items' agent_run_completed; synthesis_started (if any) follows the last
#   phase_completed, and any synthesis tool/progress events fall between
#   synthesis_started and the terminal event; exactly one terminal event
#   (workflow_completed XOR workflow_failed XOR workflow_cancelled — the last
#   synthesized by the run registry, never the engine) comes last. A stream
#   that ends with no terminal frame remains an abnormal end (engine crash
#   emits a bare `error` frame). Tool and progress events appear ONLY on the
#   streamed (observer) path — workers that declared tool_sets, plus the
#   synthesis run when the engine streams deltas; other
#   pure-reasoning runs emit none (their usage arrives once, on
#   agent_run_completed). agent_run_progress carries the run's CUMULATIVE
#   usage after each completed reasoning turn (one LLM call), not a delta —
#   a dropped frame never corrupts the figure. agent_run_delta frames appear
#   only when the engine was built with stream_synthesis_deltas, only between
#   synthesis_started and the terminal event, and are pure live paint: their
#   concatenation equals the terminal workflow_completed `result`, so a
#   consumer that drops every one of them still renders the same final text.


class WorkflowEvent(BaseModel):
    """One orchestration event: a type from the vocabulary above plus payload."""

    type: str
    workflow_run_id: str
    data: dict[str, Any] = Field(default_factory=dict)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
