"""Workflow engine — concurrent fan-out/join of real sub-agent executions.

The engine walks a :class:`~neuroagent.orchestration.workflow.WorkflowSpec`
phase by phase, runs each phase's work items concurrently through a
:class:`WorkerRunner`, joins the results, and optionally hands the collected
results to a synthesis worker. Execution is streamed as
:class:`~neuroagent.orchestration.workflow.WorkflowEvent` frames.

Delegation is real or it is a failure: the engine refuses to construct
without a runner, and a worker that raises — or returns no answer — yields a
``failed`` work item, never a fabricated success. The production runner
(:class:`SubagentRunner`) executes every work item through
:func:`neuroagent.orchestration.subagent.run_subagent`, i.e. a full isolated
agent-graph run.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator, Iterable
from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import BaseModel, Field

from neuroagent.config import AgentSpec
from neuroagent.models import LLMProviderRegistry
from neuroagent.skills.registry import SkillRegistry
from neuroagent.tools.registry import ToolRegistry

from .subagent import run_subagent_outcome
from .workflow import (
    EVENT_AGENT_RUN_COMPLETED,
    EVENT_AGENT_RUN_STARTED,
    EVENT_PHASE_COMPLETED,
    EVENT_PHASE_STARTED,
    EVENT_SYNTHESIS_STARTED,
    EVENT_WORK_ITEM_STARTED,
    EVENT_WORKFLOW_COMPLETED,
    EVENT_WORKFLOW_FAILED,
    EVENT_WORKFLOW_STARTED,
    PhaseRun,
    WorkflowEvent,
    WorkflowRun,
    WorkflowSpec,
    WorkItemRun,
)


class WorkerResult(BaseModel):
    """What a worker execution hands back to the orchestrator.

    ``answer`` is the worker's final answer (its only channel back — no tool
    logs, no intermediate chatter). ``artifacts``/``evidence`` are optional
    structured audit attachments; ``usage`` is optional token accounting.
    """

    answer: str
    run_id: str = ""
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, int] = Field(default_factory=dict)


class WorkerRunner(Protocol):
    """The execution seam: turns (worker spec, brief) into a real agent run."""

    async def run(self, spec: AgentSpec, brief: str, *, run_id: str) -> WorkerResult: ...


class SubagentRunner:
    """Production runner: one work item = one isolated ``run_subagent`` run.

    Requires live registries up front — constructing without them (or trying
    to smuggle a per-item ``run_id`` through ``agent_kwargs``) raises, so a
    misconfigured workflow fails loudly instead of pretending to delegate.
    """

    def __init__(
        self,
        *,
        tool_registry: ToolRegistry,
        llm_registry: LLMProviderRegistry,
        skill_registry: SkillRegistry | None = None,
        **agent_kwargs: Any,
    ) -> None:
        if tool_registry is None or llm_registry is None:
            raise ValueError(
                "SubagentRunner requires a ToolRegistry and an LLMProviderRegistry — "
                "refusing to construct a runner that cannot really delegate"
            )
        if "run_id" in agent_kwargs:
            raise ValueError("run_id is minted per work item; do not pass it via agent_kwargs")
        self._tool_registry = tool_registry
        self._llm_registry = llm_registry
        self._skill_registry = skill_registry
        self._agent_kwargs = agent_kwargs

    async def run(self, spec: AgentSpec, brief: str, *, run_id: str) -> WorkerResult:
        outcome = await run_subagent_outcome(
            spec,
            brief,
            tool_registry=self._tool_registry,
            llm_registry=self._llm_registry,
            skill_registry=self._skill_registry,
            run_id=run_id,
            **self._agent_kwargs,
        )
        return WorkerResult(answer=outcome.answer, run_id=run_id, usage=outcome.usage)


def build_synthesis_brief(instruction: str, items: Iterable[WorkItemRun]) -> str:
    """Compose the synthesis worker's brief from results only — never logs.

    Each section carries the worker id, the original brief, the outcome status
    and the result (or error). ``evidence`` and any execution noise stay out by
    construction so the synthesizer reasons over signal, not transcripts.
    """
    sections: list[str] = []
    for item in items:
        outcome = item.result if item.status == "succeeded" else f"[failed] {item.error}"
        sections.append(
            f"### {item.worker} · {item.item_id} ({item.status})\n"
            f"Brief: {item.brief}\n"
            f"Result:\n{outcome}"
        )
    return instruction + "\n\n## Worker results\n\n" + "\n\n".join(sections)


def _now() -> datetime:
    return datetime.now(UTC)


def _normalize_usage(usage: dict[str, int] | None) -> dict[str, int]:
    """Coerce worker-reported usage to the camelCase wire shape.

    Empty stays empty (unknown is honest); otherwise ``totalTokens`` is
    derived from prompt+completion when the reporter didn't set it.
    """
    if not usage:
        return {}
    prompt = int(usage.get("promptTokens", 0) or 0)
    completion = int(usage.get("completionTokens", 0) or 0)
    total = int(usage.get("totalTokens", 0) or 0) or (prompt + completion)
    return {"promptTokens": prompt, "completionTokens": completion, "totalTokens": total}


class WorkflowEngine:
    """Drives a WorkflowSpec to completion, streaming WorkflowEvents."""

    def __init__(self, *, runner: WorkerRunner) -> None:
        if runner is None:
            raise ValueError(
                "WorkflowEngine requires a WorkerRunner — a workflow must never "
                "pretend to delegate without one"
            )
        self._runner = runner

    def execute(
        self, spec: WorkflowSpec, *, workflow_run_id: str | None = None
    ) -> AsyncGenerator[WorkflowEvent, None]:
        """Stream the events of one workflow run (terminal event included)."""
        run = self._new_run(spec, workflow_run_id)
        return self._execute(spec, run)

    async def run_to_completion(
        self, spec: WorkflowSpec, *, workflow_run_id: str | None = None
    ) -> tuple[WorkflowRun, list[WorkflowEvent]]:
        """Drain :meth:`execute`; return the final run snapshot and all events."""
        run = self._new_run(spec, workflow_run_id)
        events = [event async for event in self._execute(spec, run)]
        return run, events

    def _new_run(self, spec: WorkflowSpec, workflow_run_id: str | None) -> WorkflowRun:
        return WorkflowRun(
            id=workflow_run_id or uuid.uuid4().hex,
            workflow=spec.name,
            status="queued",
        )

    async def _execute(
        self, spec: WorkflowSpec, run: WorkflowRun
    ) -> AsyncGenerator[WorkflowEvent, None]:
        run.status = "running"
        run.started_at = _now()
        yield self._event(
            run,
            EVENT_WORKFLOW_STARTED,
            {
                "workflow": spec.name,
                "phase_count": len(spec.phases),
                "workers": sorted(spec.workers),
            },
        )

        for index, phase_spec in enumerate(spec.phases):
            phase_run = PhaseRun(
                name=phase_spec.name,
                status="running",
                items=[
                    WorkItemRun(
                        item_id=item.id,
                        phase=phase_spec.name,
                        worker=item.worker,
                        brief=item.brief,
                    )
                    for item in phase_spec.items
                ],
            )
            run.phases.append(phase_run)
            yield self._event(
                run,
                EVENT_PHASE_STARTED,
                {"phase": phase_spec.name, "index": index, "item_count": len(phase_run.items)},
            )

            async for event in self._run_phase(spec, run, phase_run):
                yield event

            # Last line of defence: the join must never derive success from a
            # non-terminal item (e.g. a crash path that skipped both except
            # branches). Anything not terminal counts as failed, loudly.
            for item in phase_run.items:
                if item.status not in ("succeeded", "failed"):
                    item.status = "failed"
                    item.error = "worker exited without a terminal status"
                    item.finished_at = item.finished_at or _now()

            failed = phase_run.failed_items
            succeeded = phase_run.succeeded_items
            if not failed:
                phase_run.status = "succeeded"
            elif succeeded:
                phase_run.status = "partial"
            else:
                phase_run.status = "failed"
            yield self._event(
                run,
                EVENT_PHASE_COMPLETED,
                {
                    "phase": phase_spec.name,
                    "index": index,
                    "status": phase_run.status,
                    "succeeded": len(succeeded),
                    "failed": len(failed),
                },
            )

            # `abort` fails at the phase join (in-flight siblings have already
            # finished); preemptive cancellation is a future concern.
            if phase_run.status == "failed" or (failed and phase_spec.on_failure == "abort"):
                reason = (
                    f"phase {phase_spec.name!r} failed: "
                    f"{len(failed)}/{len(phase_run.items)} work items failed"
                )
                async for event in self._fail(run, reason):
                    yield event
                return

        if spec.synthesis is not None:
            synthesis_brief = build_synthesis_brief(
                spec.synthesis.instruction,
                [item for phase in run.phases for item in phase.items],
            )
            synthesis_run_id = uuid.uuid4().hex
            run.synthesis_run_id = synthesis_run_id
            yield self._event(
                run,
                EVENT_SYNTHESIS_STARTED,
                {
                    "worker": spec.synthesis.worker,
                    "run_id": synthesis_run_id,
                    "brief": synthesis_brief,
                },
            )
            try:
                result = await self._runner.run(
                    spec.workers[spec.synthesis.worker],
                    synthesis_brief,
                    run_id=synthesis_run_id,
                )
                if not result.answer.strip():
                    raise RuntimeError("synthesis worker returned no final answer")
            except Exception as exc:
                async for event in self._fail(run, f"synthesis failed: {exc}"):
                    yield event
                return
            run.synthesis_result = result.answer
            run.synthesis_usage = _normalize_usage(result.usage)

        run.status = (
            "partial" if any(phase.status == "partial" for phase in run.phases) else "succeeded"
        )
        run.finished_at = _now()
        yield self._event(
            run,
            EVENT_WORKFLOW_COMPLETED,
            {
                "status": run.status,
                "result": run.synthesis_result,
                "usage": run.usage_total(),
                "summary": run.summary(),
            },
        )

    async def _run_phase(
        self, spec: WorkflowSpec, run: WorkflowRun, phase_run: PhaseRun
    ) -> AsyncGenerator[WorkflowEvent, None]:
        """Fan the phase's items out concurrently; relay their events in FIFO order.

        Each item task pushes its events to a queue and a ``None`` sentinel
        last, so once every sentinel is seen the queue holds no stragglers.
        """
        queue: asyncio.Queue[WorkflowEvent | None] = asyncio.Queue()
        semaphore = asyncio.Semaphore(spec.max_concurrency)

        async def run_item(item_run: WorkItemRun) -> None:
            try:
                async with semaphore:
                    item_run.status = "running"
                    item_run.started_at = _now()
                    queue.put_nowait(
                        self._event(
                            run,
                            EVENT_WORK_ITEM_STARTED,
                            {
                                "item_id": item_run.item_id,
                                "phase": item_run.phase,
                                "worker": item_run.worker,
                                "brief": item_run.brief,
                            },
                        )
                    )
                    run_id = uuid.uuid4().hex
                    item_run.run_id = run_id
                    worker_spec = spec.workers[item_run.worker]
                    queue.put_nowait(
                        self._event(
                            run,
                            EVENT_AGENT_RUN_STARTED,
                            {
                                "item_id": item_run.item_id,
                                "run_id": run_id,
                                "worker": item_run.worker,
                                "llm_provider": worker_spec.llm_provider,
                                "model": worker_spec.model,
                            },
                        )
                    )
                    try:
                        result = await self._runner.run(worker_spec, item_run.brief, run_id=run_id)
                        # Usage is recorded on both outcomes: an empty answer
                        # still burned the reported tokens.
                        item_run.usage = _normalize_usage(result.usage)
                        if result.answer.strip():
                            item_run.status = "succeeded"
                            item_run.result = result.answer
                            item_run.artifacts = list(result.artifacts)
                            item_run.evidence = dict(result.evidence)
                        else:
                            item_run.status = "failed"
                            item_run.error = "worker returned no final answer"
                    except Exception as exc:
                        item_run.status = "failed"
                        item_run.error = str(exc)
                    except BaseException as exc:
                        # A leaked CancelledError (e.g. from an httpx/anyio
                        # timeout scope inside the runner) must not leave the
                        # item non-terminal — that would let the phase join
                        # fabricate a success. Mark it failed, stay honest on
                        # the wire, then re-raise to keep cancellation intact.
                        item_run.status = "failed"
                        item_run.error = f"worker crashed with {type(exc).__name__}: {exc}"
                        item_run.finished_at = _now()
                        queue.put_nowait(
                            self._event(
                                run,
                                EVENT_AGENT_RUN_COMPLETED,
                                {
                                    "item_id": item_run.item_id,
                                    "run_id": item_run.run_id,
                                    "worker": item_run.worker,
                                    "status": item_run.status,
                                    "result": "",
                                    "error": item_run.error,
                                    "artifacts": [],
                                    "usage": item_run.usage,
                                    "duration_ms": item_run.duration_ms,
                                },
                            )
                        )
                        raise
                    item_run.finished_at = _now()
                    queue.put_nowait(
                        self._event(
                            run,
                            EVENT_AGENT_RUN_COMPLETED,
                            {
                                "item_id": item_run.item_id,
                                "run_id": item_run.run_id,
                                "worker": item_run.worker,
                                "status": item_run.status,
                                "result": item_run.result,
                                "error": item_run.error,
                                "artifacts": item_run.artifacts,
                                "usage": item_run.usage,
                                "duration_ms": item_run.duration_ms,
                            },
                        )
                    )
            finally:
                queue.put_nowait(None)

        tasks = [asyncio.create_task(run_item(item)) for item in phase_run.items]
        finished = 0
        try:
            while finished < len(tasks):
                event = await queue.get()
                if event is None:
                    finished += 1
                    continue
                yield event
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _fail(self, run: WorkflowRun, reason: str) -> AsyncGenerator[WorkflowEvent, None]:
        run.status = "failed"
        run.error = reason
        run.finished_at = _now()
        yield self._event(
            run,
            EVENT_WORKFLOW_FAILED,
            {"error": reason, "usage": run.usage_total(), "summary": run.summary()},
        )

    def _event(self, run: WorkflowRun, event_type: str, data: dict[str, Any]) -> WorkflowEvent:
        return WorkflowEvent(type=event_type, workflow_run_id=run.id, data=data)
