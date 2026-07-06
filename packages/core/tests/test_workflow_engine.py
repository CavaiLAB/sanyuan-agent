"""Workflow engine tests — fan-out/join, failure isolation, synthesis, honesty.

Covers the orchestration contract: two workers in a phase really run
concurrently and join; a failed worker is isolated and reported as partial;
synthesis reads only worker results (never evidence/logs); event order obeys
the documented invariants; and delegation is never faked — no runner, a
broken runner, or an empty answer all surface as loud failures.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import litellm
import pytest
from neuroagent.config import AgentSpec
from neuroagent.models import LLMProvider, LLMProviderRegistry
from neuroagent.orchestration import (
    EVENT_AGENT_RUN_COMPLETED,
    EVENT_AGENT_RUN_STARTED,
    EVENT_PHASE_COMPLETED,
    EVENT_PHASE_STARTED,
    EVENT_SYNTHESIS_STARTED,
    EVENT_WORK_ITEM_STARTED,
    EVENT_WORKFLOW_COMPLETED,
    EVENT_WORKFLOW_FAILED,
    EVENT_WORKFLOW_STARTED,
    PhaseSpec,
    SubagentRunner,
    SynthesisSpec,
    WorkerResult,
    WorkerToolBinding,
    WorkflowEngine,
    WorkflowEvent,
    WorkflowSpec,
    WorkItemSpec,
)
from neuroagent.sandbox.abc import SandboxResult, SandboxSession
from neuroagent.tools.bash import BashToolProvider
from neuroagent.tools.catalog import DEFAULT_TOOL_SETS
from neuroagent.tools.file_ops import FileOpsToolProvider
from neuroagent.tools.registry import ToolRegistry

# ── Helpers ──────────────────────────────────────────────────────────────────


def _worker_spec(name: str = "researcher") -> AgentSpec:
    return AgentSpec(
        name=name,
        system_prompt="You are a focused worker. Answer concisely.",
        llm_provider="higress",
    )


def _two_item_spec(
    *,
    on_failure: str = "continue",
    synthesis: bool = False,
) -> WorkflowSpec:
    return WorkflowSpec(
        name="research",
        workers={
            "alpha": _worker_spec("alpha"),
            "beta": _worker_spec("beta"),
            "editor": _worker_spec("editor"),
        },
        phases=[
            PhaseSpec(
                name="investigate",
                on_failure=on_failure,  # type: ignore[arg-type]
                items=[
                    WorkItemSpec(id="item-a", worker="alpha", brief="investigate topic A"),
                    WorkItemSpec(id="item-b", worker="beta", brief="investigate topic B"),
                ],
            )
        ],
        synthesis=(
            SynthesisSpec(worker="editor", instruction="Merge the findings into one report.")
            if synthesis
            else None
        ),
    )


class RecordingRunner:
    """Deterministic fake worker: outcome keyed by worker name.

    An Exception outcome is raised; a str outcome is the answer. Evidence
    always carries a noise marker so tests can prove synthesis never sees it.
    """

    def __init__(self, outcomes: dict[str, str | Exception]) -> None:
        self.outcomes = outcomes
        self.calls: list[tuple[str, str, str]] = []  # (worker, brief, run_id)

    async def run(self, spec: AgentSpec, brief: str, *, run_id: str) -> WorkerResult:
        self.calls.append((spec.name, brief, run_id))
        outcome = self.outcomes[spec.name]
        if isinstance(outcome, Exception):
            raise outcome
        return WorkerResult(
            answer=outcome,
            run_id=run_id,
            evidence={"tool_log": f"NOISY_TOOL_LOG for {spec.name}"},
        )


def _types(events: list[WorkflowEvent]) -> list[str]:
    return [event.type for event in events]


def _events_of(events: list[WorkflowEvent], event_type: str) -> list[WorkflowEvent]:
    return [event for event in events if event.type == event_type]


# ── Spec validation ──────────────────────────────────────────────────────────


class TestWorkflowSpecValidation:
    def test_unknown_item_worker_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown worker 'ghost'"):
            WorkflowSpec(
                name="w",
                workers={"alpha": _worker_spec("alpha")},
                phases=[
                    PhaseSpec(
                        name="p",
                        items=[WorkItemSpec(id="i1", worker="ghost", brief="do it")],
                    )
                ],
            )

    def test_unknown_synthesis_worker_rejected(self) -> None:
        with pytest.raises(ValueError, match="synthesis references unknown worker"):
            WorkflowSpec(
                name="w",
                workers={"alpha": _worker_spec("alpha")},
                phases=[
                    PhaseSpec(
                        name="p",
                        items=[WorkItemSpec(id="i1", worker="alpha", brief="do it")],
                    )
                ],
                synthesis=SynthesisSpec(worker="ghost", instruction="merge"),
            )

    def test_duplicate_item_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate work item id"):
            WorkflowSpec(
                name="w",
                workers={"alpha": _worker_spec("alpha")},
                phases=[
                    PhaseSpec(
                        name="p1",
                        items=[WorkItemSpec(id="dup", worker="alpha", brief="one")],
                    ),
                    PhaseSpec(
                        name="p2",
                        items=[WorkItemSpec(id="dup", worker="alpha", brief="two")],
                    ),
                ],
            )


# ── Engine honesty: no runner → no pretend delegation ────────────────────────


class TestNoFakeDelegation:
    def test_engine_requires_runner(self) -> None:
        with pytest.raises(ValueError, match="requires a WorkerRunner"):
            WorkflowEngine(runner=None)  # type: ignore[arg-type]

    def test_subagent_runner_requires_registries(self) -> None:
        with pytest.raises(ValueError, match="refusing to construct"):
            SubagentRunner(tool_registry=None, llm_registry=None)  # type: ignore[arg-type]

    def test_subagent_runner_rejects_smuggled_run_id(self) -> None:
        with pytest.raises(ValueError, match="run_id is minted per work item"):
            SubagentRunner(
                tool_registry=ToolRegistry(),
                llm_registry=LLMProviderRegistry(),
                run_id="fixed",
            )

    async def test_empty_answer_is_a_failure_not_a_success(self) -> None:
        runner = RecordingRunner({"alpha": "   ", "beta": "solid answer"})
        engine = WorkflowEngine(runner=runner)
        run, events = await engine.run_to_completion(_two_item_spec())

        item_a = run.phases[0].items[0]
        assert item_a.status == "failed"
        assert item_a.error == "worker returned no final answer"
        assert run.status == "partial"
        assert _types(events)[-1] == EVENT_WORKFLOW_COMPLETED

    async def test_broken_delegate_surfaces_as_workflow_failure(self) -> None:
        # A production runner whose registries cannot resolve the worker's
        # provider must fail the items — never report a fabricated success.
        runner = SubagentRunner(
            tool_registry=ToolRegistry(),
            llm_registry=LLMProviderRegistry(),  # empty: provider lookup raises
        )
        engine = WorkflowEngine(runner=runner)
        run, events = await engine.run_to_completion(_two_item_spec())

        assert run.status == "failed"
        assert all(item.status == "failed" for item in run.phases[0].items)
        assert _types(events)[-1] == EVENT_WORKFLOW_FAILED
        assert EVENT_WORKFLOW_COMPLETED not in _types(events)
        assert run.synthesis_result == ""


# ── Fan-out / join ───────────────────────────────────────────────────────────


class TestFanOutJoin:
    async def test_leaked_cancellation_is_a_failure_not_a_success(self) -> None:
        # A CancelledError leaking out of the runner (e.g. an httpx timeout
        # cancel scope) is a BaseException — the item must still end terminal
        # and honest, never let the phase join fabricate a success.
        class LeakyRunner:
            async def run(self, spec: AgentSpec, brief: str, *, run_id: str) -> WorkerResult:
                if spec.name == "alpha":
                    raise asyncio.CancelledError("leaked cancel scope")
                return WorkerResult(answer=f"done:{spec.name}", run_id=run_id)

        engine = WorkflowEngine(runner=LeakyRunner())
        run, events = await asyncio.wait_for(
            engine.run_to_completion(_two_item_spec()), timeout=5.0
        )

        item_a, item_b = run.phases[0].items
        assert item_a.status == "failed"
        assert "worker crashed with CancelledError" in item_a.error
        assert item_b.status == "succeeded"
        assert run.phases[0].status == "partial"
        assert run.status == "partial"

        completed_a = [
            event
            for event in _events_of(events, EVENT_AGENT_RUN_COMPLETED)
            if event.data["item_id"] == "item-a"
        ]
        assert completed_a and completed_a[0].data["status"] == "failed"
        phase_completed = _events_of(events, EVENT_PHASE_COMPLETED)[0]
        assert phase_completed.data["failed"] == 1

    async def test_closing_the_event_stream_cancels_inflight_workers(self) -> None:
        started = asyncio.Event()
        cancelled = asyncio.Event()

        class HangingRunner:
            async def run(self, spec: AgentSpec, brief: str, *, run_id: str) -> WorkerResult:
                started.set()
                try:
                    await asyncio.Event().wait()  # hangs until cancelled
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
                return WorkerResult(answer="unreachable", run_id=run_id)

        engine = WorkflowEngine(runner=HangingRunner())
        stream = engine.execute(_two_item_spec())
        # Drain until delegation has been announced, then wait for a worker
        # to really be inside the runner before closing the stream.
        async for event in stream:
            if event.type == EVENT_AGENT_RUN_STARTED:
                break
        await asyncio.wait_for(started.wait(), timeout=5.0)
        await stream.aclose()
        await asyncio.wait_for(cancelled.wait(), timeout=5.0)

    async def test_two_workers_run_concurrently_and_join(self) -> None:
        # A barrier only releases when both workers are inside run() at the
        # same time — sequential execution would deadlock (and time out).
        barrier = asyncio.Barrier(2)

        class BarrierRunner:
            async def run(self, spec: AgentSpec, brief: str, *, run_id: str) -> WorkerResult:
                await barrier.wait()
                return WorkerResult(answer=f"done:{spec.name}", run_id=run_id)

        engine = WorkflowEngine(runner=BarrierRunner())
        run, events = await asyncio.wait_for(
            engine.run_to_completion(_two_item_spec()), timeout=5.0
        )

        assert run.status == "succeeded"
        assert {item.result for item in run.phases[0].items} == {"done:alpha", "done:beta"}
        completed = _events_of(events, EVENT_PHASE_COMPLETED)[0]
        assert completed.data["succeeded"] == 2
        assert completed.data["failed"] == 0

    async def test_failure_is_isolated_and_reported_as_partial(self) -> None:
        runner = RecordingRunner(
            {"alpha": RuntimeError("worker exploded"), "beta": "beta findings"}
        )
        engine = WorkflowEngine(runner=runner)
        run, events = await engine.run_to_completion(_two_item_spec())

        item_a, item_b = run.phases[0].items
        assert item_a.status == "failed"
        assert "worker exploded" in item_a.error
        assert item_b.status == "succeeded"
        assert item_b.result == "beta findings"

        assert run.phases[0].status == "partial"
        assert run.status == "partial"
        assert _types(events)[-1] == EVENT_WORKFLOW_COMPLETED
        completed = _events_of(events, EVENT_WORKFLOW_COMPLETED)[0]
        assert completed.data["status"] == "partial"

        failed_run_events = [
            event
            for event in _events_of(events, EVENT_AGENT_RUN_COMPLETED)
            if event.data["item_id"] == "item-a"
        ]
        assert failed_run_events[0].data["status"] == "failed"
        assert "worker exploded" in failed_run_events[0].data["error"]

    async def test_all_items_failing_fails_the_workflow(self) -> None:
        runner = RecordingRunner({"alpha": RuntimeError("boom-a"), "beta": RuntimeError("boom-b")})
        engine = WorkflowEngine(runner=runner)
        run, events = await engine.run_to_completion(_two_item_spec())

        assert run.status == "failed"
        assert "2/2 work items failed" in run.error
        assert _types(events)[-1] == EVENT_WORKFLOW_FAILED

    async def test_abort_policy_fails_workflow_despite_success(self) -> None:
        runner = RecordingRunner({"alpha": RuntimeError("boom"), "beta": "fine"})
        engine = WorkflowEngine(runner=runner)
        run, events = await engine.run_to_completion(_two_item_spec(on_failure="abort"))

        assert run.status == "failed"
        assert _types(events)[-1] == EVENT_WORKFLOW_FAILED
        # The surviving sibling's honest result is still recorded for audit.
        assert run.phases[0].items[1].result == "fine"


# ── Synthesis ────────────────────────────────────────────────────────────────


class TestSynthesis:
    async def test_synthesis_reads_results_only_never_evidence(self) -> None:
        runner = RecordingRunner(
            {
                "alpha": "alpha findings",
                "beta": RuntimeError("beta exploded"),
                "editor": "merged report",
            }
        )
        engine = WorkflowEngine(runner=runner)
        run, events = await engine.run_to_completion(_two_item_spec(synthesis=True))

        assert run.synthesis_result == "merged report"
        synthesis_calls = [call for call in runner.calls if call[0] == "editor"]
        assert len(synthesis_calls) == 1
        brief = synthesis_calls[0][1]
        # Signal: instruction, worker briefs, results, and the honest failure.
        assert "Merge the findings into one report." in brief
        assert "alpha findings" in brief
        assert "investigate topic A" in brief
        assert "[failed] beta exploded" in brief
        # Noise: worker evidence/tool logs must never reach the synthesizer.
        assert "NOISY_TOOL_LOG" not in brief

        started = _events_of(events, EVENT_SYNTHESIS_STARTED)[0]
        assert started.data["worker"] == "editor"
        assert started.data["brief"] == brief
        assert started.data["run_id"] == run.synthesis_run_id

    async def test_synthesis_failure_fails_the_workflow(self) -> None:
        runner = RecordingRunner({"alpha": "a", "beta": "b", "editor": RuntimeError("synth broke")})
        engine = WorkflowEngine(runner=runner)
        run, events = await engine.run_to_completion(_two_item_spec(synthesis=True))

        assert run.status == "failed"
        assert "synthesis failed: synth broke" in run.error
        assert _types(events)[-1] == EVENT_WORKFLOW_FAILED


# ── Usage accounting ─────────────────────────────────────────────────────────


class UsageRunner:
    """Fake worker reporting per-worker token usage (or raising)."""

    def __init__(self, usages: dict[str, dict[str, int] | Exception]) -> None:
        self.usages = usages

    async def run(self, spec: AgentSpec, brief: str, *, run_id: str) -> WorkerResult:
        outcome = self.usages.get(spec.name, {})
        if isinstance(outcome, Exception):
            raise outcome
        return WorkerResult(answer=f"answer:{spec.name}", run_id=run_id, usage=outcome)


class TestUsageAccounting:
    async def test_item_and_synthesis_usage_roll_up(self) -> None:
        runner = UsageRunner(
            {
                # No totalTokens → the engine derives it (10+5=15).
                "alpha": {"promptTokens": 10, "completionTokens": 5},
                "beta": {"promptTokens": 7, "completionTokens": 3, "totalTokens": 10},
                "editor": {"promptTokens": 20, "completionTokens": 8},
            }
        )
        engine = WorkflowEngine(runner=runner)
        run, events = await engine.run_to_completion(_two_item_spec(synthesis=True))

        completed = {
            event.data["item_id"]: event.data
            for event in _events_of(events, EVENT_AGENT_RUN_COMPLETED)
        }
        assert completed["item-a"]["usage"] == {
            "promptTokens": 10,
            "completionTokens": 5,
            "totalTokens": 15,
        }
        assert completed["item-b"]["usage"]["totalTokens"] == 10
        assert run.synthesis_usage == {
            "promptTokens": 20,
            "completionTokens": 8,
            "totalTokens": 28,
        }

        expected_total = {"promptTokens": 37, "completionTokens": 16, "totalTokens": 53}
        assert run.usage_total() == expected_total

        final = _events_of(events, EVENT_WORKFLOW_COMPLETED)[0]
        assert final.data["usage"] == expected_total
        summary = final.data["summary"]
        assert summary["usage"] == expected_total
        item_summaries = {item["item_id"]: item for item in summary["phases"][0]["items"]}
        assert item_summaries["item-a"]["usage"]["totalTokens"] == 15

    async def test_unknown_usage_contributes_zero_not_fabricated(self) -> None:
        # beta crashes before reporting → its usage is honestly empty and the
        # total is a floor over what was reported, not an estimate.
        runner = UsageRunner(
            {
                "alpha": {"promptTokens": 10, "completionTokens": 5},
                "beta": RuntimeError("died before reporting"),
            }
        )
        engine = WorkflowEngine(runner=runner)
        run, events = await engine.run_to_completion(_two_item_spec())

        item_a, item_b = run.phases[0].items
        assert item_a.usage["totalTokens"] == 15
        assert item_b.usage == {}
        assert run.status == "partial"
        assert run.usage_total() == {
            "promptTokens": 10,
            "completionTokens": 5,
            "totalTokens": 15,
        }
        final = _events_of(events, EVENT_WORKFLOW_COMPLETED)[0]
        assert final.data["usage"]["totalTokens"] == 15

    async def test_workflow_failed_still_reports_burned_tokens(self) -> None:
        runner = UsageRunner(
            {
                "alpha": RuntimeError("boom-a"),
                "beta": RuntimeError("boom-b"),
            }
        )
        engine = WorkflowEngine(runner=runner)
        run, events = await engine.run_to_completion(_two_item_spec())

        failed = _events_of(events, EVENT_WORKFLOW_FAILED)[0]
        assert failed.data["usage"] == {
            "promptTokens": 0,
            "completionTokens": 0,
            "totalTokens": 0,
        }
        assert run.usage_total()["totalTokens"] == 0


# ── Event ordering invariants ────────────────────────────────────────────────


class TestEventOrdering:
    async def test_order_invariants_hold_with_concurrency(self) -> None:
        runner = RecordingRunner({"alpha": "a", "beta": "b", "editor": "report"})
        engine = WorkflowEngine(runner=runner)
        _run, events = await engine.run_to_completion(_two_item_spec(synthesis=True))

        types = _types(events)
        assert types[0] == EVENT_WORKFLOW_STARTED
        assert types[-1] == EVENT_WORKFLOW_COMPLETED
        assert types.count(EVENT_WORKFLOW_COMPLETED) + types.count(EVENT_WORKFLOW_FAILED) == 1

        phase_started_idx = types.index(EVENT_PHASE_STARTED)
        phase_completed_idx = types.index(EVENT_PHASE_COMPLETED)
        synthesis_idx = types.index(EVENT_SYNTHESIS_STARTED)
        assert phase_started_idx < phase_completed_idx < synthesis_idx

        # Per item: work_item_started → agent_run_started → agent_run_completed,
        # all inside the phase window.
        for item_id in ("item-a", "item-b"):
            item_indexes = {
                event_type: next(
                    i
                    for i, event in enumerate(events)
                    if event.type == event_type and event.data.get("item_id") == item_id
                )
                for event_type in (
                    EVENT_WORK_ITEM_STARTED,
                    EVENT_AGENT_RUN_STARTED,
                    EVENT_AGENT_RUN_COMPLETED,
                )
            }
            assert (
                phase_started_idx
                < item_indexes[EVENT_WORK_ITEM_STARTED]
                < item_indexes[EVENT_AGENT_RUN_STARTED]
                < item_indexes[EVENT_AGENT_RUN_COMPLETED]
                < phase_completed_idx
            )

    async def test_audit_fields_present_in_events(self) -> None:
        runner = RecordingRunner({"alpha": "a", "beta": "b", "editor": "report"})
        engine = WorkflowEngine(runner=runner)
        run, events = await engine.run_to_completion(_two_item_spec(synthesis=True))

        started = _events_of(events, EVENT_WORK_ITEM_STARTED)
        assert {event.data["brief"] for event in started} == {
            "investigate topic A",
            "investigate topic B",
        }
        for event in _events_of(events, EVENT_AGENT_RUN_STARTED):
            assert event.data["run_id"]
            assert event.data["worker"] in {"alpha", "beta"}
            assert event.data["llm_provider"] == "higress"
        for event in _events_of(events, EVENT_AGENT_RUN_COMPLETED):
            assert event.data["run_id"]
            assert event.data["duration_ms"] is not None
        summary = _events_of(events, EVENT_WORKFLOW_COMPLETED)[0].data["summary"]
        assert summary["workflow_run_id"] == run.id
        item_summaries = summary["phases"][0]["items"]
        assert all(item["run_id"] for item in item_summaries)


# ── Production path: SubagentRunner drives real agent graphs ─────────────────


def _final(content: str) -> Any:
    """An LLM response with no tool_calls → the worker's final answer."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None))],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
    )


@pytest.fixture()
def llm_registry() -> LLMProviderRegistry:
    reg = LLMProviderRegistry()
    reg.register(
        LLMProvider(
            name="higress",
            provider_kind="openai_compatible",
            default_model="openai/deepseek-v4-flash",
        )
    )
    return reg


class TestSubagentRunnerProductionPath:
    async def test_work_items_execute_real_agent_graphs(
        self, llm_registry: LLMProviderRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Only the LLM transport is faked; each work item still runs a full
        # isolated agent graph via run_subagent (build → seed → ainvoke).
        async def fake_acompletion(**kwargs: Any) -> Any:
            user_turn = next(
                message["content"] for message in kwargs["messages"] if message["role"] == "user"
            )
            return _final(f"findings for [{user_turn}]")

        acompletion = AsyncMock(side_effect=fake_acompletion)
        monkeypatch.setattr(litellm, "acompletion", acompletion)

        runner = SubagentRunner(tool_registry=ToolRegistry(), llm_registry=llm_registry)
        engine = WorkflowEngine(runner=runner)
        run, _events = await engine.run_to_completion(_two_item_spec())

        assert run.status == "succeeded"
        assert acompletion.await_count == 2
        item_a, item_b = run.phases[0].items
        assert item_a.result == "findings for [investigate topic A]"
        assert item_b.result == "findings for [investigate topic B]"
        # Usage flows out of the real agent-graph state (3+2 per _final call).
        assert item_a.usage == {"promptTokens": 3, "completionTokens": 2, "totalTokens": 5}
        assert item_b.usage["totalTokens"] == 5
        assert run.usage_total() == {
            "promptTokens": 6,
            "completionTokens": 4,
            "totalTokens": 10,
        }
        # Each delegation was seeded with the worker's own system prompt.
        for call in acompletion.await_args_list:
            assert call.kwargs["messages"][0]["role"] == "system"
            assert "focused worker" in call.kwargs["messages"][0]["content"]


# ── Worker tools: sandbox-bound, isolated, never faked ───────────────────────


def _tool_worker_spec(name: str = "ops") -> AgentSpec:
    return AgentSpec(
        name=name,
        system_prompt="You operate inside the sandbox. Answer concisely.",
        llm_provider="higress",
        tool_sets=["sandbox_workspace"],
        providers=["bash", "filesystem"],
    )


def _tool_workflow(item_count: int = 1) -> WorkflowSpec:
    return WorkflowSpec(
        name="toolwf",
        workers={"ops": _tool_worker_spec()},
        phases=[
            PhaseSpec(
                name="operate",
                items=[
                    WorkItemSpec(id=f"t{i}", worker="ops", brief=f"do task {i}")
                    for i in range(item_count)
                ],
            )
        ],
    )


class FakeSandboxProvider:
    """Duck-typed ISandboxProvider: records every call, executes nothing on the host."""

    def __init__(self) -> None:
        self.shell_calls: list[tuple[str, str]] = []
        self.files: dict[str, str] = {}

    async def run_shell(self, session_id: str, cmd: str) -> SandboxResult:
        self.shell_calls.append((session_id, cmd))
        return SandboxResult(success=True, stdout=f"sandboxed:{cmd}", exit_code=0)

    async def read_file(self, session_id: str, path: str) -> str:
        return self.files.get(path, "")

    async def write_file(self, session_id: str, path: str, content: str) -> None:
        self.files[path] = content

    async def list_dir(self, session_id: str, path: str = ".") -> list[str]:
        return sorted(self.files)

    async def upload_file(self, session_id: str, local_path: str, remote_path: str) -> None:
        raise AssertionError("upload_file must not be reachable from workflow workers")


class RecordingBinder:
    """WorkerToolBinder test double: one FakeSandboxProvider per bind."""

    def __init__(self) -> None:
        self.bind_calls: list[tuple[str, str]] = []  # (worker, run_id)
        self.close_calls: list[str] = []
        self.providers: dict[str, FakeSandboxProvider] = {}

    async def __call__(self, spec: AgentSpec, run_id: str) -> WorkerToolBinding:
        self.bind_calls.append((spec.name, run_id))
        fake = FakeSandboxProvider()
        self.providers[run_id] = fake
        session = SandboxSession(
            id=f"sess-{run_id[:8]}", created_at=datetime.now(UTC), workdir="/workspace"
        )

        async def aclose() -> None:
            self.close_calls.append(run_id)

        return WorkerToolBinding(
            providers={
                "bash": BashToolProvider(fake, session),
                "filesystem": FileOpsToolProvider(fake, session),
            },
            aclose=aclose,
        )


class _MockFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _MockToolCall:
    def __init__(self, tc_id: str, name: str, arguments: str) -> None:
        self.id = tc_id
        self.type = "function"
        self.function = _MockFunction(name, arguments)


def _tool_call_response(name: str, arguments: str) -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="", tool_calls=[_MockToolCall("tc-1", name, arguments)]
                )
            )
        ],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
    )


class TestWorkerTools:
    def _runner(
        self, llm_registry: LLMProviderRegistry, binder: RecordingBinder | None
    ) -> SubagentRunner:
        return SubagentRunner(
            tool_registry=ToolRegistry(),
            llm_registry=llm_registry,
            tool_binder=binder,
            tool_sets=DEFAULT_TOOL_SETS,
        )

    async def test_pure_reasoning_worker_never_touches_binder(
        self, llm_registry: LLMProviderRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(litellm, "acompletion", AsyncMock(return_value=_final("ok")))
        binder = RecordingBinder()
        engine = WorkflowEngine(runner=self._runner(llm_registry, binder))
        run, _events = await engine.run_to_completion(_two_item_spec())

        assert run.status == "succeeded"
        assert binder.bind_calls == []

    async def test_tool_worker_without_binder_fails_not_falls_back(
        self, llm_registry: LLMProviderRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # LLM must never even be reached: the refusal happens at the binder gate.
        acompletion = AsyncMock(return_value=_final("unreachable"))
        monkeypatch.setattr(litellm, "acompletion", acompletion)
        engine = WorkflowEngine(runner=self._runner(llm_registry, binder=None))
        run, events = await engine.run_to_completion(_tool_workflow())

        assert run.status == "failed"
        item = run.phases[0].items[0]
        assert item.status == "failed"
        assert "sandbox boundary" in item.error
        assert acompletion.await_count == 0
        assert _types(events)[-1] == EVENT_WORKFLOW_FAILED

    async def test_each_item_gets_isolated_binding_and_close(
        self, llm_registry: LLMProviderRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(litellm, "acompletion", AsyncMock(return_value=_final("done")))
        binder = RecordingBinder()
        engine = WorkflowEngine(runner=self._runner(llm_registry, binder))
        run, _events = await engine.run_to_completion(_tool_workflow(item_count=2))

        assert run.status == "succeeded"
        run_ids = {item.run_id for phase in run.phases for item in phase.items}
        assert len(run_ids) == 2
        assert {run_id for _, run_id in binder.bind_calls} == run_ids
        assert set(binder.close_calls) == run_ids
        # One distinct provider instance per item — nothing shared.
        assert len(binder.providers) == 2

    async def test_binding_closed_even_when_worker_fails(
        self, llm_registry: LLMProviderRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(litellm, "acompletion", AsyncMock(side_effect=RuntimeError("llm down")))
        binder = RecordingBinder()
        engine = WorkflowEngine(runner=self._runner(llm_registry, binder))
        run, _events = await engine.run_to_completion(_tool_workflow())

        assert run.phases[0].items[0].status == "failed"
        assert len(binder.close_calls) == 1

    async def test_agent_run_started_reports_tool_sets(
        self, llm_registry: LLMProviderRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(litellm, "acompletion", AsyncMock(return_value=_final("done")))
        binder = RecordingBinder()
        engine = WorkflowEngine(runner=self._runner(llm_registry, binder))
        _run, events = await engine.run_to_completion(_tool_workflow())

        started = _events_of(events, EVENT_AGENT_RUN_STARTED)[0]
        assert started.data["tool_sets"] == ["sandbox_workspace"]

        # Pure-reasoning workers honestly report no tool sets.
        _run2, events2 = await engine.run_to_completion(_two_item_spec())
        assert all(
            event.data["tool_sets"] == [] for event in _events_of(events2, EVENT_AGENT_RUN_STARTED)
        )

    async def test_tools_execute_inside_bound_provider(
        self, llm_registry: LLMProviderRegistry
    ) -> None:
        # The decisive boundary proof: the worker's bash call lands in the
        # bound FakeSandboxProvider (session-scoped), never on the host.
        binder = RecordingBinder()
        responses = [
            # The graph registers tools by callable __name__, so the LLM-facing
            # bash tool name is "tool_bash" (same as the chat path).
            _tool_call_response("tool_bash", '{"cmd": "echo sandboxed"}'),
            _final("task complete"),
        ]
        with patch("neuroagent.agents.simple._call_llm", new=AsyncMock(side_effect=responses)):
            engine = WorkflowEngine(runner=self._runner(llm_registry, binder))
            run, _events = await engine.run_to_completion(_tool_workflow())

        item = run.phases[0].items[0]
        assert item.status == "succeeded"
        assert item.result == "task complete"
        fake = binder.providers[item.run_id]
        assert len(fake.shell_calls) == 1
        session_id, cmd = fake.shell_calls[0]
        assert cmd == "echo sandboxed"
        assert session_id.startswith("sess-")


class TestSynthesisWorkerTools:
    def _runner(self, llm_registry: LLMProviderRegistry, binder: RecordingBinder) -> SubagentRunner:
        return SubagentRunner(
            tool_registry=ToolRegistry(),
            llm_registry=llm_registry,
            tool_binder=binder,
            tool_sets=DEFAULT_TOOL_SETS,
        )

    async def test_tool_synthesis_binds_isolated_session_and_reports_tool_sets(
        self, llm_registry: LLMProviderRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(litellm, "acompletion", AsyncMock(return_value=_final("merged")))
        binder = RecordingBinder()
        spec = WorkflowSpec(
            name="toolsynth",
            workers={
                "alpha": _worker_spec("alpha"),  # pure-reasoning work item
                "ops_editor": _tool_worker_spec("ops_editor"),  # tool synthesis
            },
            phases=[
                PhaseSpec(
                    name="p",
                    items=[WorkItemSpec(id="i1", worker="alpha", brief="investigate")],
                )
            ],
            synthesis=SynthesisSpec(worker="ops_editor", instruction="merge in the sandbox"),
        )
        engine = WorkflowEngine(runner=self._runner(llm_registry, binder))
        run, events = await engine.run_to_completion(spec)

        assert run.status == "succeeded"
        assert run.synthesis_result == "merged"
        # Only the synthesis worker declared tools → exactly one binding, on
        # the synthesis run id, and it was closed.
        assert binder.bind_calls == [("ops_editor", run.synthesis_run_id)]
        assert binder.close_calls == [run.synthesis_run_id]

        started = _events_of(events, EVENT_SYNTHESIS_STARTED)[0]
        assert started.data["tool_sets"] == ["sandbox_workspace"]
        assert started.data["run_id"] == run.synthesis_run_id

    async def test_pure_reasoning_synthesis_reports_empty_tool_sets(self) -> None:
        runner = RecordingRunner({"alpha": "a", "beta": "b", "editor": "report"})
        engine = WorkflowEngine(runner=runner)
        _run, events = await engine.run_to_completion(_two_item_spec(synthesis=True))

        started = _events_of(events, EVENT_SYNTHESIS_STARTED)[0]
        assert started.data["tool_sets"] == []
