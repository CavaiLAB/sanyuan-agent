"""Multi-agent orchestration — delegate work to worker agents.

Two layers build on the same execution unit:

*Sub-agent delegation* — a worker agent declared as an
:class:`neuroagent.config.AgentSpec` runs in an isolated state seeded only
with the delegated task (:func:`run_subagent`), and ``make_subagent_tool``
exposes that as an LLM-callable tool so an orchestrator agent can hand work
off like any other tool call.

*Workflow orchestration* — a declarative :class:`WorkflowSpec` (phases →
concurrent work items → optional synthesis) executed by
:class:`WorkflowEngine`, where every work item is a real sub-agent run
(:class:`SubagentRunner`) and progress streams as :class:`WorkflowEvent`
frames. The sub-agent is the execution unit; the workflow is the
orchestration topology over those units.
"""

from .subagent import (
    SubagentOutcome,
    ToolEventObserver,
    make_subagent_tool,
    run_subagent,
    run_subagent_outcome,
)
from .workflow import (
    EVENT_AGENT_RUN_COMPLETED,
    EVENT_AGENT_RUN_STARTED,
    EVENT_AGENT_TOOL_CALL,
    EVENT_AGENT_TOOL_RESULT,
    EVENT_PHASE_COMPLETED,
    EVENT_PHASE_STARTED,
    EVENT_SYNTHESIS_STARTED,
    EVENT_WORK_ITEM_STARTED,
    EVENT_WORKFLOW_COMPLETED,
    EVENT_WORKFLOW_FAILED,
    EVENT_WORKFLOW_STARTED,
    WORKFLOW_EVENT_TYPES,
    WORKFLOW_TERMINAL_STATUSES,
    PhaseRun,
    PhaseSpec,
    SynthesisSpec,
    WorkflowEvent,
    WorkflowRun,
    WorkflowSpec,
    WorkItemRun,
    WorkItemSpec,
)
from .workflow_engine import (
    SubagentRunner,
    WorkerResult,
    WorkerRunner,
    WorkerToolBinder,
    WorkerToolBinding,
    WorkflowEngine,
    build_synthesis_brief,
)

__all__ = [
    "EVENT_AGENT_RUN_COMPLETED",
    "EVENT_AGENT_RUN_STARTED",
    "EVENT_AGENT_TOOL_CALL",
    "EVENT_AGENT_TOOL_RESULT",
    "EVENT_PHASE_COMPLETED",
    "EVENT_PHASE_STARTED",
    "EVENT_SYNTHESIS_STARTED",
    "EVENT_WORKFLOW_COMPLETED",
    "EVENT_WORKFLOW_FAILED",
    "EVENT_WORKFLOW_STARTED",
    "EVENT_WORK_ITEM_STARTED",
    "WORKFLOW_EVENT_TYPES",
    "WORKFLOW_TERMINAL_STATUSES",
    "PhaseRun",
    "PhaseSpec",
    "SubagentOutcome",
    "SubagentRunner",
    "SynthesisSpec",
    "ToolEventObserver",
    "WorkItemRun",
    "WorkItemSpec",
    "WorkerResult",
    "WorkerRunner",
    "WorkerToolBinder",
    "WorkerToolBinding",
    "WorkflowEngine",
    "WorkflowEvent",
    "WorkflowRun",
    "WorkflowSpec",
    "build_synthesis_brief",
    "make_subagent_tool",
    "run_subagent",
    "run_subagent_outcome",
]
