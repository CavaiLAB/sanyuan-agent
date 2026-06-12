"""Multi-agent orchestration — delegate work to worker agents.

The first building block is *sub-agent delegation*: a worker agent declared as
an :class:`neuroagent.config.AgentSpec` runs in an isolated state seeded only
with the delegated task, and ``make_subagent_tool`` exposes that as an
LLM-callable tool so an orchestrator agent can hand work off like any other
tool call.
"""

from .subagent import make_subagent_tool, run_subagent

__all__ = ["make_subagent_tool", "run_subagent"]
