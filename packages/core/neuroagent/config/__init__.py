"""Declarative agent configuration — build an agent from a spec, not from code.

Fulfils the SAD §8.2 ``build_lead_agent_from_config`` placeholder: an Agent can
be declared as data (a YAML file or dict) and instantiated by the loader,
instead of being hand-wired at each ``build_simple_agent`` call site.
"""

from .loader import LoadedAgent, load_agent_from_spec
from .spec import AgentSpec
from .yaml_loader import load_agent_from_yaml

__all__ = [
    "AgentSpec",
    "LoadedAgent",
    "load_agent_from_spec",
    "load_agent_from_yaml",
]
