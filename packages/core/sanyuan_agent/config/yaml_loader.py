"""Parse a YAML agent definition into an :class:`AgentSpec`."""

from __future__ import annotations

from pathlib import Path

import yaml

from .spec import AgentSpec


def load_agent_from_yaml(path: str | Path) -> AgentSpec:
    """Load and validate an :class:`AgentSpec` from a YAML file.

    Raises ``pydantic.ValidationError`` when the file is structurally valid YAML
    but does not satisfy the spec (missing/extra/mistyped fields).
    """
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"agent YAML must be a mapping, got {type(data).__name__}")
    return AgentSpec(**data)
