"""computer_use tool / provider: desktop counterpart of mobile_use.

Pins the same boundaries as the mobile provider — structured result, no raw
handle, confirm policy hint — plus end-to-end execution through the registry.
"""

from __future__ import annotations

import json

from neuroagent.tools.computer_use import (
    ComputerUseResult,
    ComputerUseToolProvider,
    FakeComputerUseClient,
)
from neuroagent.tools.provider import register_provider
from neuroagent.tools.registry import ToolRegistry


class _RecordingClient:
    """Captures the kwargs the tool forwards, returns a fixed result."""

    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    async def run(self, *, instruction: str, session_id: str) -> ComputerUseResult:
        self.calls.append({"instruction": instruction, "session_id": session_id})
        return ComputerUseResult(
            status="completed",
            summary="did it",
            actions=("a", "b"),
            session_id=session_id,
        )


class TestSpec:
    def test_single_spec_with_confirm_metadata(self) -> None:
        (spec,) = ComputerUseToolProvider(FakeComputerUseClient()).provide()
        assert spec.name == "computer_use"
        assert spec.category == "computer"
        assert spec.capabilities == ("computer", "operate")
        assert spec.side_effect == "external"
        assert spec.policy_hint == "confirm"
        assert spec.required_secrets == ()

    def test_schema_non_degenerate(self) -> None:
        (spec,) = ComputerUseToolProvider(FakeComputerUseClient()).provide()
        props = spec.schema["parameters"]["properties"]
        assert set(props) == {"instruction", "session_id"}
        assert "arguments" not in props
        assert spec.schema["parameters"]["required"] == ["instruction"]
        assert props["instruction"]["description"]  # Args docstring populated


class TestExecution:
    async def test_forwards_args_and_returns_structured_result(self) -> None:
        client = _RecordingClient()
        registry = ToolRegistry()
        register_provider(registry, ComputerUseToolProvider(client))
        result = await registry.execute(
            "computer_use",
            json.dumps({"instruction": "open settings", "session_id": "s1"}),
        )
        assert client.calls == [{"instruction": "open settings", "session_id": "s1"}]
        assert result == {
            "status": "completed",
            "summary": "did it",
            "actions": ("a", "b"),  # asdict preserves the tuple field type
            "session_id": "s1",
        }

    async def test_fake_client_is_deterministic(self) -> None:
        registry = ToolRegistry()
        register_provider(registry, ComputerUseToolProvider(FakeComputerUseClient()))
        result = await registry.execute(
            "computer_use", json.dumps({"instruction": "do x"})
        )
        assert result["status"] == "completed"
        assert "simulated computer_use: do x" in result["summary"]
        assert result["session_id"] == "default"
