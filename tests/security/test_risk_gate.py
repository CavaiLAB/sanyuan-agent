"""Tests for high-risk action classification + confirmation gate (PRD §9.1)."""

from sanyuan_agent.agents.tool_executor import tool_executor_node
from sanyuan_agent.security.risk import RiskGate, classify_tool_risk
from sanyuan_agent.tools.registry import ToolRegistry


class TestClassify:
    def test_destructive_bash_is_high(self) -> None:
        assert classify_tool_risk("bash", '{"cmd": "rm -rf /data"}') == "high"
        assert classify_tool_risk("bash", '{"cmd": "git push origin main --force"}') == "high"
        assert classify_tool_risk("bash", '{"cmd": "shutdown now"}') == "high"
        assert classify_tool_risk("bash", '{"cmd": "dd if=/dev/zero of=/dev/sda"}') == "high"

    def test_safe_bash_is_low(self) -> None:
        assert classify_tool_risk("bash", '{"cmd": "ls -la /tmp"}') == "low"

    def test_non_risky_tool_is_low(self) -> None:
        # The pattern only applies to risk-capable tools, not e.g. web_search.
        assert classify_tool_risk("web_search", '{"q": "how to rm -rf safely"}') == "low"


class TestGate:
    def test_high_risk_unconfirmed_requires(self) -> None:
        assert RiskGate().requires_confirmation("bash", '{"cmd":"rm -rf /"}', tc_id="t1") is True

    def test_high_risk_confirmed_passes(self) -> None:
        assert (
            RiskGate().requires_confirmation(
                "bash", '{"cmd":"rm -rf /"}', tc_id="t1", confirmed_ids={"t1"}
            )
            is False
        )

    def test_low_risk_never_requires(self) -> None:
        assert RiskGate().requires_confirmation("bash", '{"cmd":"ls"}', tc_id="t1") is False


def _state(tool_name: str, args: str, tc_id: str = "t1") -> dict:
    return {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": tc_id, "type": "function",
                     "function": {"name": tool_name, "arguments": args}}
                ],
            }
        ]
    }


def _registry_with_bash() -> ToolRegistry:
    reg = ToolRegistry()

    def bash(cmd: str) -> str:
        """run bash"""
        return f"ran: {cmd}"

    reg.register(
        "bash", bash,
        {"name": "bash", "description": "run bash",
         "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}},
    )
    return reg


class TestExecutorGate:
    async def test_high_risk_blocked_without_confirmation(self) -> None:
        out = await tool_executor_node(
            _state("bash", '{"cmd": "rm -rf /tmp/x"}'),
            {"configurable": {"tool_registry": _registry_with_bash(), "risk_gate": RiskGate()}},
        )
        assert "confirmation-required" in out["messages"][0].content

    async def test_high_risk_executes_when_confirmed(self) -> None:
        out = await tool_executor_node(
            _state("bash", '{"cmd": "rm -rf /tmp/x"}'),
            {"configurable": {"tool_registry": _registry_with_bash(),
                              "risk_gate": RiskGate(), "confirmed_tool_calls": {"t1"}}},
        )
        content = out["messages"][0].content
        assert "confirmation-required" not in content
        assert "ran:" in content

    async def test_low_risk_executes(self) -> None:
        out = await tool_executor_node(
            _state("bash", '{"cmd": "ls"}'),
            {"configurable": {"tool_registry": _registry_with_bash(), "risk_gate": RiskGate()}},
        )
        assert "ran:" in out["messages"][0].content

    async def test_no_gate_preserves_existing_behavior(self) -> None:
        out = await tool_executor_node(
            _state("bash", '{"cmd": "rm -rf /tmp/x"}'),
            {"configurable": {"tool_registry": _registry_with_bash()}},
        )
        assert "ran:" in out["messages"][0].content
