"""mobile_use (PR8): provider shell over an injected MobileUseClient.

Pins:

1. provide() yields exactly one ``mobile_use`` tool with the right metadata
   (category="mobile", side_effect="external", policy_hint="confirm") and a
   clean (non-degenerate) schema.
2. The tool forwards its args to the injected client and returns a structured
   MobileUseResult shape - never a raw device handle or path.
3. It registers + executes through a ToolRegistry and is selectable through
   load_agent_from_spec (providers + tools / tool_sets).

No real device / app / desktop / agent-link / network / secret is touched.
"""

from __future__ import annotations

from typing import Any

from neuroagent.config import AgentSpec, load_agent_from_spec
from neuroagent.config.loader import _resolve_tools
from neuroagent.models import LLMProvider, LLMProviderRegistry
from neuroagent.tools import (
    FakeMobileUseClient,
    MobileUseClient,
    MobileUseResult,
    MobileUseToolProvider,
)
from neuroagent.tools.provider import ToolProvider, register_provider
from neuroagent.tools.registry import ToolRegistry

_RESULT_KEYS = {"status", "summary", "actions", "app", "session_id"}


class _RecordingClient:
    """Records run() calls; returns a fixed structured result (no device)."""

    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    async def run(self, *, instruction: str, app: str, session_id: str) -> MobileUseResult:
        self.calls.append({"instruction": instruction, "app": app, "session_id": session_id})
        return MobileUseResult(
            status="completed",
            summary=f"did: {instruction}",
            actions=("tap", "type"),
            app=app,
            session_id=session_id,
        )


# ---------------------------------------------------------------------------
# Protocol + metadata
# ---------------------------------------------------------------------------


class TestProvideMetadata:
    def test_fake_client_satisfies_protocol(self) -> None:
        assert isinstance(FakeMobileUseClient(), MobileUseClient)  # runtime-checkable

    def test_recording_client_satisfies_protocol(self) -> None:
        assert isinstance(_RecordingClient(), MobileUseClient)

    def test_provider_is_runtime_checkable(self) -> None:
        assert isinstance(MobileUseToolProvider(FakeMobileUseClient()), ToolProvider)

    def test_yields_exactly_one_mobile_use_tool(self) -> None:
        specs = MobileUseToolProvider(FakeMobileUseClient()).provide()
        assert [s.name for s in specs] == ["mobile_use"]

    def test_metadata_mobile_external_confirm(self) -> None:
        (spec,) = MobileUseToolProvider(FakeMobileUseClient()).provide()
        assert spec.category == "mobile"
        assert spec.side_effect == "external"
        assert spec.policy_hint == "confirm"
        assert spec.capabilities == ("mobile", "operate")
        assert spec.artifact_output is False
        assert spec.required_secrets == ()  # shell declares no secrets

    def test_schema_is_non_degenerate(self) -> None:
        (spec,) = MobileUseToolProvider(FakeMobileUseClient()).provide()
        props = spec.schema["parameters"]["properties"]
        assert set(props) == {"instruction", "app", "session_id"}
        assert "arguments" not in props
        assert spec.schema["parameters"]["required"] == ["instruction"]  # others default


# ---------------------------------------------------------------------------
# Execution: args forwarded, structured result, no bare handle/path
# ---------------------------------------------------------------------------


class TestExecution:
    async def test_forwards_args_and_returns_structured_result(self) -> None:
        client = _RecordingClient()
        (spec,) = MobileUseToolProvider(client).provide()

        result = await spec.callable(
            instruction="open settings", app="com.example", session_id="s1"
        )

        assert client.calls == [
            {"instruction": "open settings", "app": "com.example", "session_id": "s1"}
        ]
        assert set(result) == _RESULT_KEYS
        assert result["status"] == "completed"
        assert result["app"] == "com.example"
        assert result["session_id"] == "s1"
        assert result["actions"] == ("tap", "type")
        # never a raw device handle or local path
        assert "path" not in result
        assert "handle" not in result
        assert "device" not in result

    async def test_defaults_app_and_session(self) -> None:
        client = _RecordingClient()
        (spec,) = MobileUseToolProvider(client).provide()
        await spec.callable(instruction="go home")
        assert client.calls == [
            {"instruction": "go home", "app": "", "session_id": "default"}
        ]

    async def test_fake_client_is_deterministic(self) -> None:
        # No randomness / network: same inputs -> identical result.
        a = FakeMobileUseClient()
        b = FakeMobileUseClient()
        ra = await a.run(instruction="x", app="app", session_id="s")
        rb = await b.run(instruction="x", app="app", session_id="s")
        assert ra == rb
        assert ra.status == "completed"

    async def test_register_provider_then_execute(self) -> None:
        registry = ToolRegistry()
        names = register_provider(registry, MobileUseToolProvider(FakeMobileUseClient()))

        assert names == ["mobile_use"]
        result = await registry.execute("mobile_use", '{"instruction": "tap login"}')
        assert result["status"] == "completed"
        assert "tap login" in result["summary"]


# ---------------------------------------------------------------------------
# Selectable through the loader (provider path)
# ---------------------------------------------------------------------------


def _llm_registry() -> LLMProviderRegistry:
    reg = LLMProviderRegistry()
    reg.register(
        LLMProvider(
            name="higress",
            provider_kind="openai_compatible",
            api_base="http://higress.test/v1",
            api_key_env="NEUROAGENT_LLM_API_KEY",
            default_model="openai/deepseek-v4-flash",
        )
    )
    return reg


def _spec(**over: Any) -> AgentSpec:
    base = {
        "name": "mobile-agent",
        "system_prompt": "You operate a phone.",
        "llm_provider": "higress",
    }
    base.update(over)
    return AgentSpec(**base)


class TestLoaderSelection:
    def test_candidate_not_auto_enabled(self) -> None:
        providers = {"mobile": MobileUseToolProvider(FakeMobileUseClient())}
        spec = _spec(providers=["mobile"])  # referenced, nothing selected
        assert _resolve_tools(spec, ToolRegistry(), None, providers) == []

    def test_selected_via_tools(self) -> None:
        providers = {"mobile": MobileUseToolProvider(FakeMobileUseClient())}
        spec = _spec(providers=["mobile"], tools=["mobile_use"])
        resolved = _resolve_tools(spec, ToolRegistry(), None, providers)
        assert [t.__name__ for t in resolved] == ["mobile_use"]

    def test_selected_via_tool_set(self) -> None:
        providers = {"mobile": MobileUseToolProvider(FakeMobileUseClient())}
        sets = {"phone": ["mobile_use"]}
        spec = _spec(providers=["mobile"], tool_sets=["phone"])
        resolved = _resolve_tools(spec, ToolRegistry(), sets, providers)
        assert [t.__name__ for t in resolved] == ["mobile_use"]

    def test_load_agent_builds_graph(self) -> None:
        providers = {"mobile": MobileUseToolProvider(FakeMobileUseClient())}
        spec = _spec(providers=["mobile"], tools=["mobile_use"])
        loaded = load_agent_from_spec(
            spec,
            tool_registry=ToolRegistry(),
            llm_registry=_llm_registry(),
            tool_providers=providers,
        )
        assert loaded.graph is not None
