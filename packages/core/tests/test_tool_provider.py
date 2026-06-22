"""Tool Provider protocol (PR1): specs, adapters, batch registration, equivalence.

The provider path normalizes tool declaration upstream of ``ToolRegistry`` while
leaving the legacy ``build_*_tool`` / ``registry.register`` paths untouched.
These tests pin three things:

1. ``ToolSpec`` validates its executable core and rejects broken metadata.
2. ``spec_from_callable`` / provider batch registration produce a registry that
   is *equivalent* to the legacy builders' output (same schema, same execution).
3. Descriptive metadata never leaks into the executor's contract.
"""

from __future__ import annotations

from typing import Any

import pytest
from neuroagent.agents.simple import _to_tool_def
from neuroagent.agents.tool_executor import tool_executor_node
from neuroagent.mcp import McpToolDef, make_mcp_tools
from neuroagent.tools.artifact_write import build_artifact_write_tool
from neuroagent.tools.bash import build_bash_tool
from neuroagent.tools.provider import (
    ToolProvider,
    ToolSpec,
    register_provider,
    register_tool_specs,
    spec_from_callable,
    specs_from_callables,
)
from neuroagent.tools.registry import ToolRegistry, callable_to_schema
from neuroagent.tools.web_fetch import build_web_fetch_tool
from neuroagent.tools.web_search import build_web_search_tool


def _echo_schema() -> dict[str, Any]:
    return {
        "name": "echo",
        "description": "Echo text back.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    }


async def echo(text: str) -> str:
    """Echo text back."""
    return f"echo: {text}"


# ---------------------------------------------------------------------------
# ToolSpec validation
# ---------------------------------------------------------------------------


class TestToolSpec:
    def test_minimal_spec_holds_executable_core(self) -> None:
        spec = ToolSpec(name="echo", callable=echo, schema=_echo_schema())
        assert spec.name == "echo"
        assert spec.callable is echo
        assert spec.schema["parameters"]["required"] == ["text"]
        # Metadata defaults are inert.
        assert spec.category == ""
        assert spec.capabilities == ()
        assert spec.side_effect == "read"
        assert spec.required_secrets == ()
        assert spec.artifact_output is False
        assert spec.policy_hint is None

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="name must be non-empty"):
            ToolSpec(name="", callable=echo, schema=_echo_schema())

    def test_non_callable_rejected(self) -> None:
        with pytest.raises(ValueError, match="not callable"):
            ToolSpec(name="x", callable="nope", schema=_echo_schema())  # type: ignore[arg-type]

    def test_non_dict_schema_rejected(self) -> None:
        with pytest.raises(ValueError, match="schema for 'x' must be a dict"):
            ToolSpec(name="x", callable=echo, schema="nope")  # type: ignore[arg-type]

    def test_non_dict_parameters_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"parameters'\] for 'x' must be a dict"):
            ToolSpec(name="x", callable=echo, schema={"name": "x", "parameters": []})

    def test_unknown_policy_hint_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown policy_hint"):
            ToolSpec(name="x", callable=echo, schema=_echo_schema(), policy_hint="ask")

    def test_valid_policy_hints_accepted(self) -> None:
        for hint in ("allow", "deny", "confirm"):
            spec = ToolSpec(name="x", callable=echo, schema=_echo_schema(), policy_hint=hint)
            assert spec.policy_hint == hint


# ---------------------------------------------------------------------------
# spec_from_callable: schema resolution priority
# ---------------------------------------------------------------------------


class TestSpecFromCallable:
    def test_reflects_schema_from_signature(self) -> None:
        spec = spec_from_callable(echo, category="util")
        assert spec.name == "echo"
        assert spec.category == "util"
        assert spec.schema == callable_to_schema(echo)

    def test_explicit_schema_takes_priority(self) -> None:
        custom = _echo_schema()
        spec = spec_from_callable(echo, schema=custom)
        assert spec.schema is custom

    async def test_honours_carried_tool_schema(self) -> None:
        """MCP proxies carry ``__tool_schema__`` — adapter must use it verbatim."""
        weather = McpToolDef(
            name="get-weather",
            description="Get the weather for a city.",
            input_schema={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )

        class FakeClient:
            server_name = "wx"

            async def list_tools(self) -> list[McpToolDef]:
                return [weather]

            async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
                return f"called {name}"

        (proxy,) = await make_mcp_tools(FakeClient())
        spec = spec_from_callable(proxy, category="mcp")

        assert spec.name == "mcp__wx__get-weather"
        # Schema matches what the agent builder would derive — no reflection drift.
        assert spec.schema == _to_tool_def(proxy)["function"]

    def test_explicit_name_overrides(self) -> None:
        spec = spec_from_callable(echo, name="aliased")
        assert spec.name == "aliased"


# ---------------------------------------------------------------------------
# Registration helpers
# ---------------------------------------------------------------------------


class TestRegisterSpecs:
    def test_register_specs_returns_names(self) -> None:
        registry = ToolRegistry()
        names = register_tool_specs(registry, [spec_from_callable(echo)])
        assert names == ["echo"]
        assert registry.has_tool("echo")
        assert registry.get_schema("echo") == callable_to_schema(echo)

    def test_duplicate_name_rejected_by_default(self) -> None:
        registry = ToolRegistry()
        register_tool_specs(registry, [spec_from_callable(echo)])
        with pytest.raises(ValueError, match="already registered"):
            register_tool_specs(registry, [spec_from_callable(echo)])

    def test_overwrite_allows_replacement(self) -> None:
        registry = ToolRegistry()
        register_tool_specs(registry, [spec_from_callable(echo)])
        names = register_tool_specs(registry, [spec_from_callable(echo)], overwrite=True)
        assert names == ["echo"]

    def test_specs_from_callables_shares_metadata(self) -> None:
        async def a(x: str) -> str:
            """A."""
            return x

        async def b(y: str) -> str:
            """B."""
            return y

        specs = specs_from_callables([a, b], category="browser", side_effect="write")
        assert [s.name for s in specs] == ["a", "b"]
        assert all(s.category == "browser" for s in specs)
        assert all(s.side_effect == "write" for s in specs)


class TestRegisterProvider:
    def test_provider_batch_registers(self) -> None:
        class EchoProvider:
            def provide(self) -> list[ToolSpec]:
                return [spec_from_callable(echo, category="util")]

        provider = EchoProvider()
        assert isinstance(provider, ToolProvider)  # runtime-checkable Protocol

        registry = ToolRegistry()
        names = register_provider(registry, provider)
        assert names == ["echo"]
        assert registry.has_tool("echo")


# ---------------------------------------------------------------------------
# Equivalence with legacy build_*_tool paths
# ---------------------------------------------------------------------------


class _FakeArtifactClient:
    async def create_artifact(
        self, tenant_id: str, content_bytes: bytes, content_type: str, filename: str
    ) -> Any:
        from neuroagent.artifacts.client import ArtifactRef

        return ArtifactRef(
            id="fake-id",
            url="https://example.com/fake",
            sha256="abc123",
            filename=filename,
            content_type=content_type,
        )


class _FakeSandboxProvider:
    """Minimal ISandboxProvider stand-in for bash equivalence."""

    async def run_shell(self, session_id: str, cmd: str) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(stdout=f"ran {cmd}", stderr="", exit_code=0)


class TestLegacyEquivalence:
    """A spec built from a legacy builder's callable registers the same schema."""

    def test_web_search_schema_equivalent(self) -> None:
        legacy = ToolRegistry()
        tool = build_web_search_tool(legacy)

        provider_reg = ToolRegistry()
        register_tool_specs(
            provider_reg,
            [spec_from_callable(tool, category="web", capabilities=("search",))],
        )

        assert legacy.get_schema("web_search") == provider_reg.get_schema("web_search")
        assert legacy.tool_defs() == provider_reg.tool_defs()

    def test_web_fetch_schema_equivalent(self) -> None:
        legacy = ToolRegistry()
        tool = build_web_fetch_tool(legacy)

        provider_reg = ToolRegistry()
        register_tool_specs(provider_reg, [spec_from_callable(tool, category="web")])

        assert legacy.get_schema("web_fetch") == provider_reg.get_schema("web_fetch")

    def test_artifact_write_schema_equivalent(self) -> None:
        legacy = ToolRegistry()
        tool = build_artifact_write_tool(legacy, _FakeArtifactClient(), tenant_id="t1")

        provider_reg = ToolRegistry()
        register_tool_specs(
            provider_reg,
            [
                spec_from_callable(
                    tool,
                    category="artifact",
                    side_effect="write",
                    artifact_output=True,
                )
            ],
        )

        assert legacy.get_schema("artifact_write") == provider_reg.get_schema("artifact_write")

    async def test_bash_execution_equivalent(self) -> None:
        from types import SimpleNamespace

        session = SimpleNamespace(id="sess-1")

        # Legacy registry-registered path is the reference for name ("bash") and
        # schema — the bare callable's __name__ is "tool_bash", so the provider
        # path reuses the legacy schema + name to wrap it without touching the
        # production builder.
        legacy = ToolRegistry()
        build_bash_tool(_FakeSandboxProvider(), session, legacy)  # type: ignore[arg-type]

        tool = build_bash_tool(_FakeSandboxProvider(), session)  # type: ignore[arg-type]
        registry = ToolRegistry()
        register_tool_specs(
            registry,
            [
                spec_from_callable(
                    tool,
                    name="bash",
                    schema=legacy.get_schema("bash"),
                    category="sandbox",
                    side_effect="write",
                )
            ],
        )

        assert registry.get_schema("bash") == legacy.get_schema("bash")

        # … and it still executes through the registry.
        result = await registry.execute("bash", '{"cmd": "ls"}')
        assert result["stdout"] == "ran ls"


# ---------------------------------------------------------------------------
# Metadata does not affect the executor contract
# ---------------------------------------------------------------------------


class TestMetadataInertAtExecution:
    async def test_rich_metadata_tool_executes_normally(self) -> None:
        registry = ToolRegistry()
        register_tool_specs(
            registry,
            [
                spec_from_callable(
                    echo,
                    category="multimodal",
                    capabilities=("image", "generate"),
                    side_effect="external",
                    required_secrets=("DASHSCOPE_API_KEY",),
                    artifact_output=True,
                    policy_hint="confirm",
                )
            ],
        )

        state = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "tc-1",
                            "type": "function",
                            "function": {"name": "echo", "arguments": '{"text": "hi"}'},
                        }
                    ],
                }
            ]
        }
        # No tool_policy in config → metadata's policy_hint is *not* enforced.
        result = await tool_executor_node(state, {"configurable": {"tool_registry": registry}})
        assert result["messages"][0].content == "echo: hi"

    def test_schema_excludes_metadata_fields(self) -> None:
        """tool_defs() the LLM sees carries only name/description/parameters."""
        registry = ToolRegistry()
        register_tool_specs(
            registry,
            [spec_from_callable(echo, category="multimodal", artifact_output=True)],
        )
        (tool_def,) = registry.tool_defs()
        assert set(tool_def["function"]) == {"name", "description", "parameters"}
