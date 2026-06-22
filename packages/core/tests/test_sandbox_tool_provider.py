"""SandboxToolProvider (PR11): providerize the sandbox_tool BaseTool path.

build_sandbox_tools yields LangChain StructuredTool objects (not plain
callables). These tests pin that the thin provider preserves their semantics:

1. runtime-checkable ToolProvider + descriptive metadata (category sandbox,
   side_effect external, policy_hint confirm, capabilities incl. shell/python).
2. schema / tool_defs equivalence to legacy build_sandbox_tools (the BaseTool's
   own convert_to_openai_function / convert_to_openai_tool output, byte-for-byte).
3. AgentSpec provider selection (providers + tools / tool_sets).
4. behavior over a fake client: the sandbox is created once and reused, and
   shell / python outputs are unchanged.
5. context identity: set_sandbox_context flows into create_sandbox unchanged.

No real IO / app / sandbox_supervisor imports / secrets - fakes only.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from langchain_core.utils.function_calling import (
    convert_to_openai_function,
    convert_to_openai_tool,
)
from neuroagent.config import AgentSpec, load_agent_from_spec
from neuroagent.config.loader import _resolve_tools
from neuroagent.models import LLMProvider, LLMProviderRegistry
from neuroagent.tools import SandboxToolProvider, build_sandbox_tools
from neuroagent.tools.provider import ToolProvider, register_provider
from neuroagent.tools.registry import ToolRegistry
from neuroagent.tools.sandbox_tool import SandboxClientProto, set_sandbox_context

EXPECTED = {"sandbox_exec_shell", "sandbox_exec_python"}


class _FakeSandboxClient:
    """Records calls; creates a fresh sandbox id per create (no real IO)."""

    def __init__(self) -> None:
        self.create_calls: list[dict[str, str]] = []
        self.exec_calls: list[tuple[Any, ...]] = []

    async def create_sandbox(
        self, template_id: str, tenant_id: str, user_id: str
    ) -> Any:
        self.create_calls.append(
            {"template_id": template_id, "tenant_id": tenant_id, "user_id": user_id}
        )
        return SimpleNamespace(id=f"sb{len(self.create_calls)}")

    async def exec_shell(self, sandbox_id: str, cmd: str, timeout: int = 30) -> Any:
        self.exec_calls.append(("shell", sandbox_id, cmd, timeout))
        return SimpleNamespace(stdout=f"shell:{cmd}", stderr="", exit_code=0)

    async def exec_python(self, sandbox_id: str, code: str, timeout: int = 30) -> Any:
        self.exec_calls.append(("python", sandbox_id, code, timeout))
        return SimpleNamespace(stdout=f"py:{code}", stderr="", exit_code=0)


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
    base = {"name": "coder", "system_prompt": "You run code.", "llm_provider": "higress"}
    base.update(over)
    return AgentSpec(**base)


# ---------------------------------------------------------------------------
# Protocol + metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_fake_client_satisfies_protocol(self) -> None:
        assert isinstance(_FakeSandboxClient(), SandboxClientProto)

    def test_provider_is_runtime_checkable(self) -> None:
        assert isinstance(SandboxToolProvider(_FakeSandboxClient()), ToolProvider)

    def test_yields_the_two_sandbox_tools(self) -> None:
        specs = {s.name: s for s in SandboxToolProvider(_FakeSandboxClient()).provide()}
        assert set(specs) == EXPECTED

    def test_metadata_external_confirm_no_secrets(self) -> None:
        for spec in SandboxToolProvider(_FakeSandboxClient()).provide():
            assert spec.category == "sandbox"
            assert spec.side_effect == "external"
            assert spec.policy_hint == "confirm"
            assert spec.artifact_output is False
            assert spec.required_secrets == ()

    def test_capabilities_tag_runtime(self) -> None:
        specs = {s.name: s for s in SandboxToolProvider(_FakeSandboxClient()).provide()}
        assert specs["sandbox_exec_shell"].capabilities == ("sandbox", "execute", "shell")
        assert specs["sandbox_exec_python"].capabilities == ("sandbox", "execute", "python")

    def test_schema_non_degenerate(self) -> None:
        specs = {s.name: s for s in SandboxToolProvider(_FakeSandboxClient()).provide()}
        props = specs["sandbox_exec_shell"].schema["parameters"]["properties"]
        assert set(props) == {"cmd", "timeout"}
        assert "arguments" not in props
        assert specs["sandbox_exec_shell"].schema["parameters"]["required"] == ["cmd"]
        py_props = specs["sandbox_exec_python"].schema["parameters"]["properties"]
        assert set(py_props) == {"code", "timeout"}


# ---------------------------------------------------------------------------
# Schema / tool_defs equivalence to legacy build_sandbox_tools (the BaseTool)
# ---------------------------------------------------------------------------


class TestEquivalence:
    def test_schema_matches_basetool_openai_function(self) -> None:
        client = _FakeSandboxClient()
        legacy_tools = build_sandbox_tools(client)
        specs = {s.name: s for s in SandboxToolProvider(client).provide()}
        for t in legacy_tools:
            assert specs[t.name].schema == convert_to_openai_function(t)

    def test_registry_tool_defs_match_basetool(self) -> None:
        client = _FakeSandboxClient()
        legacy_defs = sorted(
            (convert_to_openai_tool(t) for t in build_sandbox_tools(client)),
            key=lambda d: d["function"]["name"],
        )
        provider_reg = ToolRegistry()
        names = register_provider(provider_reg, SandboxToolProvider(client))
        assert set(names) == EXPECTED
        assert provider_reg.tool_defs() == legacy_defs

    def test_resolved_callable_carries_basetool_schema_for_build_simple_agent(self) -> None:
        # The load_agent_from_spec -> build_simple_agent path re-derives each
        # tool's schema from the resolved callable via _to_tool_def. Because
        # sandbox_tool.py uses `from __future__ import annotations`, plain
        # reflection mistypes `timeout` as string and drops its default; the
        # provider must carry the BaseTool's schema on the callable so the load
        # path stays equivalent (timeout integer + default 30).
        from neuroagent.agents.simple import _to_tool_def

        client = _FakeSandboxClient()
        by_name = {t.name: t for t in build_sandbox_tools(client)}
        providers = {"sandbox": SandboxToolProvider(client)}
        spec = _spec(
            providers=["sandbox"],
            tools=["sandbox_exec_shell", "sandbox_exec_python"],
        )
        resolved = _resolve_tools(spec, ToolRegistry(), None, providers)

        for fn in resolved:
            expected = convert_to_openai_function(by_name[fn.__name__])
            assert _to_tool_def(fn)["function"] == expected

        # explicit guard on the previously-lost detail
        shell = next(fn for fn in resolved if fn.__name__ == "sandbox_exec_shell")
        timeout = _to_tool_def(shell)["function"]["parameters"]["properties"]["timeout"]
        assert timeout == {"default": 30, "type": "integer"}


# ---------------------------------------------------------------------------
# Behavior over a fake client: create once + reuse, outputs unchanged, context
# ---------------------------------------------------------------------------


class TestBehavior:
    async def test_creates_sandbox_once_and_reuses(self) -> None:
        client = _FakeSandboxClient()
        registry = ToolRegistry()
        register_provider(registry, SandboxToolProvider(client))

        out_shell = await registry.execute("sandbox_exec_shell", '{"cmd": "ls"}')
        out_py = await registry.execute("sandbox_exec_python", '{"code": "print(1)"}')

        # one sandbox created, reused by both calls
        assert len(client.create_calls) == 1
        assert client.exec_calls[0] == ("shell", "sb1", "ls", 30)
        assert client.exec_calls[1] == ("python", "sb1", "print(1)", 30)
        # outputs unchanged from the BaseTool formatting
        assert out_shell == "shell:ls"
        assert out_py == "py:print(1)"

    async def test_stderr_and_exit_code_formatting_preserved(self) -> None:
        class _ErrClient(_FakeSandboxClient):
            async def exec_shell(self, sandbox_id: str, cmd: str, timeout: int = 30) -> Any:
                return SimpleNamespace(stdout="out", stderr="boom", exit_code=2)

        registry = ToolRegistry()
        register_provider(registry, SandboxToolProvider(_ErrClient()))
        out = await registry.execute("sandbox_exec_shell", '{"cmd": "x"}')
        assert out == "out\n[stderr] boom\n[exit_code=2]"

    async def test_set_sandbox_context_flows_into_create(self) -> None:
        set_sandbox_context("tenant-9", "user-9")
        client = _FakeSandboxClient()
        registry = ToolRegistry()
        register_provider(
            registry, SandboxToolProvider(client, template_id="custom-tmpl")
        )
        await registry.execute("sandbox_exec_shell", '{"cmd": "ls"}')
        assert client.create_calls == [
            {"template_id": "custom-tmpl", "tenant_id": "tenant-9", "user_id": "user-9"}
        ]
        # reset context so later tests are not affected
        set_sandbox_context("", "")


# ---------------------------------------------------------------------------
# Selectable through the loader (provider path)
# ---------------------------------------------------------------------------


class TestLoaderSelection:
    def test_candidate_not_auto_enabled(self) -> None:
        providers = {"sandbox": SandboxToolProvider(_FakeSandboxClient())}
        spec = _spec(providers=["sandbox"])
        assert _resolve_tools(spec, ToolRegistry(), None, providers) == []

    def test_selected_via_tools(self) -> None:
        providers = {"sandbox": SandboxToolProvider(_FakeSandboxClient())}
        spec = _spec(providers=["sandbox"], tools=["sandbox_exec_shell", "sandbox_exec_python"])
        resolved = _resolve_tools(spec, ToolRegistry(), None, providers)
        assert [t.__name__ for t in resolved] == ["sandbox_exec_shell", "sandbox_exec_python"]

    def test_selected_via_tool_set(self) -> None:
        providers = {"sandbox": SandboxToolProvider(_FakeSandboxClient())}
        sets = {"coding": ["sandbox_exec_python"]}
        spec = _spec(providers=["sandbox"], tool_sets=["coding"])
        resolved = _resolve_tools(spec, ToolRegistry(), sets, providers)
        assert [t.__name__ for t in resolved] == ["sandbox_exec_python"]

    def test_load_agent_builds_graph(self) -> None:
        providers = {"sandbox": SandboxToolProvider(_FakeSandboxClient())}
        spec = _spec(providers=["sandbox"], tools=["sandbox_exec_shell"])
        loaded = load_agent_from_spec(
            spec, tool_registry=ToolRegistry(), llm_registry=_llm_registry(),
            tool_providers=providers,
        )
        assert loaded.graph is not None
