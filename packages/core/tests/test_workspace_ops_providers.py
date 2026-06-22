"""Workspace-ops provider wrappers (PR10): artifact_write / file_ops / bash.

Each is a thin ToolProvider over an existing builder. For every provider these
tests pin: runtime-checkable Protocol, descriptive metadata, byte-identical
legacy equivalence (tool_defs match the builder-registered path), AgentSpec
provider selection, and behavior over a fake client / sandbox provider.

No new execution behavior, secrets, app imports or real IO - fakes only.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from neuroagent.artifacts.client import ArtifactRef
from neuroagent.config import AgentSpec, load_agent_from_spec
from neuroagent.config.loader import _resolve_tools
from neuroagent.models import LLMProvider, LLMProviderRegistry
from neuroagent.tools import (
    ArtifactWriteToolProvider,
    BashToolProvider,
    FileOpsToolProvider,
    build_artifact_write_tool,
    build_bash_tool,
    build_file_ops_tools,
)
from neuroagent.tools.provider import ToolProvider, register_provider
from neuroagent.tools.registry import ToolRegistry

_SESSION = SimpleNamespace(id="sess-1")


# ---------------------------------------------------------------------------
# Fakes (no real IO)
# ---------------------------------------------------------------------------


class _FakeArtifactClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create_artifact(
        self, *, tenant_id: str, content_bytes: bytes, content_type: str, filename: str
    ) -> ArtifactRef:
        self.calls.append(
            {
                "tenant_id": tenant_id,
                "content_bytes": content_bytes,
                "content_type": content_type,
                "filename": filename,
            }
        )
        return ArtifactRef(
            id="a1",
            url="/api/v1/artifacts/a1",
            sha256="sha-1",
            filename=filename,
            content_type=content_type,
        )


class _FakeSandbox:
    """Duck-typed ISandboxProvider stand-in; records calls, no real IO."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    async def read_file(self, session_id: str, path: str) -> str:
        self.calls.append(("read_file", session_id, path))
        return f"content:{path}"

    async def write_file(self, session_id: str, path: str, content: str) -> None:
        self.calls.append(("write_file", session_id, path, content))

    async def list_dir(self, session_id: str, path: str) -> list[str]:
        self.calls.append(("list_dir", session_id, path))
        return ["a.txt", "b.txt"]

    async def upload_file(
        self, session_id: str, local_path: str, remote_path: str
    ) -> None:
        self.calls.append(("upload_file", session_id, local_path, remote_path))

    async def run_shell(self, session_id: str, cmd: str) -> Any:
        self.calls.append(("run_shell", session_id, cmd))
        return SimpleNamespace(stdout=f"ran {cmd}", stderr="", exit_code=0)


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
    base = {"name": "ops", "system_prompt": "You operate files.", "llm_provider": "higress"}
    base.update(over)
    return AgentSpec(**base)


# ---------------------------------------------------------------------------
# ArtifactWriteToolProvider
# ---------------------------------------------------------------------------


class TestArtifactWriteProvider:
    def _provider(self) -> ArtifactWriteToolProvider:
        return ArtifactWriteToolProvider(_FakeArtifactClient(), tenant_id="t1")

    def test_runtime_checkable(self) -> None:
        assert isinstance(self._provider(), ToolProvider)

    def test_metadata(self) -> None:
        (spec,) = self._provider().provide()
        assert spec.name == "artifact_write"
        assert spec.category == "artifact"
        assert spec.side_effect == "write"
        assert spec.artifact_output is True
        assert spec.policy_hint == "confirm"
        assert spec.capabilities == ("artifact", "write")
        assert spec.required_secrets == ()
        props = spec.schema["parameters"]["properties"]
        assert set(props) == {"filename", "content", "content_type"}
        assert spec.schema["parameters"]["required"] == ["filename", "content"]

    def test_legacy_equivalence(self) -> None:
        client = _FakeArtifactClient()
        legacy = ToolRegistry()
        build_artifact_write_tool(legacy, client, tenant_id="t1")

        provider_reg = ToolRegistry()
        names = register_provider(
            provider_reg, ArtifactWriteToolProvider(client, tenant_id="t1")
        )
        assert names == ["artifact_write"]
        assert provider_reg.get_schema("artifact_write") == legacy.get_schema("artifact_write")
        assert provider_reg.tool_defs() == legacy.tool_defs()

    async def test_behavior_over_fake_client(self) -> None:
        client = _FakeArtifactClient()
        registry = ToolRegistry()
        register_provider(registry, ArtifactWriteToolProvider(client, tenant_id="t1"))

        result = await registry.execute(
            "artifact_write", '{"filename": "out.txt", "content": "hi"}'
        )
        assert result["artifact_id"] == "a1"
        assert result["download_url"] == "/api/v1/artifacts/a1"
        assert client.calls == [
            {
                "tenant_id": "t1",
                "content_bytes": b"hi",
                "content_type": "text/plain",
                "filename": "out.txt",
            }
        ]

    def test_loader_selection(self) -> None:
        providers = {"artifact": self._provider()}
        assert _resolve_tools(_spec(providers=["artifact"]), ToolRegistry(), None, providers) == []
        spec = _spec(providers=["artifact"], tools=["artifact_write"])
        resolved = _resolve_tools(spec, ToolRegistry(), None, providers)
        assert [t.__name__ for t in resolved] == ["artifact_write"]
        loaded = load_agent_from_spec(
            spec, tool_registry=ToolRegistry(), llm_registry=_llm_registry(),
            tool_providers=providers,
        )
        assert loaded.graph is not None


# ---------------------------------------------------------------------------
# FileOpsToolProvider
# ---------------------------------------------------------------------------


class TestFileOpsProvider:
    def _provider(self) -> FileOpsToolProvider:
        return FileOpsToolProvider(_FakeSandbox(), _SESSION)

    def test_runtime_checkable(self) -> None:
        assert isinstance(self._provider(), ToolProvider)

    def test_metadata(self) -> None:
        specs = {s.name: s for s in self._provider().provide()}
        assert set(specs) == {"read_file", "write_file", "list_dir", "upload_file"}
        for s in specs.values():
            assert s.category == "filesystem"
            assert s.required_secrets == ()
        # reads stay read with no hint
        for name in ("read_file", "list_dir"):
            assert specs[name].side_effect == "read"
            assert specs[name].policy_hint is None
        # mutations are write + confirm
        for name in ("write_file", "upload_file"):
            assert specs[name].side_effect == "write"
            assert specs[name].policy_hint == "confirm"

    def test_legacy_equivalence(self) -> None:
        sandbox = _FakeSandbox()
        legacy = ToolRegistry()
        build_file_ops_tools(sandbox, _SESSION, legacy)

        provider_reg = ToolRegistry()
        register_provider(provider_reg, FileOpsToolProvider(sandbox, _SESSION))

        assert provider_reg.list_tools() == legacy.list_tools()
        for name in legacy.list_tools():
            assert provider_reg.get_schema(name) == legacy.get_schema(name)
        assert provider_reg.tool_defs() == legacy.tool_defs()

    async def test_behavior_over_fake_sandbox(self) -> None:
        sandbox = _FakeSandbox()
        registry = ToolRegistry()
        register_provider(registry, FileOpsToolProvider(sandbox, _SESSION))

        read = await registry.execute("read_file", '{"path": "/p.txt"}')
        assert read.content == "content:/p.txt"
        await registry.execute("write_file", '{"path": "/p.txt", "content": "x"}')
        assert ("read_file", "sess-1", "/p.txt") in sandbox.calls
        assert ("write_file", "sess-1", "/p.txt", "x") in sandbox.calls

    def test_loader_selection_subset(self) -> None:
        providers = {"files": self._provider()}
        assert _resolve_tools(_spec(providers=["files"]), ToolRegistry(), None, providers) == []
        spec = _spec(providers=["files"], tools=["read_file", "write_file"])
        resolved = _resolve_tools(spec, ToolRegistry(), None, providers)
        assert [t.__name__ for t in resolved] == ["read_file", "write_file"]
        loaded = load_agent_from_spec(
            spec, tool_registry=ToolRegistry(), llm_registry=_llm_registry(),
            tool_providers=providers,
        )
        assert loaded.graph is not None


# ---------------------------------------------------------------------------
# BashToolProvider
# ---------------------------------------------------------------------------


class TestBashProvider:
    def _provider(self) -> BashToolProvider:
        return BashToolProvider(_FakeSandbox(), _SESSION)

    def test_runtime_checkable(self) -> None:
        assert isinstance(self._provider(), ToolProvider)

    def test_metadata(self) -> None:
        (spec,) = self._provider().provide()
        assert spec.name == "bash"
        assert spec.category == "sandbox"
        assert spec.capabilities == ("sandbox", "execute")
        assert spec.side_effect == "external"
        assert spec.policy_hint == "confirm"
        assert spec.required_secrets == ()
        # legacy quirk preserved: registry name "bash", schema function name "tool_bash"
        assert spec.schema["name"] == "tool_bash"
        props = spec.schema["parameters"]["properties"]
        assert set(props) == {"cmd", "timeout"}
        assert spec.schema["parameters"]["required"] == ["cmd"]

    def test_legacy_equivalence(self) -> None:
        sandbox = _FakeSandbox()
        legacy = ToolRegistry()
        build_bash_tool(sandbox, _SESSION, legacy)

        provider_reg = ToolRegistry()
        names = register_provider(provider_reg, BashToolProvider(sandbox, _SESSION))
        assert names == ["bash"]
        assert provider_reg.get_schema("bash") == legacy.get_schema("bash")
        assert provider_reg.tool_defs() == legacy.tool_defs()

    async def test_behavior_over_fake_sandbox(self) -> None:
        sandbox = _FakeSandbox()
        registry = ToolRegistry()
        register_provider(registry, BashToolProvider(sandbox, _SESSION))

        result = await registry.execute("bash", '{"cmd": "ls"}')
        assert result == {"stdout": "ran ls", "stderr": "", "exit_code": 0}
        assert ("run_shell", "sess-1", "ls") in sandbox.calls

    def test_loader_selection(self) -> None:
        providers = {"sh": self._provider()}
        assert _resolve_tools(_spec(providers=["sh"]), ToolRegistry(), None, providers) == []
        spec = _spec(providers=["sh"], tools=["bash"])
        resolved = _resolve_tools(spec, ToolRegistry(), None, providers)
        assert [t.__name__ for t in resolved] == ["tool_bash"]  # callable __name__
        loaded = load_agent_from_spec(
            spec, tool_registry=ToolRegistry(), llm_registry=_llm_registry(),
            tool_providers=providers,
        )
        assert loaded.graph is not None
