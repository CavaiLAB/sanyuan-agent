"""Built-in catalog: canonical tool sets + the one-call provider assembler."""

from __future__ import annotations

from neuroagent.tools.catalog import DEFAULT_TOOL_SETS, builtin_providers
from neuroagent.tools.computer_use import FakeComputerUseClient


class TestDefaultToolSets:
    def test_standard_set_is_a_subset_of_named_families(self) -> None:
        families = set().union(*(set(v) for k, v in DEFAULT_TOOL_SETS.items() if k != "standard"))
        assert set(DEFAULT_TOOL_SETS["standard"]) <= families

    def test_sets_are_tuples_of_names(self) -> None:
        for members in DEFAULT_TOOL_SETS.values():
            assert isinstance(members, tuple)
            assert all(isinstance(n, str) for n in members)


class TestBuiltinProviders:
    def test_web_included_by_default_others_skipped_without_clients(self) -> None:
        providers = builtin_providers()
        assert set(providers) == {"web", "link_reader"}

    def test_client_presence_drives_inclusion(self) -> None:
        providers = builtin_providers(computer_client=FakeComputerUseClient())
        assert "computer" in providers
        assert "mobile" not in providers  # no mobile_client supplied

    def test_knowledge_requires_kb_embedder_and_collection(self) -> None:
        # Missing collection id → no knowledge provider, no crash.
        providers = builtin_providers(knowledge_base=object(), embedder=object())
        assert "knowledge" not in providers

    def test_assembled_provider_yields_set_member_tool(self) -> None:
        providers = builtin_providers(computer_client=FakeComputerUseClient())
        (spec,) = providers["computer"].provide()
        assert spec.name == "computer_use"
        # The provided tool is a member of its canonical set, so an AgentSpec can
        # select it via tool_sets=["automation"] against DEFAULT_TOOL_SETS.
        assert spec.name in DEFAULT_TOOL_SETS["automation"]


class TestSandboxWorkspaceSet:
    def test_members_are_session_bound_and_exclude_host_upload(self) -> None:
        # The workflow-facing controlled set: shell + file ops served by
        # BashToolProvider/FileOpsToolProvider against ONE sandbox session.
        assert DEFAULT_TOOL_SETS["sandbox_workspace"] == (
            "bash",
            "read_file",
            "write_file",
            "list_dir",
        )
        # upload_file reads from the gateway host's local disk — it must never
        # be part of a sandbox-scoped worker set.
        assert "upload_file" not in DEFAULT_TOOL_SETS["sandbox_workspace"]
