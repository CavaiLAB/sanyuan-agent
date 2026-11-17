"""Built-in tool catalog — canonical tool sets + a one-call provider assembler.

Two standardization gaps this closes:

1. **No canonical tool sets.** :data:`DEFAULT_TOOL_SETS` names the families a
   deployment usually enables (``web`` / ``sandbox`` / ``multimodal`` / …) so an
   ``AgentSpec`` can say ``tool_sets: ["web", "sandbox"]`` instead of spelling
   every tool name, and so the names are defined in exactly one place. Pass it as
   ``load_agent_from_spec(tool_sets=DEFAULT_TOOL_SETS)``.

2. **No single registration entry.** :func:`builtin_providers` assembles the
   built-in :class:`~neuroagent.tools.provider.ToolProvider` map from whatever
   clients a deployment injects, *skipping* any provider whose dependency is
   absent — a deployment with no device farm simply omits ``mobile_client`` and
   gets no mobile provider rather than a broken tool. Pass the result as
   ``load_agent_from_spec(tool_providers=...)``.

The catalog only assembles providers; it enforces nothing. ``ToolPolicy`` /
``RiskGate`` remain the runtime authority, and the harness reads no real secret
here (clients arrive pre-constructed from the caller).
"""

from __future__ import annotations

from typing import Any

from neuroagent.tools.artifact_write import ArtifactWriteToolProvider
from neuroagent.tools.bash import BashToolProvider
from neuroagent.tools.browser import BrowserToolProvider
from neuroagent.tools.computer_use import ComputerUseToolProvider
from neuroagent.tools.execute_skills import ExecuteSkillsToolProvider
from neuroagent.tools.file_ops import FileOpsToolProvider
from neuroagent.tools.knowledge import KnowledgeBaseToolProvider
from neuroagent.tools.link_reader import LinkReaderToolProvider
from neuroagent.tools.memory import MemoryToolProvider
from neuroagent.tools.mobile_use import MobileUseToolProvider
from neuroagent.tools.multimodal import MultimodalToolProvider
from neuroagent.tools.provider import ToolProvider
from neuroagent.tools.sandbox_tool import SandboxToolProvider
from neuroagent.tools.web_provider import WebToolProvider

# Canonical named tool sets. Each value lists the *tool names* (as registered),
# not provider names. Members must be enabled by an in-range provider/registry at
# load time; the loader raises on a dangling name, so a typo fails loudly.
DEFAULT_TOOL_SETS: dict[str, tuple[str, ...]] = {
    "web": ("web_search", "web_fetch", "link_reader"),
    "filesystem": ("read_file", "write_file", "list_dir", "upload_file"),
    "sandbox": ("bash", "sandbox_exec_shell", "sandbox_exec_python"),
    "multimodal": ("image_generate", "image_edit", "video_generate", "text_to_speech"),
    "automation": ("mobile_use", "computer_use"),
    "browser": (
        "browser_navigate",
        "browser_click",
        "browser_type",
        "browser_screenshot",
        "browser_read_dom",
        "browser_accessibility",
        "browser_download",
        "browser_downloads",
    ),
    "knowledge": ("knowledge_search", "remember_this", "recall_memories"),
    "artifact": ("artifact_write",),
    "skills": ("execute_skills",),
    # Shell + file ops bound to ONE isolated workspace-sandbox session (the
    # ISandboxProvider path: BashToolProvider + FileOpsToolProvider). Excludes
    # upload_file deliberately — it reads from the gateway host's local disk,
    # which a sandbox-scoped worker must never do.
    "sandbox_workspace": ("bash", "read_file", "write_file", "list_dir"),
    # Research workers get only the two server-routed read tools. LinkReader is
    # omitted because web_fetch already shares ReaderRouter and its evidence
    # ledger; browser interaction remains an internal Reader fallback.
    "web_research": ("web_search", "web_fetch"),
    # A pragmatic everyday default: web + light filesystem + sandbox shell +
    # artifacts + skills. Omits external generators and device control, which a
    # deployment opts into deliberately.
    "standard": (
        "web_search",
        "web_fetch",
        "link_reader",
        "bash",
        "read_file",
        "write_file",
        "list_dir",
        "artifact_write",
        "execute_skills",
    ),
}


def builtin_providers(
    *,
    # Web (Router optional; None keeps the direct legacy path).
    include_web: bool = True,
    include_link_reader: bool = True,
    web_search_router: Any | None = None,
    web_reader_router: Any | None = None,
    # Sandbox file/shell via ISandboxProvider + a bound session.
    sandbox_provider: Any | None = None,
    sandbox_session: Any | None = None,
    # Code-exec sandbox via a SandboxClientProto.
    sandbox_client: Any | None = None,
    # Artifact write via a StorageArtifactClient.
    artifact_client: Any | None = None,
    artifact_tenant_id: str = "",
    # Skills.
    skill_registry: Any | None = None,
    # Capability clients (each optional).
    multimodal_client: Any | None = None,
    mobile_client: Any | None = None,
    computer_client: Any | None = None,
    # Knowledge base retrieval.
    knowledge_base: Any | None = None,
    embedder: Any | None = None,
    knowledge_collection_id: str = "",
    # Long-term memory.
    memory_store: Any | None = None,
    memory_tenant_id: str = "",
    memory_user_id: str = "",
    # Browser.
    browser_session_factory: Any | None = None,
) -> dict[str, ToolProvider]:
    """Assemble the built-in provider map from whatever clients are injected.

    Each provider is included only when its dependency is present, so the result
    contains exactly the capabilities this deployment can actually serve. Keys
    are provider names suitable for ``AgentSpec.providers``: ``web`` /
    ``link_reader`` / ``bash`` / ``filesystem`` / ``sandbox_exec`` / ``artifact``
    / ``skills`` / ``multimodal`` / ``mobile`` / ``computer`` / ``knowledge`` /
    ``memory`` / ``browser``.
    """
    providers: dict[str, ToolProvider] = {}

    if include_web:
        providers["web"] = WebToolProvider(web_search_router, web_reader_router)
    if include_link_reader:
        providers["link_reader"] = LinkReaderToolProvider(reader_router=web_reader_router)

    if sandbox_provider is not None and sandbox_session is not None:
        providers["bash"] = BashToolProvider(sandbox_provider, sandbox_session)
        providers["filesystem"] = FileOpsToolProvider(sandbox_provider, sandbox_session)
    if sandbox_client is not None:
        providers["sandbox_exec"] = SandboxToolProvider(sandbox_client)

    if artifact_client is not None:
        providers["artifact"] = ArtifactWriteToolProvider(
            artifact_client, tenant_id=artifact_tenant_id
        )
    if skill_registry is not None:
        providers["skills"] = ExecuteSkillsToolProvider(skill_registry)

    if multimodal_client is not None:
        providers["multimodal"] = MultimodalToolProvider(multimodal_client)
    if mobile_client is not None:
        providers["mobile"] = MobileUseToolProvider(mobile_client)
    if computer_client is not None:
        providers["computer"] = ComputerUseToolProvider(computer_client)

    if knowledge_base is not None and embedder is not None and knowledge_collection_id:
        providers["knowledge"] = KnowledgeBaseToolProvider(
            knowledge_base, embedder, collection_id=knowledge_collection_id
        )
    if memory_store is not None:
        providers["memory"] = MemoryToolProvider(
            memory_store, tenant_id=memory_tenant_id, user_id=memory_user_id
        )

    if browser_session_factory is not None:
        providers["browser"] = BrowserToolProvider(browser_session_factory)

    return providers
