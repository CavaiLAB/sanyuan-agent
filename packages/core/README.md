# neuroagent (Harness)

NeuroAgent Agent 引擎核心，开源部分。

**Apache License 2.0**

## 能做什么

- 在任何 Python 3.12+ 项目里，用标准 LangGraph + Pydantic 构建可运行、可审计、可插拔的 Agent
- 对接任意 LLM Provider（LiteLLM 兼容的都行）
- 在本地、Docker、K8s 里跑 Sandbox（Agent 有自己的文件系统 + 终端 + 浏览器）
- 加载 SKILL.md 格式的能力模块（agentskills.io 开放标准）
- 四层记忆（Profile / Working / Semantic / Trace）

## 独立可运行性

本包**不依赖任何 NeuroAgent 商业版组件**。可以独立：

```python
from neuroagent import AgentBuilder, load_config

config = load_config("my-config.yaml")
agent = AgentBuilder(config).build()

result = await agent.run("帮我调研 2026 年的 AI Coding IDE")
print(result.final_response)
```

或 CLI：

```bash
uv add neuroagent
neuroagent --config my-config.yaml --task "..."
```

## 工具注册：Provider path 与旧 builder path

工具注册有两条**等价且兼容**的路径。旧的 `build_*_tool` builder 不变，照常工作；
新增的 Provider path（`neuroagent.tools.provider`）在 `ToolRegistry` 之上统一了
声明形态——把 name / callable / schema 收敛成一个 `ToolSpec`，再附带不参与执行的
元数据（category / capabilities / side_effect / required_secrets / artifact_output /
policy_hint）。元数据仅供后续构建 `ToolPolicy`、前端展示、审计标签使用，**不**绕过
executor / `ToolPolicy` / `RiskGate`。

```python
# 旧路径（仍然可用）：builder 直接注册进 registry
from neuroagent.tools import ToolRegistry, build_web_search_tool

registry = ToolRegistry()
build_web_search_tool(registry)               # 注册 "web_search"

# 新路径：把任意现有 callable 适配成 spec，再批量注册（schema 与旧路径一致）
from neuroagent.tools import (
    ToolRegistry,
    build_web_search_tool,
    register_tool_specs,
    spec_from_callable,
)

registry = ToolRegistry()
web_search = build_web_search_tool(ToolRegistry())   # 拿到 callable
register_tool_specs(
    registry,
    [spec_from_callable(web_search, category="web", capabilities=("search",))],
)
# registry.get_schema("web_search") / registry.tool_defs() 与旧路径完全相同
```

`spec_from_callable` 的 schema 解析优先级与 agent builder 的 `_to_tool_def` 对齐：
显式 `schema` → callable 的 `__tool_schema__`（MCP proxy 携带）→ 签名反射，因此
provider path 与 `build_simple_agent(tools=[callable])` 旧路径结果等价。一个 provider
只要实现 `provide() -> list[ToolSpec]`，即可用 `register_provider(registry, provider)`
批量注册。

### AgentSpec 的 `tool_sets` / `providers`（声明式选择）

`AgentSpec` 新增两个可选字段，配合 `load_agent_from_spec(..., tool_sets=..., tool_providers=...)`
做声明式工具选择：

- **`providers`**：引用的 `ToolProvider` 只把其 `provide()` 的 `ToolSpec` 加入**候选空间**，
  *不*自动启用任何工具。
- **最终启用工具** = `tool_sets` 按声明顺序展开 + `tools`，**稳定去重**（首次出现胜，重复不报错）。
- 解析时 **base `ToolRegistry` 优先**于 provider 候选；若 provider 候选名与 base registry
  或另一个 provider 候选**同名冲突**，抛 `ValueError`（避免 MCP/builtin/provider 静默覆盖）。
- 未知 provider / 未知 tool set / 展开后未知工具名，均抛 `ValueError`。
- **不 mutate** 传入的 `ToolRegistry`——候选只在加载期解析，绝不注册进传入的 registry。

```python
from neuroagent.config import AgentSpec, load_agent_from_spec

spec = AgentSpec(
    name="researcher",
    system_prompt="...",
    llm_provider="higress",
    providers=["web"],            # 仅贡献候选
    tool_sets=["search_bundle"],  # 展开后并入启用集
    tools=["web_search"],         # 与 set 重复的名字会被稳定去重
)
loaded = load_agent_from_spec(
    spec,
    tool_registry=base_registry,
    llm_registry=llm_registry,
    tool_sets={"search_bundle": ["web_search", "web_fetch"]},
    tool_providers={"web": my_web_provider},
)
```

### 多模态工具 provider

`MultimodalToolProvider(client)` 产出 4 个 VeADK 风格工具：`image_generate` /
`image_edit` / `video_generate` / `text_to_speech`。两条边界：

- **不接真实外部 API**：provider 只调用注入的 `MultimodalClient`（Protocol）；商业侧
  后续注入真实 client（vendor + 凭证），测试注入 fake。harness 不 import
  `neuroagent_app` / `sandbox_supervisor`、不读 env。
- **artifact/workspace 返回，绝不裸路径**：每个工具返回 `MultimodalArtifact` 形态
  （`artifact_id` / `filename` / `content_type` / `download_url` / `sha256` / 可选
  `workspace_ref`），生成物是一等 artifact。

工具携带 provider metadata：`category="multimodal"`、`side_effect="external"`、
`artifact_output=True`、`policy_hint="confirm"`，per-tool `capabilities` 与
`required_secrets`（如 `MODEL_IMAGE_NAME`）。这些 metadata 与 PR1 一致：仅供构建
`ToolPolicy` / 前端 / 审计**描述用**，不参与执行；`required_secrets` 只**声明**
env-var 名，本模块**不读取**。

```python
from neuroagent.tools import MultimodalToolProvider, ToolRegistry, register_provider

registry = ToolRegistry()
register_provider(registry, MultimodalToolProvider(my_multimodal_client))
# registry 含 image_generate / image_edit / video_generate / text_to_speech
```

#### app 注入 seam（PR4，默认关闭）

harness 这一侧只提供**协议 + 工具壳**（`MultimodalToolProvider` + `MultimodalClient`
Protocol），从不接真实供应商、从不读 secret。商业侧（`neuroagent_app`）的 **PR4** 增加
注入缝把它接进网关：

- `neuroagent_app.config.Settings` 增加多模态**开关 + 模型名**字段
  （`multimodal_tools_enabled` / `multimodal_fake_client` / `multimodal_*_model`），
  **默认全关**；字段只声明启用开关与模型**名**，**绝不**存放或日志 secret VALUE。
- `neuroagent_app.multimodal.build_multimodal_provider(settings)` 是决策点：只有在
  **显式启用且选中 fake client** 时才返回 provider（绑定 `FakeMultimodalClient`——
  确定性 stub artifact、无外网、无 secret 读取）；启用但无 fake → 返回 `None`。
- `assemble_agent_tools(..., multimodal_provider=...)` 接收候选 provider，把其 4 个工具
  callable 展开进工具列表；**不传则默认工具列表不变**，生产路径默认不暴露外部生成工具。

**真实供应商 client 与 secret resolver 是 PR5+**。PR4 只把缝接好（Settings → factory →
fake client → `assemble_agent_tools` → agent → SSE artifact 抽取），不引入任何付费 API
或外网调用。

## 开发

本包作为 `neuroagent-python` 仓库的 uv workspace 成员存在。开发细节见父仓 README。

## 设计致谢

Sandbox 抽象、Skills 格式、Tool-Call Recovery、Provider Registry 等设计借鉴自 [DeerFlow 2.0](https://github.com/bytedance/deer-flow)（MIT License）。
