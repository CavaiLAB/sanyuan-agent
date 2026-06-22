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

## 开发

本包作为 `neuroagent-python` 仓库的 uv workspace 成员存在。开发细节见父仓 README。

## 设计致谢

Sandbox 抽象、Skills 格式、Tool-Call Recovery、Provider Registry 等设计借鉴自 [DeerFlow 2.0](https://github.com/bytedance/deer-flow)（MIT License）。
