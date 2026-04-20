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

## 开发

本包作为 `neuroagent-python` 仓库的 uv workspace 成员存在。开发细节见父仓 README。

## 设计致谢

Sandbox 抽象、Skills 格式、Tool-Call Recovery、Provider Registry 等设计借鉴自 [DeerFlow 2.0](https://github.com/bytedance/deer-flow)（MIT License）。
