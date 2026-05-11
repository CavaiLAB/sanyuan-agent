<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-20 | Updated: 2026-05-12 -->

# harness

## Purpose

Agent **开源核心**（Apache 2.0）。可独立发布到 PyPI、企业可裸用，并严格不依赖 `app/` 或 `sandbox_supervisor/` 的任何符号。

## Status

**Phase 1 已进入实现态**。当前包已包含 CLI、agents、tools、sandbox、skills、memory、models、costs、artifacts、security 等模块以及包内测试；后续扩展必须优先保持 harness 的独立安装与独立运行能力。

## Key Files

| File | Description |
|---|---|
| `pyproject.toml` | Harness 独立包配置 |
| `LICENSE` | Apache 2.0 全文 |
| `README.md` | Harness 独立说明 |
| `neuroagent/` | Harness 源码 |
| `tests/` | Harness 包内测试 |

## For AI Agents

### Working In This Directory

- **License 合规**：所有新加依赖必须 Apache 2.0 兼容（MIT / BSD / Apache 2.0 / ISC）；GPL 类传染性 License 不行。
- **开源友好**：commit 信息假定会被外部阅读，不要带公司内部细节。
- **Derived from**：重度借鉴 DeerFlow 的文件要加 `# Derived from bytedance/deer-flow (MIT): <path>` header（见 `../../../../docs/services/python/deerflow-reference.md` §9.3）。
- **边界铁律**：不得 import `neuroagent_app` 或 `neuroagent_sandbox_supervisor`。

### Testing Requirements

- 包内测试在 `tests/` 下。
- Workspace 级边界与回归测试在 `../../tests/` 下。
- 修改 public API 后同时更新 `README.md` 或服务侧 docs 中的示例。

## Dependencies

### Internal

- **不能** 依赖 `app/` 或 `sandbox_supervisor/`（CI / 测试硬检查）。

### External

- LangGraph / LiteLLM / Playwright / Pydantic v2 / structlog。

<!-- MANUAL: -->
