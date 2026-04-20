<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-20 | Updated: 2026-04-20 -->

# harness

## Purpose

Agent **开源核心**（Apache 2.0）。可独立发布到 PyPI、企业可裸用。严格不依赖 `app/` 或 `sandbox_supervisor/` 的任何符号。

## Status

**Phase 1 Week 1-2 空骨架**。后续将新增 `neuroagent/` 包（按 SAD §8）：

```
neuroagent/
├── cli.py             # 独立运行入口，CI 必跑
├── agents/            # LangGraph 状态机
├── tools/             # Tool 基类 + builtins
├── sandbox/           # SandboxClient + Provider 接口（不是实现）
├── skills/            # SKILL.md 解析 + Ed25519 签名
├── memory/            # profile/working/semantic 三层
├── models/            # LiteLLM 包装 + Provider Registry + Cost Tracker
├── recovery/          # Tool-call recovery（借鉴 DeerFlow middleware）
└── config/
```

## Key Files

| File | Description |
|------|-------------|
| `pyproject.toml` | Harness 独立包配置 |
| `LICENSE` | Apache 2.0 全文 |
| `README.md` | Harness 独立说明 |

## For AI Agents

### Working In This Directory

- **License 合规**：所有新加依赖必须 Apache 2.0 兼容（MIT / BSD / Apache 2.0 / ISC）；GPL 类传染性 License 不行
- **开源友好**：commit 信息假定会被外部阅读，不要带公司内部细节
- **Derived from**：重度借鉴 DeerFlow 的文件要加 `# Derived from bytedance/deer-flow (MIT): <path>` header（见 `../../docs/deerflow-reference.md` §9.3）

### Testing Requirements

- 自己的单元测试在 `../../tests/unit/harness/` 下
- CI 跑 `test_harness_boundary.py` + `test_harness_standalone.py`

## Dependencies

### Internal

- **不能** 依赖 `app/` 或 `sandbox_supervisor/`（CI 硬检查）

### External

- LangGraph / LiteLLM / Playwright / Pydantic v2 / structlog

<!-- MANUAL: -->
