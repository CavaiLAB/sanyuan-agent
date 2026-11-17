# Harness 开源核心

Apache 2.0 Agent 内核；入口见 [`README.md`](README.md) 和 `pyproject.toml`，源码在 `sanyuan_agent/`。

- 不得依赖 `neuroagent_app` 或任何宿主侧包；保留独立安装、CLI 基本运行和公开 API 的兼容性。
- 新依赖需符合本项目 Apache 2.0 许可约束；MIT / BSD / Apache 2.0 / ISC 可按现有政策核验，禁止引入 GPL 类不兼容依赖。
- 重度借鉴 DeerFlow 的文件保留来源及许可 header，规则见 [`deerflow-reference.md`](docs/deerflow-reference.md)；提交和开源文件不泄露内部秘密。
- 修改行为运行相关 `tests/`；边界变化同时运行 workspace 的 `../../tests/test_harness_boundary.py`，按需检查独立安装与入口。
- 修改 public API 时同步 README 或长期服务文档的示例；工具执行、审计和租户边界沿用上级规则。
