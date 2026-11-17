# AGENTS.md：给编码代理的入口说明

这个仓库是 **SanYuanAgent** 的家：匠心智能体平台的智能体引擎（Python）。引擎的核心在 `packages/core`，2026-11-17 带着历史从 iClaw 仓库的 Python Agent 引擎迁入，迁入时改了名。服务层与媒体适配层还在原仓库，随后再迁。

## 要改代码

- 核心在 `packages/core/`：包目录 `sanyuan_agent/`、核心自带的测试 `tests/`、核心的说明 [`packages/core/README.md`](packages/core/README.md) 与 [`packages/core/AGENTS.md`](packages/core/AGENTS.md)。根目录的 `tests/` 是工作区级的测试，也只测核心。
- 名字：发行名 `sanyuan-agent`，导入名 `sanyuan_agent`，命令行 `sanyuan-agent`，环境变量前缀 `SANYUAN_AGENT_`。仓库里剩下的旧名字都登记在 `tools/migration/rename/exceptions.tsv`，每处写了为什么不改；新加没登记的旧名字，残留检查会失败。
- 核心不得引用服务层与媒体适配层的包。原来的名字 `neuroagent_app`、`neuroagent_media`，以及改名后的对应名 `sanyuan_agent_app`、`sanyuan_agent_media`（暂定），都在 CI 的静态扫描和边界测试里查。核心要能单独安装、命令行能起来。
- 核心的新依赖要和 Apache-2.0 兼容：MIT、BSD、Apache-2.0、ISC 可以，GPL 类的不行。
- 借鉴开源项目的地方保留来源与许可声明，规则见 [`packages/core/docs/deerflow-reference.md`](packages/core/docs/deerflow-reference.md)；版权与上游许可在 [`packages/core/NOTICE`](packages/core/NOTICE)。

装环境、跑检查与测试（在仓库根目录，要先装 [uv](https://docs.astral.sh/uv/)）：

```bash
uv sync --all-packages
uvx ruff check packages/
uv run pytest packages/core/tests
uv run pytest tests
python3 tools/migration/rename/check_residue.py
```

两组测试里有同名的测试文件，要分两次跑。迁移之前就不过的几个用例，CI 用显式清单排除并写了原因，见 [`.github/workflows/ci.yml`](.github/workflows/ci.yml)。mypy 迁移之前就不过，不是门禁。

## 过渡期

- 已定：原仓库里那一份核心这一步不动，照旧在用。它上面的新改动同步到这里，做法见 [`tools/migration/README.md`](tools/migration/README.md)。同步过来的提交要用 merge commit 合并，squash 或 rebase 会改写导入的历史。
- 待定：过渡期能不能先在本仓库对核心做行为上的改动。

## 要改文档

1. 先读 [README.md](README.md) 与平台总览仓库的[名称对照](https://github.com/CavaiLAB/jiangxin/blob/main/docs/glossary.md)。
2. 文档用中文，不写参考平台和对标项目的名字。例外：许可与署名里对所借鉴开源项目的指名（`packages/core/NOTICE`、借鉴说明、代码里的来源与许可注释）照常保留，不要当成违规清掉。
3. 分清「已定」「提议」「待定」。发行名、核心的许可（Apache-2.0）、仓库公开已定；服务层与媒体适配层的名字和许可仍待定。
4. 说现状的时候写清是「已有」还是「规划」。核心的代码就在本仓库，可以对着代码写。原仓库是私有的：它的其余部分只写框架层面的内容，不写内部实现细节、文件路径和安全相关的细节；不放产品清单与建设状态，对它只写可以承接的能力类别，不写功能盘点与差距。安全组件只写它是什么，不写「一定会拦住」这类行为保证，除非对着代码核对过它在那条运行路径上确实启用。
5. 平台级的口径（名称、三个框架的分工、契约、Agent 协同、三种形态）在 [jiangxin](https://github.com/CavaiLAB/jiangxin) 里改，这里只引用。
6. 在 `docs/` 下新增文档，要在 `docs/_meta.json` 里加上它的文件名（不带扩展名）。这个文件定这些文档在平台文档站侧栏里的顺序；漏列的话，文档站升级到这个版本时构建会报错。核心自己的长文档放在 `packages/core/` 下，不放进根 `docs/`。
7. 改动走分支和 PR，不直接推 `main`。建仓期间的初始文档除外，见平台总览的[协作约定](https://github.com/CavaiLAB/jiangxin/blob/main/docs/07-governance.md)第 2 节。
