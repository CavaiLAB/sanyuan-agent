# 迁移工具

核心带着历史从 iClaw 仓库迁入本仓库时用的脚本，以及过渡期从原仓库同步新改动的做法。为什么迁、迁了什么、进度如何，见 [docs/migration.md](../../docs/migration.md)。

原仓库是私有的，重跑导入要有它的读权限。

## 目录

| 文件 | 做什么 |
|---|---|
| `import/import.sh` | 从来源的一个提交过滤出核心的历史，放进一个新建的裸仓库，结果在其中的 `imported/iclaw-core` 分支 |
| `import/commit_callback.py` | `git filter-repo` 对每个提交调用的回调：按路径规则过滤并改路径，处理提交说明，加 trailer |
| `import/paths-iclaw.txt`、`import/paths-predecessor.txt` | 路径规则：iClaw 布局那一段、前身仓库那一段各一份 |
| `import/message_rewrites.tsv` | 提交说明的逐行处理规则：原行内容的 SHA-256 → 替换后的行，空表示删除 |
| `import/verify_import.py` | 逐个提交对照来源核对导入结果 |
| `rename/rename.py` | 改名脚本，可以重复跑 |
| `rename/exceptions.tsv` | 改名之后仍保留旧名的地方：位置、写法、类别、原因、以后由谁定 |
| `rename/check_residue.py` | 旧名残留检查，CI 里也跑 |
| `rename/exception_list.py` | 上面两个脚本共用的、读例外清单的代码 |

## 导入是怎么做的

- 范围：原仓库里核心的目录、只引用核心的工作区级测试、借鉴说明三类路径，改到本仓库的新位置（`packages/core/`、`tests/`、`packages/core/docs/`）。工作区根的 `pyproject.toml` 与 `uv.lock` 不带历史，按内容新写。
- 两段历史：核心 2026-05-11 以前在前身仓库 `neuroagent-python` 里，之后以 git subtree 并入 iClaw。两段都带：`commit_callback.py` 按提交所在的那一支选路径规则，前身仓库的规则不会误配到 iClaw 主线上同名的路径。
- 提交说明：PR / issue 编号前加上来源仓库名（`iclaw#123`、`neuroagent-python#45`），在本仓库里不会被 GitHub 识别成链接；每个提交末尾加一条 `Imported-from: iclaw@<原提交的完整 SHA>`，放进已有的 trailer 块。个别提交说明里的内部环境信息在导入时做了处理。
- 不变的：文件内容、作者、提交者与时间。原提交的签名保不住。
- 确定性：同样的来源提交、同样版本的 git 与 git-filter-repo，得到同样的提交哈希。第一次导入（git 2.54、git-filter-repo 2.47.0）：来源 `b0b6fd5f0ffb4458bc9b7a247b6da24bfa7a9b92` → `imported/iclaw-core` 为 `757156e6fe502eb6a7e0a1c23edbfa35f9bb7c49`，共 127 个提交。
- 导入结果用一个合并提交（`--no-ff --allow-unrelated-histories`）并入主干。之后合并含有导入历史的分支都要用 merge commit，squash 或 rebase 会改写导入的历史。

## 重跑导入

需要 git（带 git-filter-repo）、Python 3，以及对原仓库的读权限。

```bash
tools/migration/import/import.sh /tmp/iclaw-core.git
git -C /tmp/iclaw-core.git rev-parse imported/iclaw-core
python3 tools/migration/import/verify_import.py <来源仓库的完整克隆> b0b6fd5f0ffb4458bc9b7a247b6da24bfa7a9b92 /tmp/iclaw-core.git
```

`import.sh` 的第二、三个参数是来源提交和来源仓库地址（缺省是第一次导入用的提交和原仓库的克隆地址，也可以写本机路径）。第一行跑完，第二行应当打印 `757156e6fe502eb6a7e0a1c23edbfa35f9bb7c49`。`verify_import.py` 逐项核对作者与时间、提交说明、每个提交的整棵树、有没有漏掉的提交、两段历史的接合处，以及头提交与来源逐文件相同，任何一项不过就以非零退出。

## 改名

```bash
python3 tools/migration/rename/rename.py
python3 tools/migration/rename/check_residue.py
```

`rename.py` 按已定的规则改名：包目录 `git mv`，导入名、发行名、命令行、环境变量前缀与品牌写法逐处替换（同一个旧名在不同语境里换成不同写法），最后重新生成 `uv.lock`、按 isort 规则重排 import。登记在 `exceptions.tsv` 里的写法原样保留。脚本可以重复跑，第二次跑不改任何东西。`--dry-run` 只列出会改的文件。

`check_residue.py` 要求仓库里剩下的每一处旧名（不分大小写，连带几种变体写法）都落在 `exceptions.tsv` 登记的写法里；登记了却一处都匹配不到的条目也算失败。

例外清单里的类别：宿主侧包名（边界检查要用）、遥测标识（指标名与 meter / tracer 的作用域名）、对外协议里报给对端的名字、外部服务上登记的资源名、原产品桌面端的名字、版权人署名、历史遗留文件、迁移工具本身、迁移说明里环境变量前缀的新旧对照。每类以后改不改，由清单里写的人定。

## 过渡期的同步

已定：过渡期原仓库那一份核心照旧在用，它上面的新改动要同步过来。下面是提议的做法，已经演练过（见本节末尾）。

1. **重跑导入。** 对原仓库更新后的提交跑 `import.sh <输出目录> <新的来源提交>`，再用 `verify_import.py` 核对。过滤是确定性的，已经导入的提交哈希不变，新提交接在后面。
2. **确认是快进。** `git merge-base --is-ancestor <本仓库的 imported/iclaw-core> <新结果的 imported/iclaw-core>` 要成立；然后把新结果推到本仓库的 `imported/iclaw-core`，只许快进。
3. **合并进来。** 从 `main` 切分支，`git merge --no-ff --no-commit imported/iclaw-core`。有冲突的文件，取导入一侧的整份内容（`git checkout --theirs -- <文件>`），交给下一步重新改名：改名脚本是幂等的，会把整份文件改回新名字。例外是改名之后做过人工改动的文件（`packages/core/README.md`、`packages/core/pyproject.toml`、`packages/core/tests/test_harness_standalone.py`、`tests/test_harness_boundary.py`，以及 `packages/core/sanyuan_agent/tools/` 下的 `computer_use.py`、`execute_skills.py`、`mobile_use.py`、`multimodal.py` 和 `packages/core/tests/test_multimodal_provider.py`），这几个要手工合。导入一侧新加在旧包目录下的文件，改名脚本会挪进新目录。
4. **改名并检查。** 跑 `rename.py`、`check_residue.py` 和测试；同步进来的新内容照公开仓库的规矩过一遍（见根目录 `AGENTS.md`「要改文档」）。作为一个合并提交提交，开 PR，用 merge commit 合并。

要注意的：

- 路径规则只能追加以后才出现的路径。要加的路径如果在已导入的历史里出现过，已导入的提交哈希会变，就不能快进了，要另行处理。`message_rewrites.tsv` 同理：只能加新的行。
- 新提交可能带进新的旧名字，或者动到例外清单里登记的写法。残留检查会报出来，逐条决定改名还是登记。

演练（2026-11-17）：把「当时是在更早的来源提交上迁移的」按真实的迁移步骤重建出来，再按上面的做法同步到 `b0b6fd5f`，做了两次。

| 起点 | 要同步的提交 | 冲突 | 结果 |
|---|---|---|---|
| 最后一个核心提交之前 | 1 个（只改核心的 `pyproject.toml`） | 无 | 与直接在 `b0b6fd5f` 上迁移得到的树相同 |
| 再往前一个核心提交之前 | 2 个（改了包里的代码与测试） | 核心的 `AGENTS.md` 和一个测试文件，按第 3 步处理 | 同上 |

两次的结果都只在改名脚本本身上和直接迁移的不同（演练用的是加上「挪旧目录下的文件」之后的脚本），ruff 与测试都过；已导入的提交哈希都没变，`imported/iclaw-core` 都是快进。没有演练到的情况：动到人工改动过的文件、要追加路径规则、新提交的说明里又有要处理的内容。
