"""迁入之后的改名。规则是 2026-11-17 定的：

- 发行名 neuroagent → sanyuan-agent，导入名 neuroagent → sanyuan_agent，命令名 neuroagent → sanyuan-agent；
- 环境变量前缀 NEUROAGENT_ → SANYUAN_AGENT_，不再读旧前缀；
- 品牌 NeuroAgent → SanYuanAgent；工作区根的项目名 neuroagent-workspace → sanyuan-agent-workspace。

可以重复跑，第二次跑不会再改任何东西：
1. 包目录 packages/core/neuroagent 用 git mv 改成 packages/core/sanyuan_agent（已经改过就跳过）。
2. 对 SCOPE 里 git 跟踪的文本文件逐行替换：
   a. 先把 exceptions.tsv 里登记、适用于这个文件的写法遮住，替换完再放回，所以它们原样保留；
   b. 再按 CONTEXT_RULES 处理要看上下文的地方（发行名、命令名、仓库名、布局路径）；
   c. 最后按 GENERIC_RULES 做通用替换。
   不在 SCOPE 里的不碰：tools/migration/（这些脚本本身要写旧名）、uv.lock（改完用 uv lock 重新生成）、
   根目录的 README.md、AGENTS.md 和 docs/（迁移说明要写原名，人工改）。
3. uv lock 重新生成锁文件（只有包名变化）。
4. ruff 按仓库的 isort 规则重排 packages/ 与 tests/ 里的 import：改名后 sanyuan_agent 在字母序上
   排到了 pydantic 等包的后面，不重排 ruff 会报 I001。只动 import 的先后，不改别的。
改完用 check_residue.py 查剩下的旧名：每一处都必须登记在 exceptions.tsv 里。

用法（在仓库根目录）：
  python3 tools/migration/rename/rename.py            改名
  python3 tools/migration/rename/rename.py --dry-run  只列出会改的文件，不动工作区（不跑第 3、4 步）
需要 git 与 uv；第 4 步用锁文件里的 ruff 版本。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from exception_list import Entry, load_exceptions

PKG_OLD = "packages/core/neuroagent"
PKG_NEW = "packages/core/sanyuan_agent"
SCOPE = ("packages/", "tests/", "pyproject.toml")

# （路径, 原文, 改成, 为什么）——按顺序做，路径是改名之后的路径
CONTEXT_RULES: list[tuple[str, str, str, str]] = [
    ("pyproject.toml", 'name = "neuroagent-workspace"', 'name = "sanyuan-agent-workspace"', "工作区根的项目名"),
    ("pyproject.toml", "neuroagent = { workspace = true }", "sanyuan-agent = { workspace = true }", "工作区成员按发行名引用"),
    ("packages/core/pyproject.toml", 'name = "neuroagent"', 'name = "sanyuan-agent"', "发行名"),
    ("packages/core/pyproject.toml", 'neuroagent = "neuroagent.cli:app"', 'sanyuan-agent = "sanyuan_agent.cli:app"', "命令名"),
    ("packages/core/sanyuan_agent/cli.py", 'f"neuroagent {__version__}"', 'f"sanyuan-agent {__version__}"', "--version 输出里的命令名"),
    ("packages/core/tests/test_harness_standalone.py", '["neuroagent", "--version"]', '["sanyuan-agent", "--version"]', "命令名"),
    ("packages/core/tests/test_harness_standalone.py", 'assert "neuroagent" in result.stdout', 'assert "sanyuan-agent" in result.stdout', "命令名"),
    ("packages/core/README.md", "# neuroagent (Harness)", "# sanyuan-agent (Harness)", "发行名"),
    ("packages/core/README.md", "uv add neuroagent", "uv add sanyuan-agent", "发行名"),
    ("packages/core/README.md", "neuroagent --config", "sanyuan-agent --config", "命令名"),
    ("packages/core/README.md", "`neuroagent desktop-run`", "`sanyuan-agent desktop-run`", "命令名"),
    ("packages/core/README.md", "`neuroagent-python` 仓库", "`sanyuan-agent` 仓库", "仓库名"),
    ("packages/core/AGENTS.md", "../../../../docs/services/python/deerflow-reference.md", "docs/deerflow-reference.md", "这份文档随导入挪到了核心目录下"),
    ("packages/core/docs/deerflow-reference.md", "packages/harness/neuroagent-python/NOTICE", "packages/core/NOTICE", "NOTICE 在新布局里的位置"),
    ("packages/core/docs/deerflow-reference.md", "NeuroAgent `packages/harness/`", "SanYuanAgent `packages/core/`", "布局路径"),
    ("packages/core/docs/deerflow-reference.md", "packages/harness/neuroagent", "packages/core/sanyuan_agent", "布局路径；上游自己的 packages/harness/deerflow 不动"),
]

GENERIC_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"NEUROAGENT_"), "SANYUAN_AGENT_"),
    (re.compile(r"NeuroAgent"), "SanYuanAgent"),
    (re.compile(r"\bneuroagent\b"), "sanyuan_agent"),
]


def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def rename_line(line: str, entries: list[Entry], context: list[tuple[str, str]]) -> str:
    masked: list[str] = []

    def mask(m: re.Match[str]) -> str:
        masked.append(m.group(0))
        return f"\0{len(masked) - 1}\0"

    for e in entries:
        line = e.pattern.sub(mask, line)
    for old, new in context:
        line = line.replace(old, new)
    for pattern, new in GENERIC_RULES:
        line = pattern.sub(new, line)
    return re.sub(r"\0(\d+)\0", lambda m: masked[int(m.group(1))], line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="只列出会改的文件，不动工作区")
    args = parser.parse_args()

    root = Path(git("rev-parse", "--show-toplevel").strip())
    entries = load_exceptions()

    moved = bool(git("-C", str(root), "ls-files", PKG_NEW).strip())
    if not moved:
        if (root / PKG_NEW).exists():
            raise SystemExit(f"{PKG_NEW} 已经存在但没有被 git 跟踪，先清掉再跑")
        if args.dry_run:
            print(f"会 git mv {PKG_OLD} {PKG_NEW}")
        else:
            git("-C", str(root), "mv", PKG_OLD, PKG_NEW)
            print(f"git mv {PKG_OLD} {PKG_NEW}")
            moved = True

    changed: list[str] = []
    for path in git("-C", str(root), "ls-files", "-z").split("\0"):
        if not path or not path.startswith(SCOPE):
            continue
        # 预演时包目录还没挪，规则与例外按挪过之后的路径找
        logical = path if moved else path.replace(PKG_OLD + "/", PKG_NEW + "/", 1)
        file = root / path
        data = file.read_bytes()
        if b"\0" in data[:8192]:
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            print(f"跳过（不是 UTF-8）：{path}", file=sys.stderr)
            continue
        applicable = [e for e in entries if e.applies_to(logical)]
        context = [(old, new) for p, old, new, _why in CONTEXT_RULES if p == logical]
        new_text = "".join(rename_line(line, applicable, context) for line in text.splitlines(keepends=True))
        if new_text != text:
            changed.append(logical)
            if not args.dry_run:
                file.write_bytes(new_text.encode("utf-8"))

    verb = "会改" if args.dry_run else "改了"
    print(f"{verb} {len(changed)} 个文件")
    for p in changed:
        print("  " + p)
    if args.dry_run:
        return 0

    subprocess.run(["uv", "lock", "--quiet"], cwd=root, check=True)
    sorter = subprocess.run(
        ["uv", "run", "--locked", "--quiet", "ruff", "check", "--select", "I", "--fix", "--quiet", "packages", "tests"],
        cwd=root,
        check=False,
    )
    if sorter.returncode != 0:
        raise SystemExit("ruff 重排 import 之后仍有问题，见上面的输出")
    print("已重新生成 uv.lock，并按 isort 规则重排了 import")
    return 0


if __name__ == "__main__":
    sys.exit(main())
