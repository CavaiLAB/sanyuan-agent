"""残留检查：旧名 neuroagent 在仓库里还剩的每一处，都必须落在 exceptions.tsv 登记的写法里。

旧名不分大小写，连带 neuro-agent、neuro_agent、neuro agent 这几种写法。
查的范围是 git 跟踪的全部文件：文件路径里不许出现旧名；文本内容逐行查，二进制文件跳过。
有下面任何一种情况就失败（退出码 1）：
- 有没登记的旧名（逐处列出 路径:行号: 那一行）；
- exceptions.tsv 里有条目一处都没匹配到（清单要跟着代码走，过时的条目要删掉）。

用法（在仓库根目录）：python3 tools/migration/rename/check_residue.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from exception_list import load_exceptions

OLD = re.compile(r"neuro[-_ ]?agent", re.IGNORECASE)


def main() -> int:
    root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    entries = load_exceptions()
    used = {e.lineno: 0 for e in entries}
    problems: list[str] = []
    total = 0

    listing = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"], check=True, capture_output=True
    ).stdout.decode("utf-8")
    for path in (p for p in listing.split("\0") if p):
        if OLD.search(path):
            problems.append(f"{path}: 文件路径里有旧名")
        file = root / path
        if not file.is_file():
            continue
        data = file.read_bytes()
        if b"\0" in data[:8192]:
            continue
        applicable = [e for e in entries if e.applies_to(path)]
        for lineno, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
            hits = list(OLD.finditer(line))
            if not hits:
                continue
            spans = [(m.start(), m.end(), e.lineno) for e in applicable for m in e.pattern.finditer(line)]
            for hit in hits:
                total += 1
                covering = [s for s in spans if s[0] <= hit.start() and hit.end() <= s[1]]
                for s in covering:
                    used[s[2]] += 1
                if not covering:
                    problems.append(f"{path}:{lineno}: {line.strip()[:200]}")

    stale = [e for e in entries if used[e.lineno] == 0]
    for e in stale:
        problems.append(f"exceptions.tsv:{e.lineno}: 这一条一处都没匹配到（{e.path}  {e.pattern.pattern}）")

    if problems:
        print("残留检查失败：下面这些旧名没有登记在 tools/migration/rename/exceptions.tsv 里，或者登记的条目已经过时。")
        print("要么改掉它，要么在 exceptions.tsv 里登记写法、类别、原因和以后由谁定。")
        for p in problems:
            print("  " + p)
        return 1
    print(f"残留检查通过：旧名还剩 {total} 处，全部落在 exceptions.tsv 登记的 {len(entries)} 条写法里。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
