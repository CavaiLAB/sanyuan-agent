"""exceptions.tsv 的读取，rename.py 与 check_residue.py 共用。

exceptions.tsv 每行五列，用制表符分开：
  路径      fnmatch 写法，* 可以跨目录；写改名之后的路径
  写法      Python 正则，区分大小写（要忽略大小写就写 (?i)）
  类别      为什么不改，归到哪一类
  原因      具体原因
  由谁定    以后由谁决定改不改
以 # 开头的行和空行跳过。
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path

TSV = Path(__file__).resolve().parent / "exceptions.tsv"


@dataclass(frozen=True)
class Entry:
    lineno: int
    path: str
    pattern: re.Pattern[str]
    category: str
    reason: str
    decide: str

    def applies_to(self, path: str) -> bool:
        return fnmatch.fnmatchcase(path, self.path)


def load_exceptions(tsv: Path = TSV) -> list[Entry]:
    entries: list[Entry] = []
    for lineno, raw in enumerate(tsv.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.startswith("#"):
            continue
        cols = raw.split("\t")
        if len(cols) != 5 or not all(c.strip() for c in cols):
            raise SystemExit(f"{tsv}:{lineno}: 每行要五列、都不能空（实际 {len(cols)} 列）")
        path, pattern, category, reason, decide = cols
        entries.append(Entry(lineno, path, re.compile(pattern), category, reason, decide))
    return entries
