#!/usr/bin/env python3
"""核对历史导入的结果，逐个提交对照来源仓库。

用法：verify_import.py <来源仓库> <来源提交> <导入结果仓库> [导入分支]

导入分支缺省为 refs/heads/imported/iclaw-core。前身仓库那一段的起点（predecessor_tip）
从同目录的 import.sh 里读。逐项检查，任何一项不过就以非零退出：
  1. 每个导入的提交都带且只带一条 Imported-from trailer，排在最后，指向来源提交的祖先；
  2. 作者、提交者的名字、邮箱、时间与原提交相同；
  3. 提交说明除了按 message_rewrites.tsv 改写的行、在 PR / issue 编号前插入来源仓库名、
     末尾加 Imported-from 以外，与原提交相同；原有的 trailer 一条不少、顺序不变；
  4. 每个导入的提交的整棵树，等于原提交的树按它所在那一支的路径规则过滤、改路径之后的结果
     （逐个文件比对模式与 blob 哈希）；
  5. 来源提交的祖先里，凡是改动过所列路径的非合并提交都已导入（两段分别算）；
     iClaw 布局那一段的提交都在 subtree 并入之后（没有误收别的并入历史）；
  6. 接合处：前身仓库最后一个导入提交的树，等于 subtree 并入提交按 iClaw 规则过滤后的树；
  7. 导入分支的头提交与来源提交按文件逐个对照（三类路径分别计数）。
改写表每一条用了几次也列出来。只读，不改动任何仓库。
"""

from __future__ import annotations

import collections
import difflib
import hashlib
import os
import re
import subprocess
import sys

NEUTRAL_ENV = dict(os.environ, GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, LC_ALL="C")
ASCII_WS = " \t\r\v\f"
HERE = os.path.dirname(os.path.abspath(__file__))


def git(repo: str, *args: str) -> bytes:
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, check=True, env=NEUTRAL_ENV).stdout


def load_rules(path: str) -> tuple[list[str], list[tuple[str, str]]]:
    filters: list[str] = []
    renames: list[tuple[str, str]] = []
    with open(path, "rb") as fh:
        for raw in fh:
            line = raw.rstrip(b"\r\n").decode("utf-8")
            if not line or line.startswith("#"):
                continue
            if "==>" in line:
                old, new = line.rsplit("==>", 1)
                renames.append((old, new))
            else:
                filters.append(line)
    return filters, renames


def load_rewrites(path: str) -> dict[str, str]:
    table: dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.rstrip("\n")
            if not raw or raw.startswith("#"):
                continue
            digest, _sep, replacement = raw.partition("\t")
            table[digest] = replacement
    return table


def matches(expr: str, path: str) -> bool:
    n = len(expr)
    return path.startswith(expr) and (expr.endswith("/") or len(path) == n or path[n : n + 1] == "/")


def new_name(path: str, rules: tuple[list[str], list[tuple[str, str]]]) -> str | None:
    filters, renames = rules
    if not any(matches(f, path) for f in filters):
        return None
    for old, new in renames:
        if matches(old, path):
            path = path.replace(old, new, 1)
    return path


def tree(repo: str, commit: str) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    for entry in git(repo, "ls-tree", "-r", "-z", "--full-tree", commit).split(b"\0"):
        if not entry:
            continue
        meta, path = entry.split(b"\t", 1)
        mode, _type, sha = meta.split(b" ")
        out[path.decode("utf-8", "surrogateescape")] = (mode.decode(), sha.decode())
    return out


def filtered_tree(repo: str, commit: str, rules) -> dict[str, tuple[str, str]]:
    out = {}
    for path, val in tree(repo, commit).items():
        nn = new_name(path, rules)
        if nn is not None:
            out[nn] = val
    return out


def message(repo: str, commit: str) -> str:
    raw = git(repo, "cat-file", "commit", commit)
    return raw.split(b"\n\n", 1)[1].decode("utf-8", "surrogateescape") if b"\n\n" in raw else ""


def people(repo: str, commit: str) -> str:
    return git(repo, "log", "-1", "--format=%an%x00%ae%x00%ad%x00%cn%x00%ce%x00%cd", "--date=raw", commit).decode()


def trailers(text: str) -> list[str]:
    out = subprocess.run(
        ["git", "interpret-trailers", "--parse", "--no-divider"],
        input=text.encode("utf-8", "surrogateescape"),
        capture_output=True,
        check=True,
        env=NEUTRAL_ENV,
    ).stdout.decode("utf-8", "surrogateescape")
    return [line for line in out.split("\n") if line]


def apply_rewrites(text: str, table: dict[str, str], usage: collections.Counter, log: list[str], tag: str) -> str:
    out: list[str] = []
    dropped = False
    for lineno, line in enumerate(text.split("\n"), 1):
        digest = hashlib.sha256(line.encode("utf-8", "surrogateescape")).hexdigest()
        if digest in table:
            usage[digest] += 1
            if table[digest] == "":
                log.append(f"{tag} 第 {lineno} 行 删除")
                dropped = True
                continue
            log.append(f"{tag} 第 {lineno} 行 替换")
            line = table[digest]
        if dropped and line.strip(ASCII_WS) == "" and out and out[-1].strip(ASCII_WS) == "":
            dropped = False
            continue
        dropped = False
        out.append(line)
    return "\n".join(out)


def only_ref_insertions(orig: str, new: str, prefix: str) -> list[str]:
    """new 相对 orig 只允许在「#数字」前插入来源仓库名；返回不合规的改动描述。"""
    problems = []
    sm = difflib.SequenceMatcher(a=orig, b=new, autojunk=False)
    for op, a1, a2, b1, b2 in sm.get_opcodes():
        if op == "equal":
            continue
        if op == "insert" and new[b1:b2] == prefix and re.match(r"#\d", new[b2 : b2 + 2]):
            continue
        problems.append(f"{op} {orig[a1:a2]!r} -> {new[b1:b2]!r}")
    return problems


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    src, src_sha, dst = sys.argv[1:4]
    branch = sys.argv[4] if len(sys.argv) > 4 else "refs/heads/imported/iclaw-core"
    import_sh = open(os.path.join(HERE, "import.sh"), encoding="utf-8").read()
    pred_tip = re.search(r'^predecessor_tip="([0-9a-f]{40})"', import_sh, re.M).group(1)
    subtree_dir = re.search(r'^subtree_dir="([^"]+)"', import_sh, re.M).group(1)
    rules = {
        "default": load_rules(os.path.join(HERE, "paths-iclaw.txt")),
        "predecessor": load_rules(os.path.join(HERE, "paths-predecessor.txt")),
    }
    prefixes = {"default": "iclaw", "predecessor": "neuroagent-python"}
    table = load_rewrites(os.path.join(HERE, "message_rewrites.tsv"))

    src_sha = git(src, "rev-parse", src_sha + "^{commit}").decode().strip()
    head = git(dst, "rev-parse", branch).decode().strip()
    ancestors = set(git(src, "rev-list", src_sha).decode().split())
    pred = set(git(src, "rev-list", pred_tip).decode().split())
    subtree_merges = [
        c for c in git(src, "log", "--format=%H", f"--grep=git-subtree-split: {pred_tip}", src_sha).decode().split()
    ]
    if len(subtree_merges) != 1:
        print(f"FAILED: 找到 {len(subtree_merges)} 个指向 {pred_tip} 的 subtree 并入提交")
        return 1
    subtree_merge = subtree_merges[0]

    imported = git(dst, "rev-list", "--topo-order", "--reverse", branch).decode().split()
    failures: list[str] = []
    mapping: dict[str, str] = {}
    section_of: dict[str, str] = {}
    usage: collections.Counter = collections.Counter()
    rewrite_log: list[str] = []
    print(f"source  {src_sha}")
    print(f"head    {head}")
    print(f"head tree {git(dst, 'rev-parse', head + '^{tree}').decode().strip()}")
    print(f"predecessor tip {pred_tip} ({len(pred)} commits); subtree merge {subtree_merge}")
    print(f"imported commits: {len(imported)}")

    for c in imported:
        new_msg = message(dst, c)
        found = re.findall(r"^Imported-from: iclaw@([0-9a-f]{40})$", new_msg, re.M)
        if len(found) != 1:
            failures.append(f"{c}: Imported-from 条数 {len(found)}")
            continue
        o = found[0]
        mapping[c] = o
        if o not in ancestors:
            failures.append(f"{c}: {o} 不是来源提交的祖先")
            continue
        section = "predecessor" if o in pred else "default"
        section_of[c] = section
        if section == "default":
            if subprocess.run(["git", "-C", src, "merge-base", "--is-ancestor", subtree_merge, o],
                              env=NEUTRAL_ENV).returncode != 0:
                failures.append(f"{c}: iClaw 段的原提交 {o} 不在 subtree 并入之后")
        trailer_line = f"Imported-from: iclaw@{o}\n"
        if not new_msg.endswith(trailer_line):
            failures.append(f"{c}: Imported-from 不在最后一行")
            continue
        new_trs = trailers(new_msg)
        if not new_trs or new_trs[-1] != trailer_line.strip():
            failures.append(f"{c}: git 没有把 Imported-from 认作 trailer")
        expected = apply_rewrites(message(src, o), table, usage, rewrite_log, f"{c[:12]}（原 {o[:12]}）")
        exp_trs = trailers(expected)
        prefix = prefixes[section]
        if [t.replace(prefix + "#", "#") for t in new_trs[:-1]] != [t.replace(prefix + "#", "#") for t in exp_trs]:
            failures.append(f"{c}: 原有 trailer 变了 {exp_trs} -> {new_trs[:-1]}")
        stripped = new_msg[: -len(trailer_line)].rstrip()
        bad = only_ref_insertions(expected.rstrip(), stripped, prefix)
        if bad:
            failures.append(f"{c}: 提交说明有预期以外的改动 {bad}")
        if people(dst, c) != people(src, o):
            failures.append(f"{c}: 作者或提交者与 {o} 不同")
        if filtered_tree(src, o, rules[section]) != tree(dst, c):
            failures.append(f"{c}: 树与 {o} 按 {section} 规则过滤后的结果不同")

    counts = collections.Counter(section_of.values())
    print(f"imported by section: predecessor {counts['predecessor']}, iclaw {counts['default']}")

    # 5. 覆盖
    imported_origs = set(mapping.values())
    for section, tip in (("predecessor", pred_tip), ("default", src_sha)):
        pathspecs = [f.rstrip("/") for f in rules[section][0]]
        touching = set(git(src, "rev-list", "--full-history", "--no-merges", tip, "--", *pathspecs).decode().split())
        if section == "default":
            touching -= pred
        else:
            touching &= pred
        missing = sorted(touching - imported_origs)
        print(f"[{section}] non-merge source commits touching the paths: {len(touching)}; missing: {len(missing)}")
        if missing:
            failures.append(f"[{section}] 改动过所列路径、却没有导入的提交：{missing}")
    merges = [o for o in imported_origs if len(git(src, "rev-list", "--parents", "-n", "1", o).decode().split()) > 2]
    print(f"imported commits whose original is a merge: {len(merges)}")
    for o in sorted(merges):
        kept_as = [c for c, oo in mapping.items() if oo == o][0]
        nnp = len(git(dst, "rev-list", "--parents", "-n", "1", kept_as).decode().split()) - 1
        print(f"  {o[:12]} -> 导入后 {nnp} 个父提交  {git(src, 'log', '-1', '--format=%s', o).decode().strip()}")
    pruned_merge = subtree_merge not in imported_origs
    print(f"subtree merge {subtree_merge[:12]} imported: {not pruned_merge}")

    # 6. 接合处
    core_a = git(src, "rev-parse", f"{pred_tip}:packages/harness").decode().strip()
    core_b = git(src, "rev-parse", f"{subtree_merge}:{subtree_dir}/packages/harness").decode().strip()
    print(f"junction core tree: predecessor tip {core_a} / subtree merge {core_b}")
    if core_a != core_b:
        failures.append("接合处：前身仓库最后的核心树与 subtree 并入时的核心树不同")
    last_pred = [c for c in imported if section_of.get(c) == "predecessor"]
    if last_pred:
        last_pred_c = last_pred[-1]
        expect = filtered_tree(src, subtree_merge, rules["default"])
        same = tree(dst, last_pred_c) == expect
        print(f"junction: last predecessor commit {last_pred_c[:12]} tree == subtree merge filtered by iClaw rules: {same}")
        if not same:
            failures.append("接合处：前身仓库最后一个导入提交的树与 subtree 并入提交过滤后的树不同")
        first_default = [c for c in imported if section_of.get(c) == "default"][0]
        parents = git(dst, "rev-list", "--parents", "-n", "1", first_default).decode().split()[1:]
        print(f"junction: first iClaw commit {first_default[:12]} parents {[p[:12] for p in parents]}")
        if parents != [last_pred_c]:
            failures.append("接合处：iClaw 段第一个提交的父提交不是前身仓库最后一个导入提交")

    # 改写表
    print(f"rewrite table: {len(table)} entries; applications: {sum(usage.values())}")
    for digest in sorted(table):
        n = usage[digest]
        flag = "" if n == 1 else ("  ← 没用到" if n == 0 else "  ← 用到不止一次")
        print(f"  {digest[:16]}… {n}{flag}")
        if n == 0:
            failures.append(f"改写表里有没用到的条目 {digest[:16]}…")
    for line in rewrite_log:
        print(f"  {line}")

    # 7. 头提交逐文件对照
    src_tree = tree(src, src_sha)
    head_tree = tree(dst, head)
    classes = {"core": 0, "tests": 0, "docs": 0}
    listing = []
    for path, val in sorted(src_tree.items()):
        nn = new_name(path, rules["default"])
        if nn is None:
            continue
        if head_tree.get(nn) != val:
            failures.append(f"头提交缺文件或内容不同：{path} -> {nn}")
        cls = "tests" if nn.startswith("tests/") else ("docs" if nn.startswith("packages/core/docs/") else "core")
        classes[cls] += 1
        listing.append(f"{val[0]} {val[1]} {path} -> {nn}")
    if len(listing) != len(head_tree):
        failures.append(f"头提交文件数 {len(head_tree)} 与来源过滤后的 {len(listing)} 不同")
    digest = hashlib.sha256("\n".join(listing).encode()).hexdigest()
    print(f"head files: {len(head_tree)} (core {classes['core']}, tests {classes['tests']}, docs {classes['docs']})")
    print(f"head listing sha256: {digest}")
    for line in listing:
        print("  " + line)

    if failures:
        print(f"FAILED ({len(failures)})")
        for f in failures:
            print("  " + f)
        return 1
    print("OK: 全部检查通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
