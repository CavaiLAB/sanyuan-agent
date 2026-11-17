#!/bin/bash
# 历史导入：从来源仓库的一个提交过滤出核心的历史，放进一个新建的裸仓库，
# 结果在其中的 imported/iclaw-core 分支上。脚本只读来源仓库，不往任何远端推送。
#
# 用法：tools/migration/import/import.sh <输出目录> [来源提交] [来源仓库地址]
#   输出目录   必须还不存在
#   来源提交   缺省为第一次导入用的 b0b6fd5f0ffb4458bc9b7a247b6da24bfa7a9b92
#   来源仓库   缺省为 git@github.com:idadawn/iclaw.git（私有），也可以是本机路径
#
# 来源里的核心有两段历史：2026-05-11 以前在前身仓库 neuroagent-python 里（以 git subtree
# 并入 iClaw 的 services/python/），之后在 iClaw 里。两段都带：前身仓库的提交按
# paths-predecessor.txt 过滤，其余提交按 paths-iclaw.txt 过滤（由 commit_callback.py
# 按提交所在的那一支选规则）。
#
# 做法：全新裸克隆（不带标签）→ 只留一个指向来源提交的分支 → git filter-repo 按
#   commit_callback.py 过滤并改路径、按 message_rewrites.tsv 改写个别提交说明、改写
#   PR 编号、加 Imported-from trailer；不改文件内容，作者、提交者与时间原样保留。
# 同样的输入、同样版本的 git 与 git-filter-repo，得到的提交哈希相同。
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="${1:?用法: import.sh <输出目录> [来源提交] [来源仓库地址]}"
src_sha="${2:-b0b6fd5f0ffb4458bc9b7a247b6da24bfa7a9b92}"
src_url="${3:-git@github.com:idadawn/iclaw.git}"
branch="imported/iclaw-core"
# 前身仓库并入 iClaw 时的 git-subtree-split 提交；它和它的祖先就是前身仓库那一段
predecessor_tip="10532187bc3fd55623606f1a795950bdd1f26a7a"
subtree_dir="services/python"

if [ -e "$out" ]; then
  echo "输出目录已存在：$out" >&2
  exit 1
fi
command -v git-filter-repo >/dev/null || { echo "缺少 git-filter-repo" >&2; exit 1; }

git clone --quiet --bare --no-local --no-tags "$src_url" "$out"
git -C "$out" cat-file -e "${src_sha}^{commit}"
src_full="$(git -C "$out" rev-parse "${src_sha}^{commit}")"

# 前身仓库那一段：核对并入记录只有一次、指向的正是 predecessor_tip，再列出这一段的提交
splits="$(git -C "$out" log --format='%(trailers:key=git-subtree-dir,valueonly,separator=%x2C)%x09%(trailers:key=git-subtree-split,valueonly,separator=%x2C)' "$src_full" \
  | awk -F'\t' -v d="$subtree_dir" '$1 == d { print $2 }' | sort -u)"
if [ "$splits" != "$predecessor_tip" ]; then
  echo "来源里 ${subtree_dir} 的 subtree 并入记录与预期不符：${splits:-（没有）}" >&2
  exit 1
fi
pred_list="$(mktemp)"
trap 'rm -f "$pred_list"' EXIT
git -C "$out" rev-list "$predecessor_tip" > "$pred_list"

# 只留一个分支，指向来源提交；其余引用（分支、标签、远端）一律删掉
git -C "$out" update-ref "refs/heads/$branch" "$src_full"
git -C "$out" symbolic-ref HEAD "refs/heads/$branch"
git -C "$out" for-each-ref --format='%(refname)' | while read -r ref; do
  if [ "$ref" != "refs/heads/$branch" ]; then
    git -C "$out" update-ref -d "$ref"
  fi
done
git -C "$out" remote remove origin

# --force：上面改过引用，filter-repo 不再把它当「全新克隆」，要显式放行。
# --preserve-commit-hashes：提交说明里的旧哈希不改写（Imported-from 给出对照）。
# --prune-empty always / --prune-degenerate always：过滤后没有改动的提交与退化的合并
#   一律去掉，原仓库里本来就是空提交的也去掉（缺省的 auto 会留下它们）。
IMPORT_RULES_DEFAULT="$here/paths-iclaw.txt" \
IMPORT_RULES_PREDECESSOR="$here/paths-predecessor.txt" \
IMPORT_PREDECESSOR_COMMITS="$pred_list" \
IMPORT_MESSAGE_REWRITES="$here/message_rewrites.tsv" \
git -C "$out" filter-repo --force --quiet \
  --commit-callback "$(cat "$here/commit_callback.py")" \
  --preserve-commit-hashes \
  --prune-empty always \
  --prune-degenerate always \
  --replace-refs delete-no-add

echo "source $src_full"
echo "imported $(git -C "$out" rev-parse "refs/heads/$branch")"
