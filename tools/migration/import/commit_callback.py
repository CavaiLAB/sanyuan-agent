# 历史导入时 git filter-repo 对每个提交调用的回调（--commit-callback 的函数体）。
# filter-repo 会把这段代码包进 def callback(commit, ...) 里执行，所以不能用多行字符串
# （每行都会被加缩进）。配置文件的位置由 import.sh 用环境变量传进来。
#
# 对每个提交做四件事，不碰文件内容、作者、提交者与时间：
# 1. 按提交所在的那一支选路径规则：前身仓库（neuroagent-python）的提交用
#    IMPORT_RULES_PREDECESSOR，其余的用 IMPORT_RULES_DEFAULT（iClaw 布局）；
#    只留规则里列的路径并改到新位置，语义照 filter-repo 的 --paths-from-file。
#    这样前身布局的规则不会误匹配到 iClaw 主线上同名的顶层路径。
# 2. 提交说明逐行查 IMPORT_MESSAGE_REWRITES 表（原行内容的 SHA-256 → 替换后的行，空表示删除）；
#    删掉一段后不留两个连着的空行。
# 3. 会被 GitHub 自动识别成链接的 PR / issue 编号，改成不会被识别的写法，前缀用来源仓库名
#    （iClaw 的提交用 iclaw，前身仓库的提交用 neuroagent-python）：
#    - 指向 GitHub 的 PR / issue 网址：https://github.com/<所有者>/<仓库>/pull/123 → <仓库>#123
#    - 跨仓库写法：<所有者>/<仓库>#123 → <仓库>#123
#    - GH-123 → <前缀>#123
#    - 裸的 #123 → <前缀>#123；前面紧挨字母、数字、下划线、/、&、#、- 的不动
#      （比如已经是 iclaw-deploy#5 这种写法的，或 HTML 实体 &#123;）
# 4. 加一条 trailer：Imported-from: iclaw@<原提交的完整 SHA>（前身仓库的提交在 iClaw 的
#    历史里同样够得着）。最后一段已经是 trailer 块就接在块尾，否则另起一段。trailer 块的判定
#    照 git 自己的规则（trailer.c 的 find_trailer_block_start）：第一段是标题、不算；从最后一行
#    往上数，全是 trailer 行，或者有 git 生成的前缀行且 trailer 行不少于四分之一，就是 trailer 块。
#    原有的 Co-authored-by 等 trailer 留在原处、仍在 trailer 块里。
import hashlib
import os
import re

G = globals()
if '_import_cfg' not in G:
    def _load_rules(path):
        filters, renames = [], []
        with open(path, 'rb') as fh:
            for raw in fh:
                line = raw.rstrip(b'\r\n')
                if not line or line.startswith(b'#'):
                    continue
                if b'==>' in line:
                    old, new = line.rsplit(b'==>', 1)
                    renames.append((old, new))
                else:
                    filters.append(line)
        return filters, renames

    def _load_rewrites(path):
        table = {}
        with open(path, encoding='utf-8') as fh:
            for raw in fh:
                raw = raw.rstrip('\n')
                if not raw or raw.startswith('#'):
                    continue
                digest, _sep, replacement = raw.partition('\t')
                table[digest] = replacement
        return table

    with open(os.environ['IMPORT_PREDECESSOR_COMMITS'], 'rb') as fh:
        _pred = set(fh.read().split())
    G['_import_cfg'] = {
        'default': _load_rules(os.environ['IMPORT_RULES_DEFAULT']),
        'predecessor': _load_rules(os.environ['IMPORT_RULES_PREDECESSOR']),
        'pred_commits': _pred,
        'rewrites': _load_rewrites(os.environ['IMPORT_MESSAGE_REWRITES']),
    }
cfg = G['_import_cfg']


def _matches(expr, path):
    n = len(expr)
    return path.startswith(expr) and (expr.endswith(b'/') or len(path) == n or path[n:n + 1] == b'/')


# 1. 路径
is_pred = commit.original_id in cfg['pred_commits']
filters, renames = cfg['predecessor'] if is_pred else cfg['default']
kept = {}
for change in commit.file_changes:
    if change.type not in (b'M', b'D'):
        raise SystemExit('unexpected file change type %r in %s' % (change.type, commit.original_id))
    path = change.filename
    if not any(_matches(f, path) for f in filters):
        continue
    for old, new in renames:
        if _matches(old, path):
            path = path.replace(old, new, 1)
    if path in kept:
        raise SystemExit('path collision %r in %s' % (path, commit.original_id))
    change.filename = path
    kept[path] = change
commit.file_changes = list(kept.values())

# 2. 逐行改写
ascii_ws = ' \t\r\v\f'  # git 用 C 的 isspace，只认 ASCII 空白
out = []
dropped = False
for line in commit.message.decode('utf-8', 'surrogateescape').split('\n'):
    digest = hashlib.sha256(line.encode('utf-8', 'surrogateescape')).hexdigest()
    if digest in cfg['rewrites']:
        replacement = cfg['rewrites'][digest]
        if replacement == '':
            dropped = True
            continue
        line = replacement
    if dropped and line.strip(ascii_ws) == '' and out and out[-1].strip(ascii_ws) == '':
        dropped = False
        continue
    dropped = False
    out.append(line)
msg = '\n'.join(out)

# 3. PR / issue 编号
prefix = 'neuroagent-python' if is_pred else 'iclaw'
msg = re.sub(r'https?://github\.com/[A-Za-z0-9-]+/([A-Za-z0-9._-]+)/(?:pull|issues)/(\d+)\b',
             r'\1#\2', msg)
msg = re.sub(r'(?<![\w.-])[A-Za-z0-9-]+/([A-Za-z0-9._-]+)#(\d+)\b', r'\1#\2', msg)
msg = re.sub(r'(?<![\w-])[Gg][Hh]-(\d+)\b', prefix + r'#\1', msg)
msg = re.sub(r'(?<![\w/&#-])#(\d+)\b', prefix + r'#\1', msg)

# 4. trailer
body = msg.rstrip(ascii_ws + '\n')
lines = body.split('\n')

# 标题段：从头到第一个空白行（以 # 开头的注释行跳过），对应 git 的 end_of_title。
title_end = len(lines)
for i, line in enumerate(lines):
    if line.startswith('#'):
        continue
    if line.strip(ascii_ws) == '':
        title_end = i
        break

is_trailer_block = False
trailer_lines = 0
non_trailer_lines = 0
possible_continuation_lines = 0
recognized_prefix = False
for i in range(len(lines) - 1, title_end - 1, -1):
    line = lines[i]
    if line.startswith('#'):
        non_trailer_lines += possible_continuation_lines
        possible_continuation_lines = 0
        continue
    if line.strip(ascii_ws) == '':
        non_trailer_lines += possible_continuation_lines
        if recognized_prefix and trailer_lines * 3 >= non_trailer_lines:
            is_trailer_block = True
        elif trailer_lines and not non_trailer_lines:
            is_trailer_block = True
        break
    if line.startswith('Signed-off-by: ') or line.startswith('(cherry picked from commit '):
        trailer_lines += 1
        possible_continuation_lines = 0
        recognized_prefix = True
        continue
    if line[:1] not in tuple(ascii_ws) and re.match(r'[A-Za-z0-9-]+[ \t]*:', line):
        trailer_lines += 1
        possible_continuation_lines = 0
    elif line[:1] and line[:1] in ascii_ws:
        possible_continuation_lines += 1
    else:
        non_trailer_lines += 1 + possible_continuation_lines
        possible_continuation_lines = 0

trailer = 'Imported-from: iclaw@' + commit.original_id.decode('ascii')
if is_trailer_block:
    body = body + '\n' + trailer + '\n'
else:
    body = body + '\n\n' + trailer + '\n'

commit.message = body.encode('utf-8', 'surrogateescape')
