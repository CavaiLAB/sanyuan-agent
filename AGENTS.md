# AGENTS.md：给编码代理的入口说明

这个仓库是 **SanYuanAgent** 的新家：匠心智能体平台的智能体引擎（Python）。**目前只有框架级文档，没有代码。**

## 要改代码

代码还在 iClaw 仓库的 Python Agent 引擎目录里，现在的名字是 NeuroAgent。去那边改，按那边的 `AGENTS.md` 办。

不要在本仓库里新建代码目录，也不要把原仓库的代码复制过来。代码带着历史一次迁入，迁入时改名。

## 要改文档

1. 先读 [README.md](README.md) 与平台总览仓库的[名称对照](https://github.com/CavaiLAB/jiangxin/blob/main/docs/glossary.md)。
2. 文档用中文，不写参考平台和对标项目的名字。
3. 分清「已定」「提议」「待定」。发行名、许可、是否公开都是待定，不写成已定。
4. 说现状的时候写清是「已有」（在原仓库里已经实现）还是「规划」。原仓库是私有的：只写框架层面的内容，不写它的内部实现细节、文件路径和安全相关的细节。
5. 平台级的口径（名称、三个框架的分工、契约、Agent 协同）在 [jiangxin](https://github.com/CavaiLAB/jiangxin) 里改，这里只引用。
6. 改动走分支和 PR，不直接推 `main`。
