"""知识库问题分类与文本清洗。"""
from __future__ import annotations

import re

# 只问「有哪些文档/文件」—— 列目录，不总结知识
_KB_CATALOG = re.compile(
    r"(知识库|库里|库中).{0,8}(有(哪些|什么)(文档|文件|论文|文献|资料))|"
    r"(列出|目录).{0,6}(文档|文件|论文|文献)|"
    r"有(多少|几)(篇|个)(文档|论文|文献)|"
    r"(上传|现有).{0,6}(了)?(哪些|什么)(文档|文件|论文)|"
    r"(说说|说一说|介绍|描述).{0,12}(知识库|库里).{0,8}(里|中)?(的)?(内容|有什么|有哪些)|"
    r"(知识库|库里).{0,6}(里|中).{0,6}(有|的).{0,4}内容"
)

# 要求「总结/概括实质知识」—— 综合检索后归纳，不是列文件名
_KB_SUMMARY = re.compile(
    r"(总结|概括|归纳|提炼|梳理|综述).{0,12}(核心)?(知识|内容|观点|结论|要点|发现)|"
    r"(核心|主要)(知识|观点|内容|结论|发现|方法)|"
    r"(上传|已有|现有).{0,12}(文献|资料|论文|知识库).{0,8}(总结|概括|归纳|核心)|"
    r"知识库.{0,8}(总结|概括|归纳).{0,8}(知识|内容|核心)"
)


def is_kb_catalog_question(question: str) -> bool:
    """用户只想看文档列表/目录。"""
    q = question.strip()
    if _KB_SUMMARY.search(q):
        return False  # 「总结知识」优先于「列目录」
    return bool(_KB_CATALOG.search(q))


def is_kb_knowledge_summary(question: str) -> bool:
    """用户要的是知识实质的综合总结。"""
    return bool(_KB_SUMMARY.search(question.strip()))


def is_kb_meta_question(question: str) -> bool:
    """兼容旧调用：目录类或（已废弃）广义元问题。"""
    return is_kb_catalog_question(question)


def normalize_chunk(text: str) -> str:
    if not text:
        return text
    cn = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    if cn > 10 and text.count(" ") > cn * 0.25:
        return text.replace(" ", "")
    return text


def build_kb_overview_answer() -> str:
    """文档目录（仅列清单，不做知识归纳）。"""
    from src.rag.retriever import retriever

    sources = retriever.list_sources()
    total = retriever.count()
    if not sources:
        return "知识库当前为空。请在左侧上传 PDF 并点击「确认入库」。"

    lines = [
        f"知识库共有 **{len(sources)}** 篇文档、**{total}** 个文本块。",
        "",
        "### 文档列表",
        "",
    ]
    for s in sources:
        lines.append(f"- **{s['filename']}**（{s['chunks']} 块）")
    lines.append("")
    lines.append("_如需了解文献中的核心观点，请提问：「总结知识库中的核心知识」_")
    return "\n".join(lines)
