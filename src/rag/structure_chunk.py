"""
src/rag/structure_chunk.py — 论文结构化切块

在语义/滑窗之前按文档结构切：
  - 摘要整块保留
  - 章节标题分段
  - 表格转为 Markdown 块（表题+表体同块）
  - 参考文献丢弃或低权重
  - 页眉页脚/页码等垃圾行过滤
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from config.settings import settings
from src.rag.chunking import hybrid_chunk_text
from src.utils.logger import get_logger

log = get_logger("rag.structure")

# 章节标题（行首）
_SECTION_HEAD = re.compile(
    r"^(?:"
    r"(?:\d+(?:\.\d+)*\.?\s+)?"
    r"(?:"
    r"abstract|摘要|关键\s*词|keywords?|"
    r"introduction|引言|前言|"
    r"related\s+work|相关工作|"
    r"materials?\s+and\s+methods?|methodology|方法|材料与方法|"
    r"results?(?:\s+and\s+discussion)?|实验结果|结果(?:与讨论)?|"
    r"discussion|讨论|"
    r"conclusions?|结论|总结|"
    r"acknowledg(?:e)?ments?|致谢|"
    r"references?|bibliography|参考文献|引用文献"
    r")"
    r")\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# 页码、页眉页脚类垃圾
_GARBAGE_LINE = re.compile(
    r"^(?:"
    r"\d{1,4}"  # 纯页码
    r"|page\s*\d+(\s*of\s*\d+)?"
    r"|第\s*\d+\s*页"
    r"|https?://\S+"
    r"|doi:\s*\S+"
    r"|©.+"
    r"|all\s+rights\s+reserved"
    r"|downloaded\s+from.+"
    r"|authorized\s+licensed\s+use.+"
    r")$",
    re.I,
)

_TABLE_CAPTION = re.compile(
    r"^(?:table|tab\.?|表)\s*[\d一二三四五六七八九十]+[.:：、]?\s*.+$",
    re.I,
)


@dataclass
class StructuredChunk:
    text: str
    section: str = "body"
    weight: float = 1.0


def _norm_section(title: str) -> str:
    t = title.strip().lower()
    t = re.sub(r"^\d+(?:\.\d+)*\.?\s*", "", t)
    t = t.rstrip(":：").strip()
    mapping = [
        (("abstract", "摘要"), "abstract"),
        (("keyword", "关键"), "keywords"),
        (("introduction", "引言", "前言"), "intro"),
        (("related", "相关工作"), "related"),
        (("method", "材料与方法", "方法"), "method"),
        (("result", "结果"), "result"),
        (("discussion", "讨论"), "discussion"),
        (("conclusion", "结论", "总结"), "conclusion"),
        (("acknowledg", "致谢"), "ack"),
        (("reference", "bibliography", "参考文献", "引用文献"), "reference"),
    ]
    for keys, name in mapping:
        if any(k in t for k in keys):
            return name
    return "body"


def filter_garbage_lines(text: str) -> str:
    """去掉页码、页眉页脚、下载声明等低信息行。"""
    if not text:
        return ""
    keep: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            keep.append("")
            continue
        if _GARBAGE_LINE.match(s):
            continue
        # 过短且几乎无字母汉字
        alnum = sum(1 for c in s if c.isalnum() or "\u4e00" <= c <= "\u9fff")
        if len(s) <= 3 and alnum <= 1:
            continue
        keep.append(line)
    # 压缩多余空行
    out = re.sub(r"\n{3,}", "\n\n", "\n".join(keep)).strip()
    return out


def _split_by_headings(text: str) -> list[tuple[str, str]]:
    """按章节标题切成 (section_name, body) 列表；无标题则整篇 body。"""
    matches = list(_SECTION_HEAD.finditer(text))
    if not matches:
        return [("body", text.strip())]

    parts: list[tuple[str, str]] = []
    # 标题前的前言
    pre = text[: matches[0].start()].strip()
    if pre:
        parts.append(("front", pre))

    for i, m in enumerate(matches):
        title = m.group(0).strip()
        sec = _norm_section(title)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        # 保留标题行，便于检索定位
        block = f"{title}\n{body}".strip() if body else title
        parts.append((sec, block))
    return parts


def _column_count(line: str) -> int:
    s = line.strip()
    if "|" in s and s.count("|") >= 2:
        return len([c for c in s.strip("|").split("|")])
    parts = [c for c in re.split(r"\s{2,}|\t+", s) if c.strip()]
    return len(parts)


def _looks_like_table_row(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    cols = _column_count(s)
    if cols < 3:
        return False
    if "|" in s:
        return True
    # 多列空白分隔：含数字的数据行，或纯表头行
    parts = [c for c in re.split(r"\s{2,}|\t+", s) if c.strip()]
    nums = sum(1 for p in parts if re.search(r"\d", p))
    return nums >= 1 or cols >= 3


def _consume_table_rows(lines: list[str], start: int) -> tuple[list[str], int]:
    """从表题后或表体起始处连续吞入表行，遇空行或非表行停止。"""
    rows: list[str] = []
    i = start
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            if rows:
                break
            i += 1
            continue
        if _looks_like_table_row(lines[i]):
            rows.append(lines[i])
            i += 1
            continue
        break
    return rows, i


def _split_row_cells(row: str) -> list[str]:
    r = row.strip()
    if "|" in r:
        return [c.strip() for c in r.strip("|").split("|")]
    return [c.strip() for c in re.split(r"\s{2,}|\t+", r) if c.strip()]


def _rows_to_markdown(rows: list[str], caption: str = "") -> str:
    cleaned: list[list[str]] = []
    for r in rows:
        cells = _split_row_cells(r)
        if cells:
            cleaned.append(cells)
    if not cleaned:
        return caption.strip()

    width = max(len(r) for r in cleaned)
    for r in cleaned:
        while len(r) < width:
            r.append("")

    header = cleaned[0]
    body = cleaned[1:] if len(cleaned) > 1 else []
    if header and all(re.fullmatch(r"[\d./%\-+eE]+", c or "x") for c in header):
        body = [header] + body
        header = [f"列{i+1}" for i in range(width)]

    lines = []
    if caption:
        lines.append(caption.strip())
        lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for r in body:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def extract_table_blocks(text: str) -> tuple[list[str], str]:
    """
    从文本中抽出表格块（转为 Markdown），返回 (table_mds, 剩余正文)。
    """
    lines = text.splitlines()
    tables: list[str] = []
    remain: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if _TABLE_CAPTION.match(line.strip()):
            caption = line.strip()
            rows, i = _consume_table_rows(lines, i + 1)
            if rows:
                tables.append(_rows_to_markdown(rows, caption=caption))
            else:
                remain.append(caption)
            continue

        if _looks_like_table_row(line):
            rows, i = _consume_table_rows(lines, i)
            if len(rows) >= 2:
                tables.append(_rows_to_markdown(rows))
            else:
                remain.extend(rows)
            continue

        remain.append(line)
        i += 1

    return tables, "\n".join(remain).strip()


def _chunk_body(
    text: str,
    section: str,
    weight: float,
    *,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
) -> list[StructuredChunk]:
    if not text.strip():
        return []
    tables, rest = extract_table_blocks(text)
    out: list[StructuredChunk] = []
    for md in tables:
        if len(md.strip()) >= 10:
            out.append(StructuredChunk(text=md.strip(), section="table", weight=weight))
    if rest.strip():
        pieces = hybrid_chunk_text(rest, embed_fn=embed_fn)
        for p in pieces:
            p = p.strip()
            if len(p) >= 10:
                out.append(StructuredChunk(text=p, section=section, weight=weight))
    return out


def structure_document(
    text: str,
    *,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
) -> list[StructuredChunk]:
    """
    论文/长文档 → 带 section/weight 的结构化块列表。
    """
    if not bool(getattr(settings, "INGEST_STRUCTURED_CHUNK", True)):
        return [
            StructuredChunk(text=c, section="body", weight=1.0)
            for c in hybrid_chunk_text(text, embed_fn=embed_fn)
            if len(c.strip()) >= 10
        ]

    raw = text or ""
    if bool(getattr(settings, "INGEST_FILTER_GARBAGE", True)):
        raw = filter_garbage_lines(raw)

    drop_refs = bool(getattr(settings, "INGEST_DROP_REFERENCES", True))
    ref_weight = float(getattr(settings, "INGEST_REFERENCE_WEIGHT", 0.35) or 0.35)

    parts = _split_by_headings(raw)
    chunks: list[StructuredChunk] = []
    skipped_ref_chars = 0

    for sec, body in parts:
        if sec == "reference":
            if drop_refs:
                skipped_ref_chars += len(body)
                continue
            # 参考文献整段或粗切，低权重
            if len(body) <= int(getattr(settings, "CHUNK_SIZE", 500) or 500) * 2:
                chunks.append(StructuredChunk(text=body, section=sec, weight=ref_weight))
            else:
                for p in hybrid_chunk_text(body, embed_fn=embed_fn):
                    if len(p.strip()) >= 10:
                        chunks.append(
                            StructuredChunk(text=p.strip(), section=sec, weight=ref_weight)
                        )
            continue

        if sec == "abstract":
            # 摘要整块；过长才二次切，但仍标记 abstract
            max_abs = int(getattr(settings, "INGEST_ABSTRACT_MAX_CHARS", 3000) or 3000)
            body_stripped = body.strip()
            if len(body_stripped) <= max_abs:
                if len(body_stripped) >= 10:
                    chunks.append(
                        StructuredChunk(text=body_stripped, section="abstract", weight=1.0)
                    )
            else:
                for p in hybrid_chunk_text(body_stripped, embed_fn=embed_fn):
                    if len(p.strip()) >= 10:
                        chunks.append(
                            StructuredChunk(text=p.strip(), section="abstract", weight=1.0)
                        )
            continue

        if sec == "keywords":
            if len(body.strip()) >= 10:
                chunks.append(
                    StructuredChunk(text=body.strip(), section="keywords", weight=0.9)
                )
            continue

        chunks.extend(_chunk_body(body, sec, 1.0, embed_fn=embed_fn))

    # 无结构时仍做表抽取 + 语义切分
    if not chunks and raw.strip():
        chunks.extend(_chunk_body(raw, "body", 1.0, embed_fn=embed_fn))

    log.info(
        f"结构化切块: parts={len(parts)} chunks={len(chunks)} "
        f"drop_refs={drop_refs} skipped_ref_chars={skipped_ref_chars}"
    )
    return chunks
