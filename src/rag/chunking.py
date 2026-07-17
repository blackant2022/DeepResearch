"""
src/rag/chunking.py — 语义分块 + 动态滑动窗口

策略：
  1. 先按句/段切成原子单元
  2. 用相邻句向量余弦相似度找语义断点，合并为完整语义段落
  3. 超过 CHUNK_SIZE 的长段落再滑动窗口切分，重叠由阈值控制
  4. 过短块向后合并，避免碎片
"""
from __future__ import annotations

import re
from typing import Callable

from config.settings import settings
from src.utils.logger import get_logger

log = get_logger("rag.chunking")

# 中英文句末 + 段落边界
_SENT_SPLIT = re.compile(
    r"(?<=[。！？；!?;])\s*|(?<=\n\n)+|(?<=\.\s)|(?<=\!\s)|(?<=\?\s)"
)


def split_sentences(text: str) -> list[str]:
    """将正文切成句子级原子单元（保留有效内容）。"""
    text = (text or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p and p.strip()]
    # 过滤过短噪声（纯页码、孤立符号）
    return [p for p in parts if len(p) >= 2]


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return dot / (na * nb)


def _dynamic_overlap(chunk_len: int, base_overlap: int, overlap_ratio: float, max_size: int) -> int:
    """
    动态重叠：基础 overlap 与「块长 × 比例」取较大值，且不超过 max_size 的一半，
    保证长段落边界语义不丢。
    """
    by_ratio = int(chunk_len * overlap_ratio)
    ov = max(base_overlap, by_ratio)
    return max(0, min(ov, max_size // 2, chunk_len - 1 if chunk_len > 1 else 0))


def sliding_window_chunks(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    overlap_ratio: float,
    min_chunk: int,
) -> list[str]:
    """对超长文本做滑动窗口分块（重叠阈值可随块长动态放大）。"""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text] if len(text) >= min_chunk else ([text] if text else [])

    overlap = _dynamic_overlap(chunk_size, chunk_overlap, overlap_ratio, chunk_size)
    step = max(1, chunk_size - overlap)
    out: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        piece = text[start:end].strip()
        if len(piece) >= min_chunk:
            out.append(piece)
        elif piece and out:
            # 末尾残片并入上一块
            out[-1] = (out[-1] + piece).strip()
        if end >= n:
            break
        start += step
    return out


def _group_by_similarity(
    sentences: list[str],
    embeddings: list[list[float]],
    *,
    threshold: float,
    max_size: int,
    min_size: int,
) -> list[str]:
    """相邻句相似度低于阈值则断段；同时受 max_size 约束避免无限合并。"""
    if not sentences:
        return []
    groups: list[str] = []
    buf = sentences[0]
    for i in range(1, len(sentences)):
        sim = _cosine(embeddings[i - 1], embeddings[i])
        next_s = sentences[i]
        sep = "" if buf.endswith(("\n", " ")) or next_s.startswith(("\n", " ")) else ""
        would = f"{buf}{sep}{next_s}"
        # 语义断裂，且当前缓冲已够最小长度 → 开新段
        if sim < threshold and len(buf) >= min_size:
            groups.append(buf.strip())
            buf = next_s
        # 再拼会超长 → 先落盘再开新段
        elif len(would) > max_size and len(buf) >= min_size:
            groups.append(buf.strip())
            buf = next_s
        else:
            buf = would
    if buf.strip():
        groups.append(buf.strip())
    return groups


def hybrid_chunk_text(
    text: str,
    *,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    similarity_threshold: float | None = None,
    overlap_ratio: float | None = None,
    min_chunk: int | None = None,
) -> list[str]:
    """
    语义分块 + 长段滑动窗口。

    embed_fn: 默认使用项目 FastEmbed；单测可注入假向量。
    """
    chunk_size = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
    chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP
    similarity_threshold = (
        similarity_threshold
        if similarity_threshold is not None
        else settings.SEMANTIC_SPLIT_THRESHOLD
    )
    overlap_ratio = (
        overlap_ratio if overlap_ratio is not None else settings.CHUNK_OVERLAP_RATIO
    )
    min_chunk = min_chunk if min_chunk is not None else settings.CHUNK_MIN_SIZE

    text = (text or "").strip()
    if not text:
        return []

    sentences = split_sentences(text)
    if not sentences:
        return sliding_window_chunks(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            overlap_ratio=overlap_ratio,
            min_chunk=min_chunk,
        )

    # 单句极短文档：直接滑窗 / 整段返回
    if len(sentences) == 1:
        return sliding_window_chunks(
            sentences[0],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            overlap_ratio=overlap_ratio,
            min_chunk=min_chunk,
        )

    if embed_fn is None:
        from src.memory.embedding import encode_docs

        embed_fn = encode_docs

    try:
        embeddings = embed_fn(sentences)
    except Exception as e:  # noqa: BLE001
        log.warning(f"语义嵌入失败，回退纯滑动窗口: {e}")
        return sliding_window_chunks(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            overlap_ratio=overlap_ratio,
            min_chunk=min_chunk,
        )

    if len(embeddings) != len(sentences):
        log.warning("嵌入数量与句子不一致，回退滑动窗口")
        return sliding_window_chunks(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            overlap_ratio=overlap_ratio,
            min_chunk=min_chunk,
        )

    semantic_paras = _group_by_similarity(
        sentences,
        embeddings,
        threshold=similarity_threshold,
        max_size=chunk_size,
        min_size=min_chunk,
    )

    chunks: list[str] = []
    for para in semantic_paras:
        if len(para) <= chunk_size:
            if len(para) >= min_chunk:
                chunks.append(para)
            elif para and chunks:
                merged = chunks[-1] + para
                if len(merged) <= chunk_size * 1.2:
                    chunks[-1] = merged
                else:
                    chunks.append(para)
            elif para:
                chunks.append(para)
        else:
            chunks.extend(
                sliding_window_chunks(
                    para,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    overlap_ratio=overlap_ratio,
                    min_chunk=min_chunk,
                )
            )

    # 二次合并过短尾块
    refined: list[str] = []
    for c in chunks:
        c = c.strip()
        if not c:
            continue
        if refined and len(c) < min_chunk and len(refined[-1]) + len(c) <= int(chunk_size * 1.2):
            refined[-1] = refined[-1] + c
        else:
            refined.append(c)

    log.info(
        f"混合分块完成: sentences={len(sentences)} semantic={len(semantic_paras)} "
        f"chunks={len(refined)} (size={chunk_size}, overlap={chunk_overlap}, "
        f"sim<{similarity_threshold})"
    )
    return refined
