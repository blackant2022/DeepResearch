"""
src/rag/rerank.py — 检索结果重排序

对向量初筛的候选集做二次排序，过滤低相关片段。
优先 Cross-Encoder（FastEmbed）；失败则回退为查询-文档稠密余弦重打分。
"""
from __future__ import annotations

from typing import Any

from config.settings import settings
from src.utils.logger import get_logger

log = get_logger("rag.rerank")

_cross_encoder = None
_cross_failed = False


def _get_cross_encoder():
    global _cross_encoder, _cross_failed
    if _cross_failed:
        return None
    if _cross_encoder is not None:
        return _cross_encoder
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        name = settings.RERANK_MODEL
        log.info(f"加载重排序模型 {name} …")
        _cross_encoder = TextCrossEncoder(model_name=name)
        log.info("重排序模型就绪")
        return _cross_encoder
    except Exception as e:  # noqa: BLE001
        _cross_failed = True
        log.warning(f"Cross-Encoder 不可用，将使用稠密重打分: {e}")
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(dot / (na * nb))


def _dense_rerank(query: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from src.memory.embedding import encode_docs, encode_query

    q_vec = encode_query(query)
    docs = [str(h.get("content") or "") for h in hits]
    doc_vecs = encode_docs(docs) if docs else []
    scored = []
    for h, dv in zip(hits, doc_vecs):
        item = dict(h)
        item["score"] = round(_cosine(q_vec, dv), 4)
        item["rerank_backend"] = "dense"
        scored.append(item)
    scored.sort(key=lambda x: x.get("score", 0), reverse=True)
    return scored


def _cross_rerank(query: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    model = _get_cross_encoder()
    if model is None:
        return None
    docs = [str(h.get("content") or "") for h in hits]
    try:
        # fastembed: rerank 返回与 documents 同序的分数
        raw_scores = list(model.rerank(query, docs))
    except Exception as e:  # noqa: BLE001
        log.warning(f"Cross-Encoder rerank 失败: {e}")
        return None

    scored = []
    for h, s in zip(hits, raw_scores):
        item = dict(h)
        # 不同模型分数尺度不一：用 sigmoid 压到 (0,1) 便于门禁
        try:
            import math

            x = float(s)
            prob = 1.0 / (1.0 + math.exp(-x)) if abs(x) < 50 else (1.0 if x > 0 else 0.0)
        except (TypeError, ValueError):
            prob = 0.0
        item["score"] = round(prob, 4)
        item["rerank_raw"] = float(s) if isinstance(s, (int, float)) else s
        item["rerank_backend"] = "cross_encoder"
        scored.append(item)
    scored.sort(key=lambda x: x.get("score", 0), reverse=True)
    return scored


def rerank_hits(
    query: str,
    hits: list[dict[str, Any]],
    *,
    top_n: int | None = None,
    min_score: float | None = None,
) -> list[dict[str, Any]]:
    """
    对初筛结果二次排序并过滤低相关。
    top_n: 最终返回条数；min_score: 低于则丢弃。
    """
    if not hits:
        return []
    if not settings.RERANK_ENABLED:
        out = sorted(hits, key=lambda x: x.get("score", 0), reverse=True)
        n = top_n or len(out)
        thr = settings.RETRIEVAL_THRESHOLD if min_score is None else min_score
        return [h for h in out[:n] if float(h.get("score") or 0) >= thr] or out[: min(1, n)]

    ranked = None
    backend = (settings.RERANK_BACKEND or "auto").lower()
    if backend == "dense":
        ranked = _dense_rerank(query, hits)
    elif backend in ("auto", "cross_encoder"):
        ranked = _cross_rerank(query, hits)
    if ranked is None:
        ranked = _dense_rerank(query, hits)

    thr = settings.RERANK_MIN_SCORE if min_score is None else min_score
    filtered = [h for h in ranked if float(h.get("score") or 0) >= thr]
    n = top_n if top_n is not None else settings.TOP_K
    # 全部被滤掉时保留最高 1 条供置信度模块判兜底（避免 silent empty 无分数）
    if not filtered and ranked:
        return ranked[:1]
    return filtered[:n]
