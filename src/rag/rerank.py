"""
src/rag/rerank.py — 检索结果重排序

默认使用智源 BAAI/bge-reranker-v2-m3（transformers 直载，避免 FlagEmbedding
与 transformers 5.x 不兼容）。对外置信度仍用 vector_score。
失败时回退稠密向量重打分。
"""
from __future__ import annotations

from typing import Any

from config.settings import settings
from src.utils.logger import get_logger

log = get_logger("rag.rerank")

_bge_bundle: dict[str, Any] | None = None
_bge_failed = False


def _get_bge_reranker():
    """懒加载 BGE Cross-Encoder（tokenizer + classification head）。"""
    global _bge_bundle, _bge_failed
    if _bge_failed:
        return None
    if _bge_bundle is not None:
        return _bge_bundle
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        name = settings.RERANK_MODEL
        use_fp16 = bool(getattr(settings, "RERANK_USE_FP16", False)) and torch.cuda.is_available()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"加载重排序模型 {name}（transformers，device={device}，fp16={use_fp16}）…")
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name)
        model.eval()
        model.to(device)
        if use_fp16 and device == "cuda":
            model.half()
        _bge_bundle = {
            "tokenizer": tokenizer,
            "model": model,
            "device": device,
            "torch": torch,
        }
        log.info("重排序模型就绪")
        return _bge_bundle
    except Exception as e:  # noqa: BLE001
        _bge_failed = True
        log.warning(f"BGE Reranker 不可用，将使用稠密重打分: {e}")
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


def _ensure_vector_score(h: dict[str, Any]) -> dict[str, Any]:
    item = dict(h)
    if item.get("vector_score") is None and item.get("score") is not None:
        try:
            item["vector_score"] = float(item["score"])
        except (TypeError, ValueError):
            item["vector_score"] = 0.0
    return item


def _dense_rerank(query: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from src.memory.embedding import encode_docs, encode_query

    q_vec = encode_query(query)
    docs = [str(h.get("content") or "") for h in hits]
    doc_vecs = encode_docs(docs) if docs else []
    scored = []
    for h, dv in zip(hits, doc_vecs):
        item = _ensure_vector_score(h)
        cos = round(_cosine(q_vec, dv), 4)
        item["rerank_score"] = cos
        item["score"] = cos
        item["rerank_backend"] = "dense"
        scored.append(item)
    scored.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
    return scored


def _sigmoid(x: float) -> float:
    import math
    # 防溢出
    if x >= 30:
        return 1.0
    if x <= -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _cross_rerank(query: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    bundle = _get_bge_reranker()
    if bundle is None:
        return None

    torch = bundle["torch"]
    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    device = bundle["device"]
    docs = [str(h.get("content") or "")[:1500] for h in hits]
    pairs = [[query, d] for d in docs]

    try:
        with torch.no_grad():
            inputs = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs, return_dict=True).logits.view(-1).float().cpu()
            raw_scores = [float(x) for x in logits.tolist()]
    except Exception as e:  # noqa: BLE001
        log.warning(f"BGE rerank 失败: {e}")
        return None

    scored = []
    for h, raw in zip(hits, raw_scores):
        item = _ensure_vector_score(h)
        # normalize 到 0~1，便于日志观察；排序仍按 raw/norm 皆可
        norm = _sigmoid(raw)
        item["rerank_raw"] = raw
        item["rerank_score"] = norm
        item["score"] = float(item.get("vector_score") or 0.0)
        item["rerank_backend"] = "bge_reranker"
        scored.append(item)
    scored.sort(key=lambda x: x.get("rerank_raw", 0), reverse=True)
    return scored


def diversify_by_filename(hits: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    """同一文件优先保留最高分 1 条，再补其余，避免单文档霸榜。"""
    if top_n <= 0 or not hits:
        return []
    primary: list[dict[str, Any]] = []
    seen: set[str] = set()
    rest: list[dict[str, Any]] = []
    for h in hits:
        fn = str(h.get("filename") or h.get("source") or "?")
        if fn not in seen:
            seen.add(fn)
            primary.append(h)
        else:
            rest.append(h)
        if len(primary) >= top_n:
            break
    out = primary[:top_n]
    if len(out) < top_n:
        for h in rest:
            fn = str(h.get("filename") or h.get("source") or "?")
            if fn in {str(x.get("filename") or x.get("source") or "?") for x in out}:
                continue
            out.append(h)
            if len(out) >= top_n:
                break
    return out


def rerank_hits(
    query: str,
    hits: list[dict[str, Any]],
    *,
    top_n: int | None = None,
    min_score: float | None = None,
) -> list[dict[str, Any]]:
    """
    对初筛结果二次排序并过滤。
    过滤阈值作用在 vector_score（可信相关度）上。
    """
    if not hits:
        return []

    n = top_n if top_n is not None else settings.TOP_K
    vec_thr = settings.RETRIEVAL_THRESHOLD if min_score is None else min_score

    if not settings.RERANK_ENABLED:
        out = sorted((_ensure_vector_score(h) for h in hits), key=lambda x: x.get("score", 0), reverse=True)
        filtered = [h for h in out if float(h.get("vector_score") or h.get("score") or 0) >= vec_thr]
        ranked = filtered or out[:1]
        return diversify_by_filename(ranked, n)

    ranked = None
    backend = (settings.RERANK_BACKEND or "auto").lower()
    if backend == "dense":
        ranked = _dense_rerank(query, hits)
    elif backend in ("auto", "cross_encoder", "bge"):
        ranked = _cross_rerank(query, hits)
    if ranked is None:
        ranked = _dense_rerank(query, hits)

    filtered = [
        h for h in ranked
        if float(h.get("vector_score") or h.get("score") or 0) >= vec_thr
    ]
    if not filtered:
        return diversify_by_filename(ranked[:1], n)
    return diversify_by_filename(filtered, n)
