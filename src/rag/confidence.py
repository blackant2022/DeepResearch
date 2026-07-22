"""
src/rag/confidence.py — 问答置信度校验与兜底

优先使用向量相似度（vector_score），避免 Cross-Encoder sigmoid 分数虚高
导致域外问题也显示 0.99、门禁失效。
"""
from __future__ import annotations

import re
from typing import Any

from config.settings import settings

FALLBACK_MESSAGE = (
    "【置信度不足·已启用兜底】当前问题与本地知识库的匹配度偏低，"
    "为避免错误输出，暂不给出可能不准确的结论。\n\n"
    "建议：\n"
    "1. 换用更具体的专业表述（方法名、指标、作物/传感器等）再问；\n"
    "2. 确认相关 PDF 已入库，或补充文献后重新检索；\n"
    "3. 若需最新公开信息，可明确要求「联网搜索」。"
)

# 明显非文献域：即使向量偶发偏高，也强制低置信（知识库路径）
_OOD = re.compile(
    r"(天气|气温|下雨|写一首|作诗|写诗|歌词|笑话|量子计算|炒股|比特币|"
    r"足球比分|电影推荐|今天吃什么|外卖)"
)


def _hit_score(h: dict[str, Any]) -> float:
    """置信度打分：优先 vector_score，其次 score。"""
    for key in ("vector_score", "score"):
        try:
            if h.get(key) is not None:
                return float(h[key])
        except (TypeError, ValueError):
            continue
    return 0.0


def hits_confidence(hits: list[dict[str, Any]] | None) -> float:
    """取 Top 命中相关度：0.7×最高 + 0.3×均值。"""
    if not hits:
        return 0.0
    scores = [_hit_score(h) for h in hits]
    scores = [s for s in scores if s > 0 or s == 0.0]
    if not scores:
        return 0.0
    top = max(scores)
    avg = sum(scores) / len(scores)
    return round(0.7 * top + 0.3 * avg, 4)


def query_looks_ood(query: str) -> bool:
    q = (query or "").strip()
    if not q:
        return True
    if _OOD.search(q):
        return True
    try:
        from src.rag.domain_lexicon import match_terms
        from src.rag.kb_utils import is_kb_catalog_question, is_kb_knowledge_summary

        if is_kb_catalog_question(q) or is_kb_knowledge_summary(q):
            return False
        if match_terms(q):
            return False
        lit_keys = (
            "文献", "论文", "知识库", "文档", "资料", "研究", "遥感", "高光谱",
            "玉米", "氮", "模型", "方法", "总结", "概述", "反演", "估算",
            "实验", "精度", "算法", "深度学习", "检索", "光谱", "植被",
        )
        if any(k in q for k in lit_keys):
            return False
        if len(q) >= 6:
            return True
    except Exception:  # noqa: BLE001
        pass
    return False


def is_low_confidence(
    hits: list[dict[str, Any]] | None,
    threshold: float | None = None,
    *,
    query: str = "",
) -> bool:
    thr = settings.ANSWER_CONFIDENCE_THRESHOLD if threshold is None else threshold
    if query and query_looks_ood(query):
        # 域外：向量分也必须很高才过
        return hits_confidence(hits) < max(thr, 0.80)
    return hits_confidence(hits) < thr


def confidence_report(
    hits: list[dict[str, Any]] | None,
    *,
    query: str = "",
) -> dict[str, Any]:
    conf = hits_confidence(hits)
    thr = settings.ANSWER_CONFIDENCE_THRESHOLD
    ood = bool(query and query_looks_ood(query))
    effective_thr = max(thr, 0.80) if ood else thr
    low = conf < effective_thr
    if ood and conf < 0.85:
        # 域外且非极高向量分 → 一律兜底
        low = True
        effective_thr = max(effective_thr, round(conf + 0.01, 4))
    return {
        "confidence": conf,
        "threshold": effective_thr,
        "pass": not low,
        "hits": len(hits or []),
        "fallback": low,
        "ood_query": ood,
        "score_basis": "vector_score",
    }


def apply_answer_confidence_gate(
    answer: str,
    hits: list[dict[str, Any]] | None,
    *,
    used_knowledge: bool = True,
    query: str = "",
) -> tuple[str, dict[str, Any]]:
    report = confidence_report(hits, query=query)
    if not settings.ANSWER_CONFIDENCE_ENABLED:
        report["skipped"] = True
        return answer, report
    if not used_knowledge:
        report["skipped"] = True
        report["reason"] = "未使用知识库检索"
        return answer, report
    if report["fallback"]:
        return FALLBACK_MESSAGE, report
    return answer, report


def extract_knowledge_hits_from_messages(messages: list[dict[str, Any]] | None) -> list[dict]:
    """从 tool 消息中解析最近一次 knowledge_search 的 hits。"""
    import json

    if not messages:
        return []
    for msg in reversed(messages):
        if msg.get("role") != "tool":
            continue
        raw = msg.get("content") or ""
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(data, dict) or not data.get("ok"):
            continue
        output = data.get("output")
        if isinstance(output, dict) and "hits" in output:
            hits = output.get("hits") or []
            return [h for h in hits if isinstance(h, dict)]
        if isinstance(output, list):
            return [h for h in output if isinstance(h, dict) and ("content" in h or "score" in h)]
    return []
