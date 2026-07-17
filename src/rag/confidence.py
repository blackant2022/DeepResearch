"""
src/rag/confidence.py — 问答置信度校验与兜底

检索/重排后的相关度低于阈值时，不输出易误导的「编造答案」，
改为明确兜底提示，引导用户换问法或检查知识库。
"""
from __future__ import annotations

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


def hits_confidence(hits: list[dict[str, Any]] | None) -> float:
    """取 Top 命中相关度：优先最高分，其次均值。"""
    if not hits:
        return 0.0
    scores = []
    for h in hits:
        try:
            scores.append(float(h.get("score") or 0))
        except (TypeError, ValueError):
            continue
    if not scores:
        return 0.0
    # 最高分权重更大，兼顾均值稳定性
    top = max(scores)
    avg = sum(scores) / len(scores)
    return round(0.7 * top + 0.3 * avg, 4)


def is_low_confidence(hits: list[dict[str, Any]] | None, threshold: float | None = None) -> bool:
    thr = settings.ANSWER_CONFIDENCE_THRESHOLD if threshold is None else threshold
    return hits_confidence(hits) < thr


def confidence_report(hits: list[dict[str, Any]] | None) -> dict[str, Any]:
    conf = hits_confidence(hits)
    thr = settings.ANSWER_CONFIDENCE_THRESHOLD
    low = conf < thr
    return {
        "confidence": conf,
        "threshold": thr,
        "pass": not low,
        "hits": len(hits or []),
        "fallback": low,
    }


def apply_answer_confidence_gate(
    answer: str,
    hits: list[dict[str, Any]] | None,
    *,
    used_knowledge: bool = True,
) -> tuple[str, dict[str, Any]]:
    """
    若本轮依赖知识库且置信度不足 → 替换为兜底提示。
    返回 (最终文本, 报告)。
    """
    report = confidence_report(hits)
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
        # 新格式：{hits, confidence, ...}
        if isinstance(output, dict) and "hits" in output:
            hits = output.get("hits") or []
            return [h for h in hits if isinstance(h, dict)]
        if isinstance(output, list):
            return [h for h in output if isinstance(h, dict) and ("content" in h or "score" in h)]
    return []
