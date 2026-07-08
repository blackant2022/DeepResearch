"""
src/evaluation/rag_eval.py — 双层 RAG 评估机制

第一层（检索层）：评估召回证据与问题的语义相关度（向量 score 均值、命中数）
第二层（生成层）：评估生成答案的事实支撑率（Grounding 原子论断核验）

两层均设阈值门禁；深度研究链路双层全开，超级智能体路径至少执行检索层评估。
"""
from __future__ import annotations

from config.settings import settings


def _collect_retrieval_scores(trace: list[dict]) -> list[float]:
    scores: list[float] = []
    for step in trace:
        if step.get("step") != "tool":
            continue
        meta = step.get("meta") or {}
        for s in meta.get("retrieval_scores") or []:
            try:
                scores.append(float(s))
            except (TypeError, ValueError):
                continue
    return scores


def evaluate_retrieval_layer(trace: list[dict]) -> dict:
    """第一层：检索质量评估。"""
    scores = _collect_retrieval_scores(trace)
    threshold = settings.RETRIEVAL_THRESHOLD
    if not scores:
        return {
            "layer": "retrieval",
            "name": "检索层",
            "executed": False,
            "avg_score": None,
            "hits": 0,
            "threshold": threshold,
            "pass": None,
            "hint": "本轮未触发知识库检索",
        }
    avg = round(sum(scores) / len(scores), 3)
    return {
        "layer": "retrieval",
        "name": "检索层",
        "executed": True,
        "avg_score": avg,
        "hits": len(scores),
        "threshold": threshold,
        "pass": avg >= threshold,
        "hint": f"命中 {len(scores)} 条，均值 {avg:.2f}（阈值 {threshold:.2f}）",
    }


def evaluate_generation_layer(grounding: dict | None) -> dict:
    """第二层：生成事实支撑评估（Grounding）。"""
    threshold = settings.GROUNDING_THRESHOLD
    g = grounding or {}
    rate = g.get("support_rate")
    if rate is None:
        return {
            "layer": "generation",
            "name": "生成层",
            "executed": False,
            "support_rate": None,
            "claims_total": 0,
            "threshold": threshold,
            "pass": None,
            "hint": "本轮未触发事实核验",
        }
    rate = float(rate)
    total = int(g.get("claims_total") or 0)
    return {
        "layer": "generation",
        "name": "生成层",
        "executed": True,
        "support_rate": rate,
        "claims_total": total,
        "threshold": threshold,
        "pass": rate >= threshold,
        "hint": f"论断 {total} 条，支撑率 {rate:.0%}（阈值 {threshold:.0%}）",
    }


def evaluate_rag_dual_layer(result: dict) -> dict:
    """汇总双层 RAG 评估结果。"""
    trace = result.get("trace") or []
    layer1 = evaluate_retrieval_layer(trace)
    layer2 = evaluate_generation_layer(result.get("grounding"))

    executed = [layer1["executed"], layer2["executed"]]
    if not any(executed):
        overall_pass = None
        mode = "none"
    elif layer1["executed"] and layer2["executed"]:
        overall_pass = bool(layer1["pass"]) and bool(layer2["pass"])
        mode = "dual"
    elif layer1["executed"]:
        overall_pass = bool(layer1["pass"])
        mode = "retrieval_only"
    else:
        overall_pass = bool(layer2["pass"])
        mode = "generation_only"

    return {
        "mode": mode,
        "layer1": layer1,
        "layer2": layer2,
        "overall_pass": overall_pass,
    }
