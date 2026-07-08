"""
src/evaluation/run_metrics.py — 单次运行的量化质量指标（含双层 RAG 评估）
"""
from __future__ import annotations

from config.settings import settings
from src.evaluation.rag_eval import evaluate_rag_dual_layer


def build_run_metrics(result: dict) -> dict:
    """从 run_agent 返回值提取可展示的量化指标。"""
    grounding = result.get("grounding") or {}
    trace = result.get("trace") or []
    route = result.get("route", "")

    tool_calls = sum(1 for t in trace if t.get("step") == "tool")
    tool_ok = sum(1 for t in trace if t.get("step") == "tool" and "成功" in t.get("detail", ""))
    rag_eval = evaluate_rag_dual_layer(result)
    layer1 = rag_eval["layer1"]
    layer2 = rag_eval["layer2"]

    metrics: dict = {
        "route": route,
        "tool_calls": tool_calls,
        "tool_success_rate": round(tool_ok / tool_calls, 2) if tool_calls else None,
        "react_iterations": result.get("react_iterations", 0),
        "retrieval_avg_score": layer1.get("avg_score"),
        "retrieval_hits": layer1.get("hits", 0),
        "rag_eval": rag_eval,
    }

    if layer2.get("executed"):
        rate = float(layer2["support_rate"])
        metrics["primary_name"] = "事实支撑率"
        metrics["primary_value"] = rate
        metrics["primary_display"] = f"{rate:.0%}"
        metrics["primary_pass"] = layer2["pass"]
        metrics["primary_hint"] = layer2["hint"]
        metrics["secondary_name"] = "检索相关度"
        metrics["secondary_display"] = (
            f"{layer1['avg_score']:.2f}" if layer1.get("avg_score") is not None else "—"
        )
        metrics["secondary_pass"] = layer1.get("pass")
    elif layer1.get("executed"):
        avg = layer1["avg_score"]
        metrics["primary_name"] = "检索相关度"
        metrics["primary_value"] = avg
        metrics["primary_display"] = f"{avg:.2f}"
        metrics["primary_pass"] = layer1["pass"]
        metrics["primary_hint"] = layer1["hint"]
    elif route == "chat":
        metrics["primary_name"] = "响应模式"
        metrics["primary_value"] = 1.0
        metrics["primary_display"] = "日常对话"
        metrics["primary_pass"] = True
        metrics["primary_hint"] = "未触发 RAG 评估"
    else:
        metrics["primary_name"] = "执行轮次"
        metrics["primary_value"] = float(metrics["react_iterations"] or 0)
        metrics["primary_display"] = str(metrics["react_iterations"])
        metrics["primary_pass"] = True
        metrics["primary_hint"] = f"工具调用 {tool_calls} 次"

    metrics["rag_dual_pass"] = rag_eval.get("overall_pass")
    return metrics
