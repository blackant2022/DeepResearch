"""
src/orchestrator/nodes/tool_node.py — 工具执行节点（支持同轮并行）
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from config.settings import settings
from src.memory.working_memory import WorkingMemory
from src.middleware.pipeline import pipeline
from src.tools.base import registry


def _parse_args(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}


def _run_one_tool(tc: dict[str, Any], index: int) -> tuple[int, dict[str, Any], dict, str | None]:
    """执行单个工具，返回 (序号, tool_message, trace_item, wm_snippet)。"""
    fn = tc.get("function", {})
    name = fn.get("name", "")
    args = _parse_args(fn.get("arguments", "{}"))
    tool = registry.get(name)
    result = None

    if tool is None:
        payload = {"ok": False, "error": f"未知工具: {name}"}
        detail = f"未知工具 {name}"
    else:
        result = pipeline.invoke(tool, **args)
        if result.ok:
            payload = {"ok": True, "output": result.output}
            detail = f"{name} 成功 ({result.latency_ms}ms)"
        else:
            payload = {"ok": False, "error": result.error, "error_type": result.error_type}
            detail = f"{name} 失败: {result.error}"

    tool_message = {
        "role": "tool",
        "tool_call_id": tc.get("id") or f"call_{index}",
        "content": json.dumps(payload, ensure_ascii=False),
    }
    trace_item: dict = {"step": "tool", "detail": detail}
    if name == "knowledge_search" and result is not None and result.ok:
        raw = result.output
        hits = []
        if isinstance(raw, dict):
            hits = raw.get("hits") or []
            conf = (raw.get("confidence") or {}).get("confidence")
            if conf is not None:
                trace_item["meta"] = {
                    **(trace_item.get("meta") or {}),
                    "answer_confidence": conf,
                    "low_confidence": bool(raw.get("low_confidence")),
                }
        elif isinstance(raw, list):
            hits = raw
        scores = [h.get("score", 0) for h in hits if isinstance(h, dict)]
        if scores:
            avg = sum(scores) / len(scores)
            trace_item["detail"] = f"{detail}，相关度均值 {avg:.2f}"
            meta = dict(trace_item.get("meta") or {})
            meta["retrieval_scores"] = scores
            trace_item["meta"] = meta

    wm_snippet = None
    if tool is not None:
        wm_snippet = json.dumps(payload, ensure_ascii=False)[:2000]
    return index, tool_message, trace_item, wm_snippet


def execute_tool_calls(
    tool_calls: list[dict[str, Any]],
    wm: WorkingMemory,
    *,
    parallel: bool | None = None,
) -> tuple[list[dict[str, Any]], list[dict]]:
    """
    执行一批 tool_calls。
    同轮多个工具默认并行（检索 + 联网可同时跑），结果按原始顺序回填。
    """
    if not tool_calls:
        return [], []

    use_parallel = settings.PARALLEL_TOOLS if parallel is None else parallel
    workers = min(len(tool_calls), max(1, settings.PARALLEL_TOOL_WORKERS))

    results: list[tuple[int, dict, dict, str | None]]
    if use_parallel and len(tool_calls) > 1:
        results = [None] * len(tool_calls)  # type: ignore[list-item]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_run_one_tool, tc, i): i
                for i, tc in enumerate(tool_calls)
            }
            for fut in as_completed(futures):
                idx, msg, tr, snippet = fut.result()
                results[idx] = (idx, msg, tr, snippet)
    else:
        results = [_run_one_tool(tc, i) for i, tc in enumerate(tool_calls)]

    tool_messages: list[dict[str, Any]] = []
    trace: list[dict] = []
    for _, msg, tr, snippet in results:
        tool_messages.append(msg)
        trace.append(tr)
        if snippet is not None:
            # 工作记忆串行写入，避免并发写冲突
            wm.add("tool", snippet, kind="observation", tool="batch")

    if use_parallel and len(tool_calls) > 1:
        for item in trace:
            item["detail"] = f"[并行] {item['detail']}"

    return tool_messages, trace
