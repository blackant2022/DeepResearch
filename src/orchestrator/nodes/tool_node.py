"""
src/orchestrator/nodes/tool_node.py — ReAct 工具执行节点
"""
from __future__ import annotations

import json
from typing import Any

from src.memory.working_memory import WorkingMemory
from src.middleware.pipeline import pipeline
from src.tools.base import registry


def _parse_args(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}


def execute_tool_calls(
    tool_calls: list[dict[str, Any]],
    wm: WorkingMemory,
) -> tuple[list[dict[str, Any]], list[dict]]:
    """执行一批 tool_calls，返回 (tool_messages, trace_items)。"""
    tool_messages: list[dict[str, Any]] = []
    trace: list[dict] = []

    for tc in tool_calls:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        args = _parse_args(fn.get("arguments", "{}"))
        tool = registry.get(name)

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
            wm.add("tool", json.dumps(payload, ensure_ascii=False)[:2000], kind="observation", tool=name)

        tool_messages.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": json.dumps(payload, ensure_ascii=False),
        })
        trace.append({"step": "tool", "detail": detail})

    return tool_messages, trace
