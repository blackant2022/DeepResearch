"""
src/llm/messages.py — OpenAI / DeepSeek 兼容消息链修复

确保每条带 tool_calls 的 assistant 消息后都有对应 tool_call_id 的 tool 消息。
"""
from __future__ import annotations

from typing import Any


def repair_tool_message_chain(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """修补未配对的 tool_calls，避免 API 400 invalid_request_error。"""
    if not messages:
        return []

    out: list[dict[str, Any]] = []
    i = 0
    while i < len(messages):
        msg = dict(messages[i])
        out.append(msg)

        tool_calls = msg.get("tool_calls") or []
        if msg.get("role") == "assistant" and tool_calls:
            expected = {tc.get("id") for tc in tool_calls if tc.get("id")}
            j = i + 1
            found: set[str] = set()
            tool_batch: list[dict[str, Any]] = []
            while j < len(messages) and messages[j].get("role") == "tool":
                tool_batch.append(messages[j])
                tid = messages[j].get("tool_call_id")
                if tid:
                    found.add(tid)
                j += 1

            missing = expected - found
            if missing:
                import json
                for tc in tool_calls:
                    tid = tc.get("id")
                    if tid and tid in missing:
                        out.append({
                            "role": "tool",
                            "tool_call_id": tid,
                            "content": json.dumps(
                                {"ok": False, "error": "工具未执行或响应缺失"},
                                ensure_ascii=False,
                            ),
                        })
            else:
                out.extend(tool_batch)
            i = j
            continue

        i += 1
    return out


def strip_tool_calls_from_assistant(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """兜底：移除仍未配对的 tool_calls 字段（仅保留文本 content）。"""
    repaired = repair_tool_message_chain(messages)
    cleaned: list[dict[str, Any]] = []
    for msg in repaired:
        m = dict(msg)
        if m.get("role") == "assistant" and m.get("tool_calls"):
            # 若修复后仍紧跟非 tool 消息，则去掉 tool_calls
            cleaned.append(m)
        else:
            cleaned.append(m)
    return cleaned
