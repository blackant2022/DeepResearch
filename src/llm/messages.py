"""
src/llm/messages.py — OpenAI / DeepSeek 兼容消息链修复

确保每条带 tool_calls 的 assistant 消息后都有对应 tool_call_id 的 tool 消息，
避免 API 400: insufficient tool messages following tool_calls message。
"""
from __future__ import annotations

import json
from typing import Any


def _fake_tool(tid: str, error: str = "工具未执行或响应缺失") -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": tid,
        "content": json.dumps({"ok": False, "error": error}, ensure_ascii=False),
    }


def ensure_tool_call_ids(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """给缺失 id 的 tool_calls 补上稳定 id，避免无法配对。"""
    out: list[dict[str, Any]] = []
    for i, raw in enumerate(messages):
        msg = dict(raw)
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            fixed = []
            for j, tc in enumerate(msg["tool_calls"]):
                item = dict(tc)
                if not item.get("id"):
                    item["id"] = f"call_auto_{i}_{j}"
                fixed.append(item)
            msg["tool_calls"] = fixed
        out.append(msg)
    return out


def repair_tool_message_chain(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    修补未配对的 tool_calls：
      - 保留已有 tool 回执
      - 为缺失的 tool_call_id 补失败回执
      - 丢弃无法归属的孤立 tool 消息
    """
    if not messages:
        return []

    messages = ensure_tool_call_ids(messages)
    out: list[dict[str, Any]] = []
    i = 0
    while i < len(messages):
        msg = dict(messages[i])
        tool_calls = msg.get("tool_calls") or []

        # 孤立 tool：前面没有对应 assistant.tool_calls，丢弃
        if msg.get("role") == "tool":
            i += 1
            continue

        out.append(msg)

        if msg.get("role") == "assistant" and tool_calls:
            expected_ids = [tc.get("id") for tc in tool_calls if tc.get("id")]
            expected = set(expected_ids)
            j = i + 1
            found: dict[str, dict[str, Any]] = {}
            while j < len(messages) and messages[j].get("role") == "tool":
                tmsg = dict(messages[j])
                tid = tmsg.get("tool_call_id")
                if tid and tid in expected and tid not in found:
                    found[tid] = tmsg
                j += 1

            # 按 tool_calls 原始顺序回填，缺的补假回执
            for tid in expected_ids:
                if tid in found:
                    out.append(found[tid])
                else:
                    out.append(_fake_tool(tid))
            i = j
            continue

        i += 1
    return out


def strip_tool_protocol(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    转为纯文本对话（用于无 tools 参数的 llm.chat 兜底）。
    assistant.tool_calls 改为文字说明，tool 结果并入 assistant 旁白。
    """
    repaired = repair_tool_message_chain(messages)
    cleaned: list[dict[str, Any]] = []
    i = 0
    while i < len(repaired):
        msg = repaired[i]
        role = msg.get("role")
        if role in ("system", "user"):
            cleaned.append({"role": role, "content": str(msg.get("content") or "")})
            i += 1
            continue

        if role == "assistant":
            content = str(msg.get("content") or "").strip()
            tool_calls = msg.get("tool_calls") or []
            parts = [content] if content else []
            if tool_calls:
                names = [
                    (tc.get("function") or {}).get("name", "?") for tc in tool_calls
                ]
                parts.append(f"（已请求工具：{', '.join(names)}）")
            # 吞掉紧随的 tool 结果，压缩进文本
            j = i + 1
            tool_notes: list[str] = []
            while j < len(repaired) and repaired[j].get("role") == "tool":
                raw = repaired[j].get("content") or ""
                tool_notes.append(str(raw)[:800])
                j += 1
            if tool_notes:
                parts.append("【工具结果】\n" + "\n---\n".join(tool_notes))
            cleaned.append({"role": "assistant", "content": "\n".join(parts).strip() or "（已调用工具）"})
            i = j
            continue

        # 其他角色跳过
        i += 1
    return cleaned


def sanitize_for_tool_chat(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """发给 chat_with_tools 前：补全 tool 链。"""
    return repair_tool_message_chain(messages)


def sanitize_for_plain_chat(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """发给普通 chat 前：去掉 tool 协议，避免 400。"""
    return strip_tool_protocol(messages)
