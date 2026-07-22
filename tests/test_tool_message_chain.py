"""tests/test_tool_message_chain.py — tool_calls 消息链修复与预算压缩不破坏协议"""
from __future__ import annotations

from src.llm.messages import (
    repair_tool_message_chain,
    sanitize_for_plain_chat,
    sanitize_for_tool_chat,
)
from src.memory.context_manager import ContextManager


def _asst_tools(*ids: str) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tid,
                "type": "function",
                "function": {"name": "knowledge_search", "arguments": "{}"},
            }
            for tid in ids
        ],
    }


def _tool(tid: str, body: str = '{"ok":true}') -> dict:
    return {"role": "tool", "tool_call_id": tid, "content": body}


def test_repair_fills_missing_tool_and_keeps_existing():
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "q"},
        _asst_tools("a", "b"),
        _tool("a", '{"ok":true,"output":1}'),
        # b 缺失
        {"role": "user", "content": "继续"},
    ]
    fixed = repair_tool_message_chain(msgs)
    # assistant 后应有 a、b 两条 tool
    idx = next(i for i, m in enumerate(fixed) if m.get("tool_calls"))
    assert fixed[idx + 1]["tool_call_id"] == "a"
    assert fixed[idx + 2]["tool_call_id"] == "b"
    assert "未执行" in fixed[idx + 2]["content"] or "缺失" in fixed[idx + 2]["content"]


def test_repair_assigns_missing_ids():
    msgs = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"type": "function", "function": {"name": "calculator", "arguments": "{}"}},
            ],
        },
    ]
    fixed = sanitize_for_tool_chat(msgs)
    asst = next(m for m in fixed if m.get("role") == "assistant")
    tid = asst["tool_calls"][0]["id"]
    assert tid
    assert fixed[-1]["role"] == "tool"
    assert fixed[-1]["tool_call_id"] == tid


def test_plain_chat_strips_tool_protocol():
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "q"},
        _asst_tools("x"),
        _tool("x"),
    ]
    plain = sanitize_for_plain_chat(msgs)
    assert all(m.get("role") != "tool" for m in plain)
    assert all(not m.get("tool_calls") for m in plain)
    assert any("工具" in str(m.get("content", "")) for m in plain if m["role"] == "assistant")


def test_context_budget_does_not_orphan_tool_calls():
    cm = ContextManager(max_chars=200, tool_snippet=50)
    # 构造很长的两轮工具链
    long = "证据内容" * 80
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "问题一"},
        _asst_tools("c1"),
        _tool("c1", long),
        {"role": "user", "content": "问题二"},
        _asst_tools("c2", "c3"),
        _tool("c2", long),
        _tool("c3", long),
    ]
    out = cm.prepare_for_policy(msgs)
    # 任意 assistant.tool_calls 后必须有完整 tool 回执
    i = 0
    while i < len(out):
        m = out[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            expected = {tc["id"] for tc in m["tool_calls"]}
            j = i + 1
            found = set()
            while j < len(out) and out[j].get("role") == "tool":
                found.add(out[j].get("tool_call_id"))
                j += 1
            assert expected <= found, f"orphan tool_calls: need {expected}, got {found}"
            i = j
        else:
            i += 1
