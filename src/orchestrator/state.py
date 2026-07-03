"""
src/orchestrator/state.py — LangGraph 共享状态（超级智能体版）

messages + trace 使用 reducer 追加；wm_items 为可序列化工作记忆，兼容检查点。
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


def _last_wins(_left: int, right: int) -> int:
    return right


class AgentState(TypedDict, total=False):
    question: str
    route: str
    route_reason: str
    memory_hint: str
    # ReAct 对话历史（OpenAI messages 格式，含 tool_calls / tool 角色）
    messages: Annotated[list[dict[str, Any]], operator.add]
    react_iteration: Annotated[int, _last_wins]
    plan: list[dict]
    evaluation: dict
    replans: int
    findings: list[dict]
    draft: str
    grounding: dict
    revise_note: str
    revise_count: int
    final_answer: str
    trace: Annotated[list[dict], operator.add]
    # 工作记忆快照（msgpack 可序列化，替代不可序列化的 WorkingMemory 对象）
    wm_items: list[dict[str, Any]]
