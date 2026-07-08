"""
src/orchestrator/state.py — LangGraph 共享状态（超级智能体版）

messages + trace 使用 reducer 追加；wm_items 为可序列化工作记忆，兼容检查点。
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


def _last_wins(_left: int, right: int) -> int:
    return right


def _last_wins_any(left: Any, right: Any) -> Any:
    """同一步多来源更新时，以后写入为准（避免 INVALID_CONCURRENT_GRAPH_UPDATE）。"""
    return right


class AgentState(TypedDict, total=False):
    question: str
    original_question: str
    query_rewrite_note: str
    attachments: list[dict[str, Any]]   # 用户上传的图片/文档（可序列化）
    route: str
    route_reason: str
    memory_hint: str
    # ReAct 对话历史（OpenAI messages 格式，含 tool_calls / tool 角色）
    messages: Annotated[list[dict[str, Any]], operator.add]
    react_iteration: Annotated[int, _last_wins]
    plan: Annotated[list[dict], _last_wins_any]
    evaluation: Annotated[dict, _last_wins_any]
    replans: Annotated[int, _last_wins]
    findings: Annotated[list[dict], _last_wins_any]
    draft: Annotated[str, _last_wins_any]
    grounding: Annotated[dict, _last_wins_any]
    revise_note: Annotated[str, _last_wins_any]
    revise_count: Annotated[int, _last_wins]
    final_answer: Annotated[str, _last_wins_any]
    trace: Annotated[list[dict], operator.add]
    # 工作记忆快照（msgpack 可序列化，替代不可序列化的 WorkingMemory 对象）
    wm_items: Annotated[list[dict[str, Any]], _last_wins_any]
