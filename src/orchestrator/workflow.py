"""
src/orchestrator/workflow.py — 确定性 Workflow 层

负责「不需要模型做策略决策」的硬逻辑：
  长期记忆召回 → Query 改进 → 多模态理解 → 意图路由
策略决策留给统一 Policy Agent / Chat / Deep Research。
"""
from __future__ import annotations

from src.agents.router_agent import RouterAgent
from src.llm.provider import llm
from src.memory.long_term_memory import ltm
from src.memory.working_memory import WorkingMemory
from src.multimodal.attachments import has_images
from src.orchestrator.state import AgentState
from src.rag.query_rewrite import apply_query_rewrite, _core_question
from src.utils.logger import get_logger

log = get_logger("workflow")

_AGENT_LABEL = {
    "chat": "日常对话",
    "super": "统一策略 Agent",
    "deep_research": "深度研究",
}


def run_workflow(state: AgentState) -> dict:
    """单节点完成确定性预处理 + 路由，返回状态增量。"""
    wm = WorkingMemory.from_snapshot(state.get("wm_items"))
    question = state.get("question") or ""
    attachments = state.get("attachments") or []
    trace: list[dict] = []

    # 1) 长期记忆召回（确定性向量检索）
    hits = ltm.recall(question, k=2)
    memory_hint = "\n".join(f"- {h['text']}" for h in hits) if hits else ""
    trace.append({"step": "workflow_recall", "detail": f"召回 {len(hits)} 条长期记忆"})

    # 2) Query 改进（混合：可跳过）
    new_q, meta = apply_query_rewrite(question, memory_hint=memory_hint)
    core = _core_question(question)
    rewritten_core = _core_question(new_q)
    if not meta.get("skipped") and rewritten_core != core:
        question = new_q
        trace.append({
            "step": "workflow_rewrite",
            "detail": f"「{core[:36]}」→「{rewritten_core[:36]}」",
            "meta": {"original": core, "rewritten": rewritten_core},
        })
    else:
        trace.append({"step": "workflow_rewrite", "detail": meta.get("changes", "保持原问")})

    # 3) 多模态（仅有图时）
    images = [a for a in attachments if a.get("kind") == "image"]
    if images and has_images(attachments):
        vision_text = llm.describe_images(question, images)
        question = f"{question}\n\n【图片视觉分析】\n{vision_text}" if question.strip() else vision_text
        names = "、".join(a.get("name", "图片") for a in images)
        trace.append({"step": "workflow_multimodal", "detail": f"已理解 {len(images)} 张图片（{names}）"})

    # 4) 意图路由（规则优先，必要时 LLM）
    decision = RouterAgent(wm).dispatch(question)
    agent = decision["agent"]
    if agent == "search":
        agent = "super"
    if agent not in ("chat", "super", "deep_research"):
        agent = "super"

    trace.append({
        "step": "workflow_router",
        "detail": f"→ {_AGENT_LABEL.get(agent, agent)}（{decision['reason']}）",
    })
    log.info(f"Workflow 完成路由: {agent}")

    return {
        "question": question,
        "memory_hint": memory_hint,
        "query_rewrite_note": meta.get("changes", "") if rewritten_core != core else "",
        "route": agent,
        "route_reason": decision["reason"],
        "wm_items": wm.export_snapshot(),
        "trace": trace,
    }
