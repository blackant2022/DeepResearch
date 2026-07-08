"""
src/orchestrator/graph.py — LangGraph 超级智能体编排

主流程：
  recall_memory（长期记忆召回）
    → query_rewrite（Query 改进）
    → multimodal（图片理解）
    → router（意图路由）
        ├─ chat           日常对话（快速路径）
        ├─ super_think    超级智能体 ReAct 环
        │     ↺ super_tools → super_think
        └─ deep_research  深度研究子图（Planner→Researcher→Writer→Critic）
"""
from __future__ import annotations

import uuid
from functools import lru_cache

from langgraph.graph import END, StateGraph
from langgraph.types import Overwrite

import src.tools  # noqa: F401

from config.settings import settings
from src.agents.chat_agent import ChatAgent
from src.agents.critic_agent import CriticAgent
from src.agents.planner_agent import PlannerAgent
from src.agents.researcher_agent import ResearcherAgent
from src.agents.router_agent import RouterAgent
from src.agents.super_agent import SuperAgent
from src.agents.writer_agent import WriterAgent
from src.memory.long_term_memory import ltm
from src.memory.working_memory import WorkingMemory
from src.orchestrator.nodes.tool_node import execute_tool_calls
from src.orchestrator.state import AgentState
from src.evaluation.run_metrics import build_run_metrics
from src.multimodal.attachments import has_images
from src.llm.provider import llm
from src.utils.logger import get_logger

log = get_logger("orchestrator")
MAX_REVISE = 1


def _build_turn_input(question: str, attachments: list[dict]) -> AgentState:
    """每轮用户提问的初始状态：覆盖检查点中的旧消息/轨迹，避免答非所问。"""
    return {
        "question": question,
        "original_question": question,
        "query_rewrite_note": "",
        "attachments": attachments,
        "trace": Overwrite([]),
        "messages": Overwrite([]),
        "wm_items": Overwrite([]),
        "react_iteration": 0,
        "wm_items": [],
        "revise_count": 0,
        "revise_note": "",
        "final_answer": "",
        "draft": "",
        "route": "",
        "route_reason": "",
        "plan": [],
        "findings": [],
        "grounding": {},
    }


def _hydrate_wm(state: AgentState) -> WorkingMemory:
    return WorkingMemory.from_snapshot(state.get("wm_items"))


def _wm_update(wm: WorkingMemory, updates: dict) -> dict:
    updates["wm_items"] = wm.export_snapshot()
    return updates

_AGENT_LABEL = {
    "chat": "日常对话",
    "super": "超级智能体",
    "deep_research": "深度研究",
}


# ===================== 公共节点 ===================== #
def node_query_rewrite(state: AgentState) -> dict:
    """检索前 Query 改进：口语化问题 → 检索友好表述。"""
    from src.rag.query_rewrite import apply_query_rewrite, _core_question

    full_q = state.get("question") or ""
    new_q, meta = apply_query_rewrite(full_q, memory_hint=state.get("memory_hint", ""))
    core = _core_question(full_q)
    rewritten_core = _core_question(new_q)

    if meta.get("skipped") or rewritten_core == core:
        return {
            "trace": [{
                "step": "query_rewrite",
                "detail": meta.get("changes", "保持原问"),
            }],
        }

    return {
        "question": new_q,
        "query_rewrite_note": meta.get("changes", ""),
        "trace": [{
            "step": "query_rewrite",
            "detail": f"「{core[:36]}」→「{rewritten_core[:36]}」",
            "meta": {"original": core, "rewritten": rewritten_core},
        }],
    }


def node_recall(state: AgentState) -> dict:
    hits = ltm.recall(state["question"], k=2)
    hint = "\n".join(f"- {h['text']}" for h in hits) if hits else ""
    return {
        "memory_hint": hint,
        "trace": [{"step": "recall_memory", "detail": f"召回 {len(hits)} 条长期记忆"}],
    }


def node_multimodal(state: AgentState) -> dict:
    """图片 → 视觉模型转文字，注入 question；文档已在 run_agent 阶段拼入。"""
    attachments = state.get("attachments") or []
    images = [a for a in attachments if a.get("kind") == "image"]
    if not images:
        return {}

    vision_text = llm.describe_images(state.get("question", ""), images)
    q = (state.get("question") or "").strip()
    enhanced = f"{q}\n\n【图片视觉分析】\n{vision_text}" if q else vision_text
    names = "、".join(a.get("name", "图片") for a in images)
    return {
        "question": enhanced,
        "trace": [{"step": "multimodal", "detail": f"已理解 {len(images)} 张图片（{names}）"}],
    }


def node_router(state: AgentState) -> dict:
    wm = _hydrate_wm(state)
    decision = RouterAgent(wm).dispatch(state["question"])
    agent = decision["agent"]
    return _wm_update(wm, {
        "route": agent,
        "route_reason": decision["reason"],
        "trace": [{
            "step": "router",
            "detail": f"→ {_AGENT_LABEL.get(agent, agent)}（{decision['reason']}）",
        }],
    })


def route_by_agent(state: AgentState) -> str:
    route = state.get("route", "super")
    if route == "search":
        return "super"  # 兼容旧路由名
    return route if route in ("chat", "super", "deep_research") else "super"


def node_chat(state: AgentState) -> dict:
    wm = _hydrate_wm(state)
    answer = ChatAgent(wm).reply(state["question"])
    return _wm_update(wm, {
        "final_answer": answer,
        "trace": [{"step": "chat_agent", "detail": "日常对话完成"}],
    })


# ===================== 超级智能体 ReAct 环 ===================== #
def node_super_think(state: AgentState) -> dict:
    wm = _hydrate_wm(state)
    agent = SuperAgent(wm)
    iteration = state.get("react_iteration", 0) + 1
    history = list(state.get("messages") or [])

    if not history:
        history = agent.build_initial_messages(state["question"], state.get("memory_hint", ""))

    result = agent.think_with_tools(history)
    assistant_msg = result["assistant_message"]
    tool_calls = result.get("tool_calls") or []

    new_msgs: list[dict]
    if iteration == 1 and not (state.get("messages") or []):
        new_msgs = list(history) + [assistant_msg]
    else:
        new_msgs = [assistant_msg]

    updates: dict = {
        "messages": new_msgs,
        "react_iteration": iteration,
        "trace": [{
            "step": "super_think",
            "detail": f"第 {iteration} 轮推理"
            + (f"，调用 {len(tool_calls)} 个工具" if tool_calls else "，生成最终回答"),
        }],
    }

    if not tool_calls:
        content = result.get("content") or assistant_msg.get("content") or ""
        updates["final_answer"] = content
        ltm.consolidate(f"问题：{state['question']}\n结论：{content[:400]}", state["question"])
        updates["trace"].append({"step": "super_done", "detail": "超级智能体作答完成"})

    return _wm_update(wm, updates)


def route_after_super_think(state: AgentState) -> str:
    messages = state.get("messages") or []
    if not messages:
        return "end"
    last = messages[-1]
    tool_calls = last.get("tool_calls") or []
    iteration = state.get("react_iteration", 0)
    if tool_calls and iteration < settings.MAX_REACT_ITERATIONS:
        return "tools"
    if tool_calls and iteration >= settings.MAX_REACT_ITERATIONS:
        log.warning("ReAct 达到最大轮次，强制结束")
    return "end"


def node_super_tools(state: AgentState) -> dict:
    wm = _hydrate_wm(state)
    messages = list(state.get("messages") or [])
    last = messages[-1]
    tool_calls = last.get("tool_calls") or []
    tool_msgs, trace_items = execute_tool_calls(tool_calls, wm)
    return _wm_update(wm, {"messages": tool_msgs, "trace": trace_items})


def node_super_finalize(state: AgentState) -> dict:
    """达到最大轮次仍有 tool_calls 时，用已有上下文兜底生成答案。"""
    if state.get("final_answer"):
        return {}
    from src.llm.messages import repair_tool_message_chain

    messages = repair_tool_message_chain(list(state.get("messages") or []))
    messages.append({
        "role": "user",
        "content": "请根据以上工具结果直接给出最终回答，不要再调用工具。",
    })
    answer = llm.chat(messages)
    ltm.consolidate(f"问题：{state['question']}\n结论：{answer[:400]}", state["question"])
    return {
        "final_answer": answer,
        "trace": [{"step": "super_finalize", "detail": "达到轮次上限，兜底生成"}],
    }


# ===================== 深度研究子图 ===================== #
def node_plan(state: AgentState) -> dict:
    wm = _hydrate_wm(state)
    result = PlannerAgent(wm).plan(state["question"], memory_hint=state.get("memory_hint", ""))
    return _wm_update(wm, {
        "plan": result["plan"],
        "replans": result["replans"],
        "trace": [{
            "step": "plan",
            "detail": f"{len(result['plan'])} 个子任务，重规划 {result['replans']} 次",
            "plan": result["plan"],
        }],
    })


def node_research(state: AgentState) -> dict:
    wm = _hydrate_wm(state)
    agent = ResearcherAgent(wm)
    seeded = agent.bootstrap_rag(state["question"])
    findings = [agent.execute(st, state["question"]) for st in state["plan"]]
    return _wm_update(wm, {
        "findings": findings,
        "trace": [{"step": "research", "detail": f"预置 {seeded} 条，完成 {len(findings)} 个子任务"}],
    })


def node_write(state: AgentState) -> dict:
    wm = _hydrate_wm(state)
    draft = WriterAgent(wm).compose(
        state["question"], state["findings"], state.get("revise_note", "")
    )
    return _wm_update(wm, {
        "draft": draft,
        "trace": [{"step": "write", "detail": f"生成答案（{len(draft)} 字）"}],
    })


def node_critic(state: AgentState) -> dict:
    wm = _hydrate_wm(state)
    report = CriticAgent(wm).review(state["draft"], question=state["question"])
    updates: dict = {
        "grounding": report,
        "revise_note": report.get("revise_note", ""),
        "trace": [{"step": "critic", "detail": f"事实核查 {report['support_rate']:.0%}"}],
    }
    if not report["grounded"]:
        updates["revise_count"] = state.get("revise_count", 0) + 1
    return _wm_update(wm, updates)


def node_consolidate(state: AgentState) -> dict:
    draft = state.get("draft", "")
    ltm.consolidate(f"问题：{state['question']}\n结论：{draft[:400]}", state["question"])
    return {
        "final_answer": draft,
        "trace": [{"step": "consolidate", "detail": "定稿并写入长期记忆"}],
    }


def route_after_critic(state: AgentState) -> str:
    grounding = state.get("grounding") or {}
    if grounding.get("grounded") or state.get("revise_count", 0) >= MAX_REVISE:
        return "pass"
    return "revise"


def _build_deep_graph():
    g = StateGraph(AgentState)
    g.add_node("plan", node_plan)
    g.add_node("research", node_research)
    g.add_node("write", node_write)
    g.add_node("critic", node_critic)
    g.add_node("consolidate", node_consolidate)
    g.set_entry_point("plan")
    g.add_edge("plan", "research")
    g.add_edge("research", "write")
    g.add_edge("write", "critic")
    g.add_conditional_edges("critic", route_after_critic, {"pass": "consolidate", "revise": "write"})
    g.add_edge("consolidate", END)
    return g.compile()


_DEEP_APP = _build_deep_graph()


def node_deep_research(state: AgentState) -> dict:
    sub = _DEEP_APP.invoke(state)
    out: dict = {"trace": sub.get("trace", [])}
    for key in (
        "final_answer", "draft", "grounding", "plan", "replans", "findings",
    ):
        if key in sub:
            out[key] = sub[key]
    # wm_items 由子图内部 reducer 合并，不重复写回主图
    if "wm_items" in sub:
        out["wm_items"] = sub["wm_items"]
    return out


# ===================== 主图 ===================== #
def build_dispatch_graph(*, checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("query_rewrite", node_query_rewrite)
    g.add_node("recall_memory", node_recall)
    g.add_node("multimodal", node_multimodal)
    g.add_node("router", node_router)
    g.add_node("chat", node_chat)
    g.add_node("super_think", node_super_think)
    g.add_node("super_tools", node_super_tools)
    g.add_node("super_finalize", node_super_finalize)
    g.add_node("deep_research", node_deep_research)

    g.set_entry_point("recall_memory")
    g.add_edge("recall_memory", "query_rewrite")
    g.add_edge("query_rewrite", "multimodal")
    g.add_edge("recall_memory", "multimodal")
    g.add_edge("multimodal", "router")
    g.add_conditional_edges(
        "router",
        route_by_agent,
        {"chat": "chat", "super": "super_think", "deep_research": "deep_research"},
    )
    g.add_edge("chat", END)
    g.add_conditional_edges(
        "super_think",
        route_after_super_think,
        {"tools": "super_tools", "end": "super_finalize"},
    )
    g.add_edge("super_tools", "super_think")
    g.add_edge("super_finalize", END)
    g.add_edge("deep_research", END)
    return g.compile(checkpointer=checkpointer)


@lru_cache(maxsize=1)
def _get_checkpointer():
    from langgraph.checkpoint.memory import MemorySaver
    return MemorySaver()


@lru_cache(maxsize=1)
def _get_app():
    try:
        from src.mcp.client import register_mcp_tools
        register_mcp_tools()
    except Exception as e:
        log.warning(f"MCP 工具注册跳过: {e}")
    return build_dispatch_graph(checkpointer=_get_checkpointer())


def run_agent(
    question: str,
    thread_id: str | None = None,
    attachments: list[dict] | None = None,
) -> dict:
    """运行超级智能体；attachments 为 parse_uploads 产出的可序列化附件列表。"""
    from src.multimodal.attachments import build_enhanced_question

    log.info(f"===== 开始：{question[:60]} =====")
    tid = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": tid}}

    att = attachments or []
    enhanced_q = build_enhanced_question(question, att)
    if has_images(att):
        log.info(f"多模态输入：{sum(1 for a in att if a.get('kind')=='image')} 图，"
                 f"{sum(1 for a in att if a.get('kind')=='document')} 文档")

    init = _build_turn_input(enhanced_q, att)
    final = _get_app().invoke(init, config=config)
    log.info("===== 完成 =====")
    payload = {
        "answer": final.get("final_answer", final.get("draft", "")),
        "trace": final.get("trace", []),
        "plan": final.get("plan", []),
        "grounding": final.get("grounding", {}),
        "replans": final.get("replans", 0),
        "route": final.get("route", ""),
        "route_reason": final.get("route_reason", ""),
        "thread_id": tid,
        "react_iterations": final.get("react_iteration", 0),
    }
    payload["metrics"] = build_run_metrics(payload)
    return payload
