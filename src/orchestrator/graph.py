"""
src/orchestrator/graph.py — LangGraph 编排（Workflow + 统一策略 Agent）

架构原则（生产级精简）：
  - Workflow 层：确定性硬逻辑（记忆召回 / Query 改进 / 多模态 / 路由）
  - Agent 层：统一策略决策；ReAct 在节点内部循环，不再图级 think↔tools 空转
  - 深度研究：固定管线子图（Planner→Researcher→Writer→Critic）

主流程：
  workflow
    ├─ chat
    ├─ policy          （节点内多轮工具调用）
    └─ deep_research
"""
from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

from langgraph.graph import END, StateGraph
from langgraph.types import Overwrite

import src.tools  # noqa: F401

from config.settings import settings
from src.agents.chat_agent import ChatAgent
from src.agents.critic_agent import CriticAgent
from src.agents.planner_agent import PlannerAgent
from src.agents.policy_agent import PolicyAgent
from src.agents.researcher_agent import ResearcherAgent
from src.agents.writer_agent import WriterAgent
from src.evaluation.run_metrics import build_run_metrics
from src.memory.long_term_memory import ltm
from src.memory.working_memory import WorkingMemory
from src.multimodal.attachments import has_images
from src.orchestrator.state import AgentState
from src.orchestrator.workflow import run_workflow
from src.utils.logger import get_logger

log = get_logger("orchestrator")


def _max_revise() -> int:
    return max(0, int(getattr(settings, "MAX_REVISE", 1) or 0))


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


# ===================== Workflow ===================== #
def node_workflow(state: AgentState) -> dict:
    """确定性预处理 + 路由（单节点，消除并行边竞态）。"""
    return run_workflow(state)


def route_by_agent(state: AgentState) -> str:
    route = state.get("route", "super")
    if route == "search":
        return "policy"
    if route == "super":
        return "policy"
    return route if route in ("chat", "policy", "deep_research") else "policy"


# ===================== Chat / Policy ===================== #
def node_chat(state: AgentState) -> dict:
    wm = _hydrate_wm(state)
    answer = ChatAgent(wm).reply(state["question"])
    return _wm_update(wm, {
        "final_answer": answer,
        "trace": [{"step": "chat_agent", "detail": "日常对话完成"}],
    })


def node_policy(state: AgentState) -> dict:
    """统一策略 Agent：ReAct 在节点内完成，图上仅 1 次 hop。"""
    wm = _hydrate_wm(state)
    result = PolicyAgent(wm).run(
        question=state["question"],
        memory_hint=state.get("memory_hint", ""),
        max_iterations=settings.MAX_REACT_ITERATIONS,
    )
    return {
        "final_answer": result["final_answer"],
        "messages": Overwrite(result["messages"]),
        "react_iteration": result["react_iteration"],
        "wm_items": result["wm_items"],
        "trace": result["trace"],
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
    plan = list(state.get("plan") or [])
    question = state["question"]

    use_parallel = bool(getattr(settings, "PARALLEL_RESEARCH", True)) and len(plan) > 1
    workers = min(len(plan), max(1, int(getattr(settings, "PARALLEL_TOOL_WORKERS", 4) or 4)))

    if use_parallel:
        # 并行执行：不写 WM，结束后串行 persist，避免并发写冲突
        slots: list[dict | None] = [None] * len(plan)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {
                pool.submit(agent.execute, st, question, persist=False): i
                for i, st in enumerate(plan)
            }
            for fut in as_completed(futs):
                slots[futs[fut]] = fut.result()
        findings = [f for f in slots if f is not None]
        for f in findings:
            agent.persist_finding(f["subtask_id"], f["finding"], f["evidence"])
        detail = f"预置 {seeded} 条，并行完成 {len(findings)} 个子任务"
    else:
        findings = [agent.execute(st, question) for st in plan]
        detail = f"预置 {seeded} 条，完成 {len(findings)} 个子任务"

    return _wm_update(wm, {
        "findings": findings,
        "trace": [{"step": "research", "detail": detail}],
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
    payload = f"问题：{state['question']}\n结论：{draft[:400]}"
    question = state["question"]

    def _write_ltm() -> None:
        try:
            ltm.consolidate(payload, question)
        except Exception as e:  # noqa: BLE001
            log.warning(f"LTM 异步写入失败: {e}")

    if getattr(settings, "LTM_ASYNC_CONSOLIDATE", True):
        threading.Thread(target=_write_ltm, daemon=True, name="ltm-consolidate").start()
        detail = "定稿；LTM 后台写入"
    else:
        ltm.consolidate(payload, question)
        detail = "定稿并写入长期记忆"

    return {
        "final_answer": draft,
        "trace": [{"step": "consolidate", "detail": detail}],
    }


def node_safe_fallback(state: AgentState) -> dict:
    """达到修订上限仍未通过时，按支撑率做有风险提示或证据级拒答。"""
    report = dict(state.get("grounding") or {})
    rate = float(report.get("support_rate") or 0.0)
    min_support = float(
        getattr(settings, "GROUNDING_FALLBACK_MIN_SUPPORT", 0.4) or 0.4
    )
    unsupported = [str(x) for x in report.get("unsupported") or []]
    supported = [str(x) for x in report.get("supported") or []]

    if rate >= min_support:
        tier = "partial"
        warning = (
            f"【部分证据不足】本回答事实支撑率为 {rate:.0%}，"
            f"未达到 {settings.GROUNDING_THRESHOLD:.0%} 的通过阈值。"
            "以下内容是达到修订上限后的最新版本，请谨慎参考。"
        )
        if unsupported:
            warning += "\n\n未充分支撑的论断：\n" + "\n".join(
                f"- {claim}" for claim in unsupported[:3]
            )
        answer = f"{warning}\n\n---\n\n{state.get('draft', '')}"
        detail = f"修订达上限；部分支撑兜底（{rate:.0%}）"
    else:
        tier = "insufficient"
        lines = [
            "【证据不足·已停止扩写】",
            f"经过 {state.get('revise_count', 0)} 次修订后，事实支撑率仍为 {rate:.0%}，"
            "低于安全输出标准，因此不返回未经充分支撑的完整草稿。",
        ]
        if supported:
            lines.extend([
                "",
                "当前证据能够支持的结论：",
                *(f"- {claim}" for claim in supported),
            ])
        else:
            lines.extend([
                "",
                "当前知识库证据不足，无法形成可靠结论。",
            ])
        lines.extend([
            "",
            "请上传与问题直接相关的 PDF、DOCX、TXT、MD 文献或实验数据，"
            "完成入库后重新提问；也可以缩小问题范围后重试。",
        ])
        answer = "\n".join(lines)
        detail = f"修订达上限；低支撑拒答（{rate:.0%}）"

    report.update({
        "fallback_tier": tier,
        "forced_exit": True,
    })
    # 不把未通过 Grounding 的内容写入 LTM，避免污染后续记忆召回。
    return {
        "final_answer": answer,
        "grounding": report,
        "trace": [{"step": "safe_fallback", "detail": detail}],
    }


def route_after_critic(state: AgentState) -> str:
    grounding = state.get("grounding") or {}
    if grounding.get("grounded"):
        return "pass"
    # 纯拒答/无论断：再修订也补不出证据，直接证据不足兜底（提示上传）
    if grounding.get("no_claims") or grounding.get("refusal"):
        return "fallback"
    if state.get("revise_count", 0) >= _max_revise():
        return "fallback"
    return "revise"


def _build_deep_graph():
    g = StateGraph(AgentState)
    g.add_node("plan", node_plan)
    g.add_node("research", node_research)
    g.add_node("write", node_write)
    g.add_node("critic", node_critic)
    g.add_node("consolidate", node_consolidate)
    g.add_node("safe_fallback", node_safe_fallback)
    g.set_entry_point("plan")
    g.add_edge("plan", "research")
    g.add_edge("research", "write")
    g.add_edge("write", "critic")
    g.add_conditional_edges(
        "critic",
        route_after_critic,
        {"pass": "consolidate", "revise": "write", "fallback": "safe_fallback"},
    )
    g.add_edge("consolidate", END)
    g.add_edge("safe_fallback", END)
    return g.compile()


_DEEP_APP = _build_deep_graph()


def node_deep_research(state: AgentState) -> dict:
    sub = _DEEP_APP.invoke(state)
    out: dict = {"trace": sub.get("trace", [])}
    for key in (
        "final_answer", "draft", "grounding", "plan", "replans", "findings", "wm_items",
    ):
        if key in sub:
            out[key] = sub[key]
    return out


# ===================== 主图 ===================== #
def build_dispatch_graph(*, checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("workflow", node_workflow)
    g.add_node("chat", node_chat)
    g.add_node("policy", node_policy)
    g.add_node("deep_research", node_deep_research)

    g.set_entry_point("workflow")
    g.add_conditional_edges(
        "workflow",
        route_by_agent,
        {"chat": "chat", "policy": "policy", "deep_research": "deep_research"},
    )
    g.add_edge("chat", END)
    g.add_edge("policy", END)
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
        log.info(
            f"多模态输入：{sum(1 for a in att if a.get('kind') == 'image')} 图，"
            f"{sum(1 for a in att if a.get('kind') == 'document')} 文档"
        )

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
