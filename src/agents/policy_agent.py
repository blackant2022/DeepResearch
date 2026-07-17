"""
src/agents/policy_agent.py — 统一策略 Agent（Workflow 之外的唯一决策节点）

把图级 ReAct（think ↔ tools 节点跳转）收敛为节点内策略循环：
  Decision → Tool(s) → Observe → Decision …
直至拿到最终答案或达轮次上限。

收益：消除 Thought/Action/Iteration 多节点空转带来的图调度开销。
"""
from __future__ import annotations

from config.settings import settings
from src.agents.super_agent import SuperAgent
from src.llm.messages import repair_tool_message_chain
from src.llm.provider import llm
from src.memory.context_manager import context_manager
from src.memory.long_term_memory import ltm
from src.memory.working_memory import WorkingMemory
from src.orchestrator.nodes.tool_node import execute_tool_calls
from src.rag.confidence import (
    apply_answer_confidence_gate,
    extract_knowledge_hits_from_messages,
)
from src.utils.logger import get_logger

log = get_logger("policy_agent")


class PolicyAgent:
    """统一策略决策：一次图节点完成多轮工具调用。"""

    def __init__(self, wm: WorkingMemory | None = None) -> None:
        self.wm = wm or WorkingMemory()
        self.thinker = SuperAgent(self.wm)

    def run(
        self,
        question: str,
        memory_hint: str = "",
        max_iterations: int | None = None,
    ) -> dict:
        """
        返回:
          final_answer, messages, react_iteration, trace, wm_items
        """
        max_iter = max_iterations or settings.MAX_REACT_ITERATIONS
        messages = self.thinker.build_initial_messages(question, memory_hint)
        trace: list[dict] = []
        answer = ""
        iteration = 0

        for iteration in range(1, max_iter + 1):
            safe = context_manager.prepare_for_policy(messages)
            result = self.thinker.think_with_tools(safe)
            assistant_msg = result["assistant_message"]
            tool_calls = result.get("tool_calls") or []
            messages.append(assistant_msg)

            if not tool_calls:
                answer = result.get("content") or assistant_msg.get("content") or ""
                trace.append({
                    "step": "policy",
                    "detail": f"第 {iteration} 轮策略决策 → 最终回答",
                })
                break

            tool_names = [
                (tc.get("function") or {}).get("name", "?") for tc in tool_calls
            ]
            trace.append({
                "step": "policy",
                "detail": f"第 {iteration} 轮策略决策 → 调用 {len(tool_calls)} 个工具（{', '.join(tool_names)}）",
            })
            tool_msgs, tool_trace = execute_tool_calls(tool_calls, self.wm)
            messages.extend(tool_msgs)
            trace.extend(tool_trace)
        else:
            # 达上限仍带着 tool_calls：强制收束
            log.warning(f"策略环达到上限 {max_iter}，兜底生成")
            messages = repair_tool_message_chain(messages)
            messages.append({
                "role": "user",
                "content": "请根据以上工具结果直接给出最终回答，不要再调用工具。",
            })
            answer = llm.chat(context_manager.prepare_for_policy(messages))
            trace.append({"step": "policy_finalize", "detail": "达到轮次上限，兜底生成"})

        if answer:
            used_kb = False
            for m in messages:
                if m.get("role") != "assistant":
                    continue
                for tc in m.get("tool_calls") or []:
                    if (tc.get("function") or {}).get("name") == "knowledge_search":
                        used_kb = True
                        break
                if used_kb:
                    break
            hits = extract_knowledge_hits_from_messages(messages)
            if used_kb and settings.ANSWER_CONFIDENCE_ENABLED:
                answer, conf_report = apply_answer_confidence_gate(
                    answer, hits, used_knowledge=True
                )
                if conf_report.get("fallback"):
                    trace.append({
                        "step": "confidence_gate",
                        "detail": (
                            f"置信度 {conf_report.get('confidence')} "
                            f"< {conf_report.get('threshold')}，已输出兜底提示"
                        ),
                        "meta": conf_report,
                    })
                else:
                    trace.append({
                        "step": "confidence_gate",
                        "detail": f"置信度 {conf_report.get('confidence')} 通过",
                        "meta": conf_report,
                    })

            ltm.consolidate(f"问题：{question}\n结论：{answer[:400]}", question)
            trace.append({"step": "policy_done", "detail": "策略 Agent 作答完成"})

        return {
            "final_answer": answer,
            "messages": messages,
            "react_iteration": iteration,
            "trace": trace,
            "wm_items": self.wm.export_snapshot(),
        }
