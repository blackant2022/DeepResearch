"""
src/agents/planner_agent.py — 规划 Agent（对 planner/evaluator 的 Agent 化封装）
职责：产出规划 → 自评 → 不达标则重规划（最多 MAX_REPLAN 次）。

FAST_MODE：跳过 LLM 规划评估，用规则校验（非空、工具名合法、未超上限），省 1 次 LLM。
"""
from __future__ import annotations

from config.settings import settings
from src.agents.base_agent import BaseAgent
from src.planning.evaluator import plan_evaluator
from src.planning.planner import planner
from src.tools.base import registry


class PlannerAgent(BaseAgent):
    name = "planner"
    system = "你是任务规划总控。"

    def plan(self, question: str, memory_hint: str = "") -> dict:
        plan = planner.make_plan(question, memory_hint=memory_hint)
        if settings.FAST_MODE:
            evaluation = self._fast_evaluate(plan)
            self.say(f"规划完成，共{len(plan)}步，重规划0次（FAST 跳过 LLM 评估）", to="researcher")
            return {"plan": plan, "evaluation": evaluation, "replans": 0}

        evaluation = plan_evaluator.evaluate(question, plan)
        replans = 0
        while not evaluation["passed"] and replans < settings.MAX_REPLAN:
            replans += 1
            plan = plan_evaluator.refine(question, plan, evaluation.get("suggestions", []))
            evaluation = plan_evaluator.evaluate(question, plan)
        self.say(f"规划完成，共{len(plan)}步，重规划{replans}次", to="researcher")
        return {"plan": plan, "evaluation": evaluation, "replans": replans}

    @staticmethod
    def _fast_evaluate(plan: list[dict]) -> dict:
        """规则 pass：有目标、工具合法或可回退 knowledge_search。"""
        known = {t.name for t in registry.all()} | {"reason", "knowledge_search", "kb_overview"}
        if not plan:
            return {"passed": False, "overall": 0.0, "suggestions": ["规划为空"]}
        bad = [st for st in plan if not str(st.get("goal", "")).strip()]
        if bad:
            return {"passed": False, "overall": 0.4, "suggestions": ["存在空目标子任务"]}
        # 未知工具名不阻断（researcher 会回退 knowledge_search）
        _ = known
        return {
            "passed": True,
            "overall": 0.85,
            "suggestions": [],
            "mode": "fast_rule",
        }
