"""
src/agents/planner_agent.py — 规划 Agent（对 planner/evaluator 的 Agent 化封装）
职责：产出规划 → 自评 → 不达标则重规划（最多 MAX_REPLAN 次）。
"""
from __future__ import annotations

from config.settings import settings
from src.agents.base_agent import BaseAgent
from src.planning.evaluator import plan_evaluator
from src.planning.planner import planner


class PlannerAgent(BaseAgent):
    name = "planner"
    system = "你是任务规划总控。"

    def plan(self, question: str, memory_hint: str = "") -> dict:
        plan = planner.make_plan(question, memory_hint=memory_hint)
        evaluation = plan_evaluator.evaluate(question, plan)
        replans = 0
        while not evaluation["passed"] and replans < settings.MAX_REPLAN:
            replans += 1
            plan = plan_evaluator.refine(question, plan, evaluation.get("suggestions", []))
            evaluation = plan_evaluator.evaluate(question, plan)
        self.say(f"规划完成，共{len(plan)}步，重规划{replans}次", to="researcher")
        return {"plan": plan, "evaluation": evaluation, "replans": replans}
