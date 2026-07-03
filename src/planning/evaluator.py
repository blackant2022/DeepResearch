"""
src/planning/evaluator.py — 规划质量评估 & 重规划

【12｜如何评估 Agent 的规划质量？规划失败时如何改进？】
本模块用 LLM 作为“评委”，从四个维度给规划打分（0~1）：
  - completeness 完整性：子任务合起来能否覆盖原问题
  - feasibility  可行性：每步是否可用现有工具完成
  - ordering     顺序性：依赖关系是否合理、无循环
  - efficiency   高效性：有无冗余/可合并步骤
综合分低于阈值 → 反馈问题点，触发 replan（把评审意见喂回规划器改进）。
这构成“规划-评估-改进”的闭环，是让 Agent 稳定的关键。
"""
from __future__ import annotations

from src.llm.provider import llm
from src.planning.planner import planner
from src.utils.logger import get_logger

log = get_logger("plan_eval")

_EVAL_PROMPT = """你是规划质量评审专家。针对【原问题】评估下面的【规划】。
从 completeness / feasibility / ordering / efficiency 四维各打 0~1 分，
并给出综合分 overall(0~1) 与最多3条改进建议。

【原问题】
{question}

【规划】
{plan}

只输出 JSON：
{{"completeness":0.x,"feasibility":0.x,"ordering":0.x,"efficiency":0.x,
  "overall":0.x,"suggestions":["…"]}}"""


class PlanEvaluator:
    PASS = 0.7  # 综合分阈值

    def evaluate(self, question: str, plan: list[dict]) -> dict:
        plan_text = "\n".join(f"{p['id']}. [{p['tool']}] {p['goal']} (依赖{p['depends_on']})" for p in plan)
        score = llm.chat_json(
            [{"role": "user", "content": _EVAL_PROMPT.format(question=question, plan=plan_text)}]
        )
        overall = float(score.get("overall", 0.0) or 0.0)
        score["passed"] = overall >= self.PASS
        log.info(f"规划评分 overall={overall:.2f} → {'通过' if score['passed'] else '需改进'}")
        return score

    def refine(self, question: str, plan: list[dict], suggestions: list[str]) -> list[dict]:
        """把评审建议并入问题，重新规划（规划失败时的改进）。"""
        hint = "上一版规划的问题：" + "；".join(suggestions)
        log.info("按评审意见重新规划…")
        return planner.make_plan(question, memory_hint=hint)


plan_evaluator = PlanEvaluator()
