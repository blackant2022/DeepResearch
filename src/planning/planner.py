"""
src/planning/planner.py — 任务规划器

【10｜Agent 如何进行任务规划？规划算法有哪些？】
常见规划范式：
  - 单步 ReAct（想一步做一步，适合简单任务）
  - Plan-and-Execute（先整体规划再逐步执行，适合复杂任务）← 本项目采用
  - 树搜索类（ToT / LATS，成本高，本项目不用但在 README 讨论）
本规划器用 LLM 做“分解式规划”(Decomposition)：把用户问题拆成有序、
可独立执行、彼此依赖清晰的子任务，每个子任务标注建议使用的工具。
"""
from __future__ import annotations

from config.settings import settings
from src.llm.provider import llm
from src.rag.kb_utils import is_kb_catalog_question, is_kb_knowledge_summary
from src.tools.base import registry
from src.utils.logger import get_logger

log = get_logger("planner")

_PLAN_PROMPT = """你是资深任务规划专家。请把【用户问题】分解为 1~{max_n} 个有序子任务。
可用工具清单：
{tools}

要求：
- 每个子任务应聚焦单一目标、可独立完成。
- 为每个子任务指定最合适的一个工具名（从清单里选，纯推理则用 "reason"）。
- **涉及知识库、文档、论文、资料总结类问题，子任务必须使用 knowledge_search，不要用 reason。**
- **当用户问「知识库里有什么/有哪些文档/现有内容」时，第一个子任务必须用 kb_overview。**
- 子任务之间可有先后依赖（后面的可用前面的结果）。
- 若已有【历史相似经验】，参考它避免走弯路。

【历史相似经验】
{memory}

【用户问题】
{question}

只输出 JSON：
{{"subtasks": [{{"id": 1, "goal": "…", "tool": "knowledge_search", "depends_on": []}}]}}"""


class Planner:
    def make_plan(self, question: str, memory_hint: str = "") -> list[dict]:
        hint = memory_hint or ""
        if is_kb_catalog_question(question):
            hint += "\n【提示】这是知识库目录类问题，子任务使用 kb_overview。"
        elif is_kb_knowledge_summary(question):
            hint += "\n【提示】这是知识综合总结，子任务使用 knowledge_search 检索后归纳，不要用 kb_overview。"
        prompt = _PLAN_PROMPT.format(
            max_n=settings.MAX_SUBTASKS,
            tools=registry.manifest(),
            memory=hint or "（无）",
            question=question,
        )
        data = llm.chat_json([{"role": "user", "content": prompt}])
        subtasks = data.get("subtasks", [])
        # 规整：确保字段齐全
        clean = []
        for i, st in enumerate(subtasks[: settings.MAX_SUBTASKS], 1):
            clean.append({
                "id": st.get("id", i),
                "goal": st.get("goal", "").strip(),
                "tool": st.get("tool", "reason"),
                "depends_on": st.get("depends_on", []),
            })
        log.info(f"生成规划：{len(clean)} 个子任务")
        return clean


planner = Planner()
