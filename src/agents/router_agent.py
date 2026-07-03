"""
src/agents/router_agent.py — 路由 Agent（总调度）

先用规则快速判别；不确定时再用 LLM 分类。
输出目标：chat（日常对话）| super（超级智能体 ReAct）| deep_research（深度研究）
"""
from __future__ import annotations

import re

from config.settings import settings
from src.agents.base_agent import BaseAgent
from src.orchestrator.router import is_chitchat
from src.rag.kb_utils import is_kb_catalog_question, is_kb_knowledge_summary
from src.rag.retriever import retriever

_COMPLEX = re.compile(r"(对比|比较|综合分析|深入分析|全面|详细阐述|为什么|如何实现|优缺点)")


class RouterAgent(BaseAgent):
    name = "router"
    system = "你是意图分类专家，负责把用户问题分发给最合适的 Agent。"

    def dispatch(self, question: str) -> dict:
        q = question.strip()
        if not q:
            return {"agent": "chat", "reason": "空输入"}

        if is_chitchat(q):
            return {"agent": "chat", "reason": "日常对话/身份/寒暄"}

        if is_kb_catalog_question(q):
            return {"agent": "super", "reason": "知识库文档目录"}

        if is_kb_knowledge_summary(q):
            return {"agent": "super", "reason": "知识库内容综合总结"}

        if self._needs_deep_research(q):
            return {"agent": "deep_research", "reason": "复杂研究任务"}

        if self._looks_like_literature(q):
            return {"agent": "super", "reason": "文献/知识库检索"}

        if len(q) >= 8:
            llm_route = self._llm_classify(q)
            if llm_route:
                return llm_route

        return {"agent": "super", "reason": "默认超级智能体"}

    def _needs_deep_research(self, q: str) -> bool:
        if settings.FAST_MODE is False:
            return True
        return len(q) > 50 and bool(_COMPLEX.search(q))

    @staticmethod
    def _looks_like_literature(q: str) -> bool:
        keys = (
            "文献", "论文", "知识库", "文档", "资料", "研究", "遥感", "高光谱",
            "玉米", "氮", "模型", "方法", "总结", "概述", "是什么", "如何", "怎么",
        )
        return any(k in q for k in keys)

    def _llm_classify(self, question: str) -> dict | None:
        kb_hint = f"知识库有 {retriever.count()} 块" if retriever.count() else "知识库为空"
        data = self.think_json(
            f"用户问题：{question}\n{kb_hint}\n\n"
            "请分类到唯一目标：\n"
            "- chat：日常闲聊、问候、问AI身份/能力、与文献无关的通用对话\n"
            "- super：需查本地文献库的事实问答、总结、概念解释、工具辅助任务\n"
            "- deep_research：需多步规划对比分析的复杂研究\n\n"
            '只输出 JSON：{"agent":"chat|super|deep_research","reason":"一句话"}'
        )
        agent = data.get("agent", "")
        if agent == "search":
            agent = "super"
        if agent in ("chat", "super", "deep_research"):
            return {"agent": agent, "reason": data.get("reason", "LLM 分类")}
        return None
