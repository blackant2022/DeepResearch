"""
src/agents/router_agent.py — 路由 Agent（规则优先，FAST_MODE 下尽量不调 LLM）

输出：chat | super | deep_research
"""
from __future__ import annotations

import re

from config.settings import settings
from src.agents.base_agent import BaseAgent
from src.orchestrator.router import is_chitchat
from src.rag.kb_utils import is_kb_catalog_question, is_kb_knowledge_summary

_COMPLEX = re.compile(
    r"(对比|比较|综合分析|深入分析|全面综述|详细阐述|优缺点|多角度|"
    r"深度研究|分步分析|系统梳理|全面调研)"
)
_WEB = re.compile(r"(上网|联网|搜索|查一下|最新|新闻|天气|今日|今天)")


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

        # 短问题：FAST_MODE 下强制走 Policy，绝不进深研、不调 LLM 分类
        if settings.FAST_MODE and len(q) <= 40:
            if self._looks_like_literature(q) or _WEB.search(q):
                return {"agent": "super", "reason": "短问直达策略 Agent（快路径）"}
            return {"agent": "super", "reason": "短问默认策略 Agent（快路径）"}

        if self._needs_deep_research(q):
            return {"agent": "deep_research", "reason": "复杂研究任务"}

        if self._looks_like_literature(q):
            return {"agent": "super", "reason": "文献/知识库检索"}

        if _WEB.search(q):
            return {"agent": "super", "reason": "联网/时效类问题"}

        # FAST_MODE：剩余一律默认 super，跳过 LLM 分类
        if settings.FAST_MODE:
            return {"agent": "super", "reason": "默认策略 Agent（FAST_MODE 跳过 LLM 分类）"}

        if len(q) >= 8:
            llm_route = self._llm_classify(q)
            if llm_route:
                return llm_route

        return {"agent": "super", "reason": "默认超级智能体"}

    def _needs_deep_research(self, q: str) -> bool:
        """FAST_MODE 下深研门槛更高：长问 + 复杂意图词。"""
        if settings.FAST_MODE is False:
            return bool(_COMPLEX.search(q)) or len(q) > 80
        # 快模式：必须同时满足长度与复杂意图，避免误进 4-Agent
        return len(q) > 80 and bool(_COMPLEX.search(q))

    @staticmethod
    def _looks_like_literature(q: str) -> bool:
        keys = (
            "文献", "论文", "知识库", "文档", "资料", "研究", "遥感", "高光谱",
            "玉米", "氮", "模型", "方法", "总结", "概述", "是什么", "如何", "怎么",
            "反演", "估算", "实验", "精度", "算法", "深度学习", "检索",
        )
        return any(k in q for k in keys)

    def _llm_classify(self, question: str) -> dict | None:
        data = self.think_json(
            f"用户问题：{question}\n\n"
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
