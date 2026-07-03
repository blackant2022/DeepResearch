"""
src/agents/chat_agent.py — 日常对话 Agent

处理：问候、身份介绍、能力说明、感谢告别等与文献无关的对话。
不检索知识库，直接调用 LLM 实时回复。
"""
from __future__ import annotations

from src.agents.base_agent import BaseAgent
from src.rag.retriever import retriever


class ChatAgent(BaseAgent):
    name = "chat"
    system = (
        "你是 DeepResearch 的日常对话助手。\n"
        "【身份】基于本地文献库的多 Agent 研究系统，核心能力是检索和总结用户上传的 PDF 学术文献"
        "（如高光谱遥感、作物氮含量等）。\n"
        "【原则】\n"
        "1) 友好、简洁地用中文回答；\n"
        "2) 用户问你是谁/能做什么时，介绍上述能力，不要从文献里找答案；\n"
        "3) 若用户想查文献，引导其直接提问研究问题。\n"
        "4) 不要编造论文内容或实验数据。"
    )

    def reply(self, question: str) -> str:
        kb = retriever.count()
        extra = f"\n（当前知识库：{kb} 个文本块）" if kb else "\n（当前知识库为空，可提醒用户左侧上传 PDF）"
        answer = self.think(f"{question}{extra}", temperature=0.6)
        self.wm.add(self.name, answer, kind="scratch", tag="chat_reply")
        self.say("日常对话完成", to="all")
        return answer
