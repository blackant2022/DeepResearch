"""
src/agents/base_agent.py — Agent 抽象基类
每个 Agent 都持有：名字、系统提示词、共享工作记忆的引用。
统一 think() 便捷方法（带 system 的一次 LLM 调用）。
"""
from __future__ import annotations

from abc import ABC
from src.llm.provider import llm
from src.memory.working_memory import WorkingMemory
from src.utils.logger import get_logger


class BaseAgent(ABC):
    name: str = "agent"
    system: str = "你是一个有帮助的智能体。"

    def __init__(self, wm: WorkingMemory) -> None:
        self.wm = wm                     # 共享工作记忆（多Agent通信的载体）
        self.log = get_logger(f"agent.{self.name}")

    def think(self, user_content: str, temperature: float | None = None) -> str:
        return llm.chat(
            [{"role": "system", "content": self.system},
             {"role": "user", "content": user_content}],
            temperature=temperature,
        )

    def think_json(self, user_content: str, temperature: float = 0.0) -> dict:
        return llm.chat_json(
            [{"role": "system", "content": self.system},
             {"role": "user", "content": user_content}],
            temperature=temperature,
        )

    def say(self, content: str, to: str = "all") -> None:
        """向共享工作记忆广播一条消息（Agent 间通信）。"""
        self.wm.add_message(self.name, content, to=to)
        self.log.info(f"→[{to}] {content[:60]}")
