"""问题路由：规则层快速判别（供 RouterAgent 使用）。"""
from __future__ import annotations

import re

from src.rag.kb_utils import is_kb_catalog_question, is_kb_knowledge_summary

# 闲聊 / 身份 / 用法 —— 不走 RAG，直接对话
_CHITCHAT = re.compile(
    r"^\s*("
    r"你好|您好|嗨|hi|hello|hey|"
    r"你是谁|你是什么|你是啥|你叫什么|你的名字|"
    r"你能做什么|你会什么|你能帮什么|怎么用你|如何使用|"
    r"介绍一下你自己|自我介绍|"
    r"谢谢|感谢|多谢|"
    r"再见|拜拜|bye"
    r")\s*[？?！!。.]*\s*$",
    re.I,
)

_CHITCHAT_CONTAINS = (
    "你是谁", "你是什么", "你是哪个", "介绍一下你", "你能做", "你会做", "怎么用",
)


def is_chitchat(question: str) -> bool:
    q = question.strip()
    if not q:
        return True
    if _CHITCHAT.match(q):
        return True
    if len(q) <= 24 and any(k in q for k in _CHITCHAT_CONTAINS):
        return True
    return False


def route_question(question: str) -> str:
    """返回: chitchat | kb_catalog | kb_summary | research"""
    if is_chitchat(question):
        return "chitchat"
    if is_kb_catalog_question(question):
        return "kb_catalog"
    if is_kb_knowledge_summary(question):
        return "kb_summary"
    return "research"
