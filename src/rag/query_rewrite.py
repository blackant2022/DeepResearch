"""
src/rag/query_rewrite.py — 用户 Query 改进（默认关闭，仅短/含糊问题按需开启）
"""
from __future__ import annotations

import re

from config.settings import settings
from src.llm.provider import llm
from src.orchestrator.router import is_chitchat
from src.utils.logger import get_logger

log = get_logger("query_rewrite")

_ATTACHMENT_MARKERS = ("【用户本次上传的文档内容】", "【图片视觉分析】")
_AMBIGUOUS = re.compile(
    r"(这个|那个|它|他们|上述|刚才|上面|怎么弄|怎么办|分析一下|讲讲|说说|"
    r"帮我看看|是啥|咋样|怎么样$|有哪些(?!具体))"
)

_REWRITE_PROMPT = """你是学术文献检索领域的 Query 优化专家。请将【用户问题】改写为更适合向量检索与科研 Agent 推理的版本。

要求：
1. 严格保留用户原意与约束，不编造新需求或新事实
2. 补充领域同义词、英文缩写（如 高光谱/Hyperspectral、氮含量/LNC、深度学习/DL）
3. 将含糊指代替换为明确对象
4. 控制在 120 字以内；已清晰的问题可原样返回
5. 日常寒暄、身份询问原样返回

{memory_block}
【用户问题】
{question}

只输出 JSON：{{"rewritten": "改写后问题", "changes": "一句话说明改动（无改动则写「保持原问」）"}}"""


def _core_question(question: str) -> str:
    q = (question or "").strip()
    for marker in _ATTACHMENT_MARKERS:
        if marker in q:
            q = q.split(marker, 1)[0].strip()
    return q


def _preserve_attachments(full_question: str, rewritten_core: str) -> str:
    q = full_question or ""
    suffix_parts: list[str] = []
    for marker in _ATTACHMENT_MARKERS:
        if marker in q:
            suffix_parts.append(marker + q.split(marker, 1)[1])
    if not suffix_parts:
        return rewritten_core
    return rewritten_core + "\n\n" + "\n\n".join(suffix_parts)


def _needs_rewrite(core: str) -> bool:
    """仅对短问或含糊指代开改写；长且清晰的文献问句直接跳过。"""
    if len(core) <= 28:
        return True
    if _AMBIGUOUS.search(core) and len(core) <= 60:
        return True
    return False


def improve_query(question: str, memory_hint: str = "") -> dict:
    core = _core_question(question)
    if not settings.QUERY_REWRITE_ENABLED:
        return {"rewritten": core, "changes": "改写已关闭（快路径）", "skipped": True}
    if not core or len(core) < 4 or is_chitchat(core):
        return {"rewritten": core, "changes": "寒暄/过短，保持原问", "skipped": True}
    if not _needs_rewrite(core):
        return {"rewritten": core, "changes": "问题已清晰，跳过改写", "skipped": True}

    memory_block = ""
    if memory_hint.strip():
        memory_block = f"【可参考的历史经验】\n{memory_hint.strip()}\n\n"

    try:
        data = llm.chat_json(
            [{"role": "user", "content": _REWRITE_PROMPT.format(
                question=core, memory_block=memory_block
            )}],
            temperature=0.2,
        )
        rewritten = (data.get("rewritten") or core).strip()
        changes = (data.get("changes") or "保持原问").strip()
    except Exception as e:
        log.warning(f"Query 改写失败，使用原问: {e}")
        return {"rewritten": core, "changes": f"改写失败: {e}", "skipped": True}

    if len(rewritten) > 200:
        rewritten = core
        changes = "改写过长，保持原问"

    log.info(f"Query 改写: {core[:40]} → {rewritten[:40]}")
    return {"rewritten": rewritten, "changes": changes, "skipped": False}


def apply_query_rewrite(full_question: str, memory_hint: str = "") -> tuple[str, dict]:
    result = improve_query(full_question, memory_hint=memory_hint)
    core = _core_question(full_question)
    rewritten = result["rewritten"]
    if rewritten == core:
        return full_question, result
    return _preserve_attachments(full_question, rewritten), result
