"""
src/rag/query_rewrite.py — 用户 Query 改进（检索前意图澄清与关键词补全）

在路由与 RAG 检索之前，将口语化/含糊问题改写为更适合向量检索与 Agent 推理的表述，
提升召回相关度并降低因表述不清导致的误路由。
"""
from __future__ import annotations

from config.settings import settings
from src.llm.provider import llm
from src.orchestrator.router import is_chitchat
from src.utils.logger import get_logger

log = get_logger("query_rewrite")

_ATTACHMENT_MARKERS = ("【用户本次上传的文档内容】", "【图片视觉分析】")

_REWRITE_PROMPT = """你是学术文献检索领域的 Query 优化专家。请将【用户问题】改写为更适合向量检索与科研 Agent 推理的版本。

要求：
1. 严格保留用户原意与约束，不编造新需求或新事实
2. 补充领域同义词、英文缩写（如 高光谱/Hyperspectral、氮含量/LNC、深度学习/DL）
3. 将含糊指代替换为明确对象（如「这个方法」→ 上下文中的具体方法名）
4. 控制在 120 字以内；已清晰简短的问题可只做微调或原样返回
5. 日常寒暄、身份询问、空问题原样返回

{memory_block}
【用户问题】
{question}

只输出 JSON：{{"rewritten": "改写后问题", "changes": "一句话说明改动（无改动则写「保持原问」）"}}"""


def _core_question(question: str) -> str:
    """去掉附件块，只对用户核心问句做改写。"""
    q = (question or "").strip()
    for marker in _ATTACHMENT_MARKERS:
        if marker in q:
            q = q.split(marker, 1)[0].strip()
    return q


def _preserve_attachments(full_question: str, rewritten_core: str) -> str:
    """保留原文中的文档/图片分析块。"""
    q = full_question or ""
    suffix_parts: list[str] = []
    for marker in _ATTACHMENT_MARKERS:
        if marker in q:
            suffix_parts.append(marker + q.split(marker, 1)[1])
    if not suffix_parts:
        return rewritten_core
    return rewritten_core + "\n\n" + "\n\n".join(suffix_parts)


def improve_query(question: str, memory_hint: str = "") -> dict:
    """
    返回:
      rewritten: 改写后的核心问句
      changes: 改动说明
      skipped: 是否跳过改写
    """
    core = _core_question(question)
    if not settings.QUERY_REWRITE_ENABLED:
        return {"rewritten": core, "changes": "改写已关闭", "skipped": True}
    if not core or len(core) < 4 or is_chitchat(core):
        return {"rewritten": core, "changes": "寒暄/过短，保持原问", "skipped": True}

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

    # 防止模型过度发挥：改写后过长则截断回退
    if len(rewritten) > 200:
        rewritten = core
        changes = "改写过长，保持原问"

    log.info(f"Query 改写: {core[:40]} → {rewritten[:40]}")
    return {"rewritten": rewritten, "changes": changes, "skipped": False}


def apply_query_rewrite(full_question: str, memory_hint: str = "") -> tuple[str, dict]:
    """对完整问题（含附件块）执行改写，返回 (新问题, 元信息)。"""
    result = improve_query(full_question, memory_hint=memory_hint)
    core = _core_question(full_question)
    rewritten = result["rewritten"]
    if rewritten == core:
        return full_question, result
    return _preserve_attachments(full_question, rewritten), result
