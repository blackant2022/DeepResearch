"""
src/agents/search_agent.py — 文献搜索 Agent
"""
from __future__ import annotations

from config.settings import settings
from src.agents.base_agent import BaseAgent
from src.middleware.pipeline import pipeline
from src.rag.kb_utils import (
    build_kb_overview_answer,
    is_kb_catalog_question,
    is_kb_knowledge_summary,
    normalize_chunk,
)
from src.rag.retriever import retriever
from src.tools.base import registry

# 全库总结时的多角度检索词
_SUMMARY_QUERIES = (
    "高光谱遥感 作物 氮含量 估算",
    "深度学习 机器学习 预测模型 遥感",
    "玉米 农业 光谱 植被参数",
    "遥感 反演 方法 精度",
)


class SearchAgent(BaseAgent):
    name = "search"
    system = (
        "你是专业的文献检索与分析助手。\n"
        "只依据【检索证据】回答，证据不足必须说「根据现有资料无法确定」。\n"
        "不要编造，不要引入证据以外的知识。回答结构清晰，必要时分点。"
    )

    _SUMMARY_SYSTEM = (
        "你是学术论文分析专家。用户要求总结知识库中的【核心知识】，不是列文件名。\n"
        "请综合检索证据，按主题归纳实质内容，例如：\n"
        "- 研究领域与问题\n"
        "- 主要方法与技术路线\n"
        "- 关键结论与指标\n"
        "- 共同趋势或差异\n"
        "禁止只输出文档列表；每个要点应体现从文献中提炼的信息。"
    )

    def search_and_answer(self, question: str) -> dict:
        if retriever.count() == 0:
            return {"answer": "知识库为空，请先在左侧上传 PDF 并点击「确认入库」。", "hits": 0, "mode": "empty"}

        if is_kb_catalog_question(question):
            answer = build_kb_overview_answer()
            return {"answer": answer, "hits": retriever.stats()[1], "mode": "catalog"}

        if is_kb_knowledge_summary(question):
            return self._summarize_knowledge(question)

        return self._answer_with_rag(question)

    def _summarize_knowledge(self, question: str) -> dict:
        """多角度检索 + 综合归纳核心知识。"""
        hits = self._multi_retrieve([question, *_SUMMARY_QUERIES], per_query=4, max_total=16)
        for h in hits:
            self.wm.add_fact("search", h["content"], filename=h["filename"], score=h["score"])

        if not hits:
            return {"answer": "未能从知识库检索到可总结的内容。", "hits": 0, "mode": "summary_empty"}

        evidence = "\n\n".join(
            f"【{h['filename']}】\n{normalize_chunk(h['content'])[:700]}" for h in hits
        )
        stats = retriever.stats()
        from src.llm.provider import llm

        answer = llm.chat(
            [
                {"role": "system", "content": self._SUMMARY_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"问题：{question}\n知识库：{stats[1]} 篇文献 / {stats[0]} 个文本块\n\n"
                        f"文献证据：\n{evidence}\n\n请总结核心知识："
                    ),
                },
            ],
            temperature=0.35,
        )
        self.wm.add(self.name, answer, kind="scratch", tag="knowledge_summary")
        return {"answer": answer, "hits": len(hits), "mode": "summary"}

    def _answer_with_rag(self, question: str) -> dict:
        hits = self._retrieve(question)
        for h in hits:
            self.wm.add_fact("search", h["content"], filename=h["filename"], score=h["score"])

        if not hits:
            answer = self.think(
                f"用户问题：{question}\n检索结果：未找到相关内容。请如实告知并建议换关键词。"
            )
            return {"answer": answer, "hits": 0, "mode": "no_hit"}

        evidence = "\n\n".join(
            f"[{h['filename']}]（相关度 {h['score']:.2f}）\n{normalize_chunk(h['content'])[:600]}"
            for h in hits
        )
        answer = self.think(
            f"【用户问题】\n{question}\n\n【检索证据】\n{evidence}\n\n请回答：",
            temperature=0.3,
        )
        return {"answer": answer, "hits": len(hits), "mode": "rag"}

    def _multi_retrieve(self, queries: list[str], per_query: int = 4, max_total: int = 16) -> list[dict]:
        seen: set[str] = set()
        out: list[dict] = []
        for q in queries:
            for h in retriever.search(q, k=per_query):
                key = h["content"][:100]
                if key in seen:
                    continue
                seen.add(key)
                out.append(h)
                if len(out) >= max_total:
                    return out
        return out

    def _retrieve(self, question: str) -> list[dict]:
        tool = registry.get("knowledge_search")
        if tool:
            r = pipeline.invoke(tool, query=question, k=settings.TOP_K)
            if r.ok:
                raw = r.output
                hits = raw.get("hits") if isinstance(raw, dict) else raw
                if isinstance(hits, list):
                    return [
                        {
                            "content": x["content"],
                            "filename": x.get("source") or x.get("filename", "?"),
                            "score": x.get("score", 0),
                        }
                        for x in hits
                        if isinstance(x, dict)
                    ]
        return retriever.search(question, k=settings.TOP_K)
