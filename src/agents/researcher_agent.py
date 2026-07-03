"""
src/agents/researcher_agent.py — 研究员 Agent
"""
from __future__ import annotations

from src.agents.base_agent import BaseAgent
from src.middleware.pipeline import pipeline
from src.rag.kb_utils import is_kb_catalog_question, normalize_chunk
from src.rag.retriever import retriever
from src.tools.base import registry

# 知识库主题宽检索词（农业高光谱领域）
_BROAD_KWS = ("高光谱", "遥感", "玉米", "氮含量", "植被", "hyperspectral")


class ResearcherAgent(BaseAgent):
    name = "researcher"
    system = (
        "你是严谨的研究员。你会依据工具返回的客观证据整理要点，"
        "绝不编造。若证据不足，明确说明‘证据不足’。"
    )

    def bootstrap_rag(self, question: str) -> int:
        """研究开始前预置证据：元问题列目录，普通问题做语义检索。"""
        if retriever.count() == 0:
            self.log.warning("知识库为空，跳过预检索")
            return 0
        if is_kb_catalog_question(question):
            return self._seed_kb_overview()
        return self._seed_semantic_search(question)

    def _seed_kb_overview(self) -> int:
        sources = retriever.list_sources()
        if not sources:
            return 0
        total = retriever.count()
        overview = f"知识库共 {len(sources)} 篇文档、{total} 个文本块。"
        self.wm.add_fact("retriever", overview, tag="kb_overview")
        for d in sources:
            line = f"《{d['filename']}》共{d['chunks']}块，摘要：{d.get('sample', '')}"
            self.wm.add_fact("retriever", line, filename=d["filename"], tag="kb_doc")
        for kw in _BROAD_KWS:
            for hit in retriever.search(kw, k=1):
                self.wm.add_fact(
                    "retriever", hit["content"], filename=hit["filename"], score=hit["score"]
                )
        self.log.info(f"知识库概览：{len(sources)} 篇文档")
        return len(sources)

    def _seed_semantic_search(self, question: str) -> int:
        hits = retriever.search(question)
        if not hits:
            return 0
        for hit in hits:
            self.wm.add_fact(
                "retriever", hit["content"], filename=hit["filename"], score=hit["score"]
            )
        self.log.info(f"预检索：{len(hits)} 条（query={question[:40]}）")
        return len(hits)

    def execute(self, subtask: dict, question: str) -> dict:
        goal = subtask["goal"]
        tool_name = self._resolve_tool(subtask.get("tool", "reason"), question)
        self.log.info(f"执行子任务#{subtask['id']}：{goal} (工具={tool_name})")

        evidence: list[dict] = []
        tool_note = ""

        tool = registry.get(tool_name)
        if tool is not None:
            args = self._tool_args(tool_name, goal, question)
            result = pipeline.invoke(tool, **args)
            if result.ok:
                evidence, tool_note = self._handle_tool_ok(tool_name, result, goal, question)
            else:
                tool_note = f"[{tool_name} 失败/{result.error_type}] {result.error}"

        finding = self._compose_finding(goal, tool_note, evidence)
        self.wm.add(self.name, finding, kind="observation", subtask_id=subtask["id"])
        for ev in evidence:
            self.wm.add_fact("retriever", ev["content"], filename=ev["filename"], score=ev["score"])

        self.say(f"子任务#{subtask['id']} 完成", to="writer")
        return {"subtask_id": subtask["id"], "finding": finding, "evidence": evidence, "tool_note": tool_note}

    def _resolve_tool(self, tool_name: str, question: str) -> str:
        if is_kb_catalog_question(question):
            if tool_name in ("reason", "web_search", "knowledge_search"):
                return "kb_overview"
        if tool_name == "reason" and retriever.count() > 0:
            return "knowledge_search"
        if registry.get(tool_name) is None and retriever.count() > 0:
            return "knowledge_search"
        return tool_name

    @staticmethod
    def _tool_args(tool_name: str, goal: str, question: str) -> dict:
        if tool_name == "calculator":
            return {"expression": ResearcherAgent._extract_expr(goal)}
        if tool_name == "kb_overview":
            return {}
        return {"query": goal or question}

    def _handle_tool_ok(self, tool_name: str, result, goal: str, question: str) -> tuple[list[dict], str]:
        if tool_name == "kb_overview" and isinstance(result.output, dict):
            evidence = self._evidence_from_overview(result.output)
            note = self._format_overview(result.output)
            return evidence, note

        if tool_name == "knowledge_search" and isinstance(result.output, list):
            evidence = self._collect_evidence(result.output)
            if not evidence and goal != question:
                retry = pipeline.invoke(registry.get("knowledge_search"), query=question)
                if retry.ok and isinstance(retry.output, list):
                    evidence = self._collect_evidence(retry.output)
                    return evidence, self._format_hits(retry.output)
            return evidence, self._format_hits(result.output)

        return [], f"[{tool_name} 成功] {result.output}"

    @staticmethod
    def _evidence_from_overview(data: dict) -> list[dict]:
        out = []
        for d in data.get("documents", []):
            content = f"《{d['filename']}》共{d['chunks']}块。{d.get('sample', '')}"
            out.append({"content": content, "filename": d["filename"], "score": 1.0})
        return out

    @staticmethod
    def _format_overview(data: dict) -> str:
        lines = [f"共 {data.get('document_count', 0)} 篇文档 / {data.get('total_chunks', 0)} 块"]
        for d in data.get("documents", []):
            lines.append(f"- {d['filename']}（{d['chunks']}块）")
        return "\n".join(lines)

    @staticmethod
    def _format_hits(hits: list) -> str:
        lines = []
        for h in hits[:5]:
            src = h.get("source") or h.get("filename", "?")
            body = normalize_chunk(str(h.get("content", "")))[:280]
            lines.append(f"[{src}] {body}")
        return "\n".join(lines) if lines else "（检索无命中）"

    def _compose_finding(self, goal: str, tool_note: str, evidence: list[dict]) -> str:
        if evidence:
            raw = self._format_hits(
                [{"content": e["content"], "source": e["filename"], "score": e["score"]} for e in evidence]
            )
            summary = self.think(
                f"子任务：{goal}\n已检索到以下证据：\n{raw}\n"
                "请用2-4句话总结关键信息，必须引用上述证据，不可说「无有效信息」。"
            )
            return f"【证据】\n{raw}\n\n【要点】{summary}"

        if tool_note and "失败" not in tool_note:
            return self.think(f"子任务：{goal}\n工具结果：{tool_note}\n请总结要点。")

        return self.think(
            f"子任务：{goal}\n工具结果：{tool_note or '无'}\n"
            "如实说明证据不足，不要编造。"
        )

    @staticmethod
    def _collect_evidence(hits: list) -> list[dict]:
        return [
            {
                "content": normalize_chunk(str(r.get("content", ""))),
                "filename": r.get("source") or r.get("filename", "?"),
                "score": r.get("score", 0),
            }
            for r in hits
            if r.get("content")
        ]

    @staticmethod
    def _extract_expr(text: str) -> str:
        import re
        m = re.findall(r"[0-9\.\+\-\*\/\(\)\s%]+", text)
        return max(m, key=len).strip() if m else "0"
