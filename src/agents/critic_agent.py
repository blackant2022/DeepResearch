"""
src/agents/critic_agent.py — 批评家 / 质检 Agent

轻量：少量 RAG 补证据 + 批量 Grounding；不做大规模 per-claim 检索。
"""
from __future__ import annotations

from config.settings import settings
from src.agents.base_agent import BaseAgent
from src.rag.grounding import grounding_checker
from src.rag.retriever import retriever


class CriticAgent(BaseAgent):
    name = "critic"
    system = "你是事实核查与质量把关专家。"

    def review(self, answer: str, question: str = "") -> dict:
        evidence = [{"content": f.content} for f in self.wm.facts()]
        wm_count = len(evidence)

        rag_count = 0
        k = int(getattr(settings, "CRITIC_RAG_K", 3) or 3)
        skip_if = int(getattr(settings, "CRITIC_SKIP_RAG_IF_WM", 4) or 4)
        # WM 已有充足证据时跳过与 bootstrap 同 query 的重复检索
        if question and k > 0 and wm_count < skip_if:
            try:
                rag_hits = retriever.search(question, k=k)
                existing = {e["content"] for e in evidence}
                for hit in rag_hits:
                    if hit["content"] not in existing:
                        evidence.append({
                            "content": hit["content"],
                            "filename": hit.get("filename", "?"),
                            "score": hit.get("vector_score", hit.get("score", 0)),
                        })
                        rag_count += 1
            except Exception as e:  # noqa: BLE001
                self.log.warning(f"RAG 检索失败: {e}")
        elif wm_count >= skip_if and k > 0:
            self.log.info(f"跳过 Critic 补检索（WM={wm_count}≥{skip_if}）")

        if rag_count:
            self.log.info(f"从RAG补充 {rag_count} 条证据（WM={wm_count}）")

        report = grounding_checker.check(answer, evidence, question=question)

        if report["grounded"]:
            self.say("答案通过事实核查", to="writer")
            report["revise_note"] = ""
        else:
            note = "；".join(report["unsupported"][:3])
            self.log.warning(
                f"疑似幻觉（支撑率={report['support_rate']:.0%}）WM={wm_count} RAG={rag_count}"
            )
            self.say(f"发现疑似幻觉，需修订：{note}", to="writer")
            report["revise_note"] = note
        report["evidence_sources"] = {"working_memory": wm_count, "rag_direct": rag_count}
        return report
