"""
src/agents/critic_agent.py — 批评家 / 质检 Agent
职责：调用 grounding 幻觉核验，对 Writer 的答案做事实核查。
若判定疑似幻觉，产出"修订说明"，驱动 Writer 重写（闭环纠错）。
这是"解决幻觉"在多Agent协作里的落点。

【优化】不再仅依赖工作记忆中的碎片证据，同时主动从 RAG 知识库检索
与问题和各论断相关的证据，大幅降低因"证据不足"导致的误判。
"""
from __future__ import annotations

from src.agents.base_agent import BaseAgent
from src.rag.grounding import grounding_checker
from src.rag.retriever import retriever
from src.utils.logger import get_logger


class CriticAgent(BaseAgent):
    name = "critic"
    system = "你是事实核查与质量把关专家。"

    def review(self, answer: str, question: str = "") -> dict:
        # ----- 第一步：从工作记忆收集已有证据 -----
        evidence = [{"content": f.content} for f in self.wm.facts()]
        wm_count = len(evidence)

        # ----- 第二步：主动从 RAG 知识库检索与问题相关的证据 -----
        rag_count = 0
        if question:
            try:
                rag_hits = retriever.search(question, k=5)
                # 去重（避免与工作记忆重复）
                existing_contents = {e["content"] for e in evidence}
                for hit in rag_hits:
                    if hit["content"] not in existing_contents:
                        evidence.append({
                            "content": hit["content"],
                            "filename": hit.get("filename", "?"),
                            "score": hit.get("score", 0),
                        })
                        rag_count += 1
            except Exception as e:
                self.log.warning(f"RAG 检索失败(question={question[:40]}): {e}")

        if rag_count > 0:
            self.log.info(f"从RAG补充 {rag_count} 条证据（原工作记忆 {wm_count} 条）")

        # ----- 第三步：执行幻觉核验（含 per-claim 针对性 RAG 兜底） -----
        report = grounding_checker.check(answer, evidence, question=question)

        # ----- 第四步：输出结果 -----
        if report["grounded"]:
            self.say("答案通过事实核查", to="writer")
            report["revise_note"] = ""
        else:
            note = "；".join(report["unsupported"][:5])
            self.log.warning(
                f"疑似幻觉（支撑率={report['support_rate']:.0%}）"
                f"证据来源：WM={wm_count} RAG={rag_count}"
            )
            self.say(f"发现疑似幻觉，需修订：{note}", to="writer")
            report["revise_note"] = note
        report["evidence_sources"] = {"working_memory": wm_count, "rag_direct": rag_count}
        return report
