"""
src/agents/writer_agent.py — 写作 Agent
职责：把所有研究员的发现 + 检索到的事实，综合成结构化最终答案。
关键：prompt 层面就施加“反幻觉约束”（只用证据/未知即说不知道），
作为幻觉治理的第一道防线（第二道是 grounding 后验）。
"""
from __future__ import annotations

from src.agents.base_agent import BaseAgent


class WriterAgent(BaseAgent):
    name = "writer"
    system = (
        "你是专业的技术写作者。严格遵守：\n"
        "1) 只依据提供的【发现】与【证据】作答，不得引入外部知识或臆测；\n"
        "2) 若【可用证据】中已有文档列表、检索片段或摘要，必须据此组织答案，"
        "不可声称「所有子任务均未返回有效信息」；\n"
        "3) 证据无法覆盖的部分，明确写「根据现有资料无法确定」；\n"
        "4) 结构清晰，必要时分点；语言简洁准确。"
    )

    def compose(self, question: str, findings: list[dict], revise_note: str = "") -> str:
        findings_text = "\n".join(f"- (子任务{f['subtask_id']}) {f['finding']}" for f in findings)
        evidence_text = self.wm.render(kinds=("fact",))
        extra = f"\n\n【修订要求】上一版存在未被证据支撑的论断，请修正或删除：\n{revise_note}" if revise_note else ""
        answer = self.think(
            f"【用户问题】\n{question}\n\n【各子任务发现】\n{findings_text}\n\n"
            f"【可用证据】\n{evidence_text}{extra}\n\n请撰写最终答案。",
            temperature=0.3,
        )
        self.wm.add(self.name, answer, kind="scratch", tag="draft")
        return answer
