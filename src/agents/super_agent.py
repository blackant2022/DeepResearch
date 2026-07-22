"""
src/agents/super_agent.py — 超级智能体（ReAct + Function Calling）

LLM 自主决定何时调用何种工具，形成 think → act → observe 闭环。
"""
from __future__ import annotations

from src.agents.base_agent import BaseAgent
from src.llm.provider import llm
from src.tools.base import registry

_SUPER_SYSTEM = """你是 DeepResearch 超级智能体，面向学术文献与科研知识库。

【能力】
- 可调用工具检索本地文献、查看知识库目录、精确计算、联网搜索互联网
- 若工具列表有 [MCP] 或名称含 baidu 的工具，用户要「百度搜索」时必须优先调用它们，不要用 web_search
- 可综合多轮工具结果给出有据回答

【规则】
1. 问「知识库有什么/有哪些文档」→ 只用 kb_overview，不要 knowledge_search
2. 涉及文献内容、研究方法、实验结论 → 先 knowledge_search（全程最多 2 次，k 默认 3）
3. 用户明确要求上网、或本地库以外的最新公开信息 → web_search
4. 只依据工具返回的证据作答；证据不足必须明确说「根据现有资料无法确定」
5. 数值问题用 calculator
6. 一次检索够用就直接作答，禁止换同义说法反复 knowledge_search
7. 用户可能附带图片或文档，请参考【图片视觉分析】【用户本次上传的文档内容】
8. 用中文回答，结构清晰；引用网页时注明标题或链接"""


class SuperAgent(BaseAgent):
    name = "super"
    system = _SUPER_SYSTEM

    def build_initial_messages(self, question: str, memory_hint: str = "") -> list[dict]:
        user = question
        if memory_hint:
            user = f"【历史研究经验】\n{memory_hint}\n\n【当前问题】\n{question}"
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": user},
        ]

    def think_with_tools(self, messages: list[dict]) -> dict:
        """单轮推理：可能返回 tool_calls 或最终文本。"""
        from src.llm.messages import repair_tool_message_chain

        tools = registry.openai_schemas()
        safe_messages = repair_tool_message_chain(messages)
        result = llm.chat_with_tools(safe_messages, tools)
        assistant: dict = {"role": "assistant", "content": result.get("content")}
        if result.get("tool_calls"):
            assistant["tool_calls"] = result["tool_calls"]
        return {
            "assistant_message": assistant,
            "content": result.get("content"),
            "tool_calls": result.get("tool_calls") or [],
        }
