"""
src/tools/builtin_tools.py — 内置工具集合
演示三类典型工具：
  1. KnowledgeSearchTool  — 检索型（接 RAG 知识库），Agent 获取事实的主力
  2. CalculatorTool       — 计算型（安全表达式求值），演示确定性工具
  3. WebSearchTool        — 外部型（Tavily / DuckDuckGo 联网搜索）
最后统一注册到 registry。
"""
from __future__ import annotations

import ast
import operator as op

from config.settings import settings
from src.rag.retriever import retriever
from src.tools.base import BaseTool, registry
from src.tools.web_search import search_web

# ---------------------------------------------------------------- #
# 1) 知识库检索工具
# ---------------------------------------------------------------- #
class KnowledgeSearchTool(BaseTool):
    name = "knowledge_search"
    description = "在本地知识库中做语义检索，返回最相关的文档片段（含来源文件名与相关度）。回答具体事实性问题时使用。"
    schema = {
        "query": {"type": "str", "desc": "检索关键词或问题", "required": True},
        "k": {"type": "int", "desc": "返回条数，默认5", "required": False},
    }

    def _run(self, query: str, k: int = 5):
        hits = retriever.search(query, k=k)
        return [{"content": h["content"], "source": h["filename"], "score": h["score"]} for h in hits]


# ---------------------------------------------------------------- #
# 1b) 知识库概览工具（列目录，解决「知识库里有什么」类问题）
# ---------------------------------------------------------------- #
class KnowledgeBaseOverviewTool(BaseTool):
    name = "kb_overview"
    description = (
        "列出本地知识库全部文档的名称、块数和内容摘要。"
        "当用户问「知识库里有什么」「有哪些文档/资料」「现有内容是什么」时必须优先使用。"
    )
    schema: dict[str, dict] = {}

    def _run(self):
        sources = retriever.list_sources()
        return {
            "total_chunks": retriever.count(),
            "document_count": len(sources),
            "documents": sources,
        }


# ---------------------------------------------------------------- #
# 2) 计算器工具（安全 AST 求值，禁止任意代码执行）
# ---------------------------------------------------------------- #
_ALLOWED = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Pow: op.pow, ast.Mod: op.mod, ast.USub: op.neg, ast.FloorDiv: op.floordiv,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED:
        return _ALLOWED[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED:
        return _ALLOWED[type(node.op)](_safe_eval(node.operand))
    raise ValueError("表达式包含不允许的操作")


class CalculatorTool(BaseTool):
    name = "calculator"
    description = "计算数学表达式，仅支持 + - * / ** % //，用于需要精确数值的场景。"
    schema = {"expression": {"type": "str", "desc": "如 (3+5)*2", "required": True}}

    def _run(self, expression: str):
        tree = ast.parse(expression, mode="eval")
        return _safe_eval(tree.body)


# ---------------------------------------------------------------- #
# 3) 联网搜索工具（Tavily / DuckDuckGo）
# ---------------------------------------------------------------- #
class WebSearchTool(BaseTool):
    name = "web_search"
    description = (
        "联网搜索互联网上的最新公开信息，返回标题、链接与摘要。"
        "当用户明确要求「上网搜索/联网查询/最新资讯」，或问题超出本地知识库范围时使用。"
        "本地文献问题应优先用 knowledge_search，不要滥用本工具。"
    )
    schema = {
        "query": {"type": "str", "desc": "搜索关键词或问题", "required": True},
        "max_results": {"type": "int", "desc": "返回条数，默认5", "required": False},
    }

    def _run(self, query: str, max_results: int | None = None):
        k = max_results or settings.WEB_SEARCH_MAX_RESULTS
        results = search_web(query, max_results=k)
        if not results:
            return {"query": query, "results": [], "note": "未检索到相关网页"}
        return {"query": query, "count": len(results), "results": results}


# ---- 注册 ----
registry.register(KnowledgeSearchTool())
registry.register(KnowledgeBaseOverviewTool())
registry.register(CalculatorTool())
registry.register(WebSearchTool())
