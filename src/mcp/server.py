"""
src/mcp/server.py — DeepResearch MCP Server

将本地 RAG 检索、知识库概览、联网搜索等能力以 MCP 标准协议对外暴露，
可供 Cursor、Claude Desktop 等 MCP Client 调用。
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

import src.tools  # noqa: F401 — 注册内置工具
from src.tools.base import registry
from src.tools.web_search import search_web

mcp = FastMCP(
    name="DeepResearch",
    instructions="学术文献 RAG 与科研助手工具集：知识库检索、文献目录、联网搜索、计算。",
)


@mcp.tool()
def knowledge_search(query: str, k: int = 5) -> list[dict]:
    """在本地 Chroma 知识库中做语义检索，返回文档片段、来源与相关度。"""
    tool = registry.get("knowledge_search")
    if tool is None:
        return []
    return tool(query=query, k=k).output or []


@mcp.tool()
def kb_overview() -> dict:
    """列出知识库全部文档名称、块数与内容摘要。"""
    tool = registry.get("kb_overview")
    if tool is None:
        return {"total_chunks": 0, "document_count": 0, "documents": []}
    return tool().output or {}


@mcp.tool()
def web_search(query: str, max_results: int = 5) -> dict:
    """联网搜索最新公开信息（Tavily / DuckDuckGo）。"""
    results = search_web(query, max_results=max_results)
    return {"query": query, "count": len(results), "results": results}


@mcp.tool()
def calculator(expression: str) -> float | int:
    """安全计算数学表达式（+ - * / ** % //）。"""
    tool = registry.get("calculator")
    if tool is None:
        raise RuntimeError("calculator 工具未注册")
    result = tool(expression=expression)
    if not result.ok:
        raise ValueError(result.error)
    return result.output


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
