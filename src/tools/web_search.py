"""
src/tools/web_search.py — 联网搜索后端

优先级（auto 模式）：
  1. Tavily API（需 TAVILY_API_KEY，适合 Agent）
  2. DuckDuckGo（免费，无需 Key）
  3. mock（仅当 WEB_SEARCH_PROVIDER=mock）
"""
from __future__ import annotations

from config.settings import secret_value, settings
from src.utils.logger import get_logger

log = get_logger("web_search")


def search_web(query: str, max_results: int | None = None) -> list[dict]:
    """统一联网搜索入口，返回 [{title, url, snippet}, ...]。"""
    query = (query or "").strip()
    if not query:
        return []

    k = max_results or settings.WEB_SEARCH_MAX_RESULTS
    provider = _resolve_provider()

    if provider == "tavily":
        return _search_tavily(query, k)
    if provider == "duckduckgo":
        return _search_duckduckgo(query, k)
    return _search_mock(query, k)


def _resolve_provider() -> str:
    p = (settings.WEB_SEARCH_PROVIDER or "auto").lower()
    if p == "mock":
        return "mock"
    if p == "tavily":
        if not secret_value(settings.TAVILY_API_KEY):
            raise RuntimeError("WEB_SEARCH_PROVIDER=tavily 但未在 .env 配置 TAVILY_API_KEY")
        return "tavily"
    if p == "duckduckgo":
        return "duckduckgo"
    # auto
    if secret_value(settings.TAVILY_API_KEY):
        return "tavily"
    return "duckduckgo"


def _search_tavily(query: str, k: int) -> list[dict]:
    try:
        from tavily import TavilyClient
    except ImportError as e:
        raise RuntimeError("请安装 tavily-python：pip install tavily-python") from e

    client = TavilyClient(api_key=secret_value(settings.TAVILY_API_KEY))
    resp = client.search(query, max_results=k, search_depth=settings.TAVILY_SEARCH_DEPTH)
    out: list[dict] = []
    for item in resp.get("results", []):
        out.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("content", "")[:800],
        })
    log.info(f"Tavily 返回 {len(out)} 条")
    return out


def _search_duckduckgo(query: str, k: int) -> list[dict]:
    try:
        from duckduckgo_search import DDGS
    except ImportError as e:
        raise RuntimeError("请安装 duckduckgo-search：pip install duckduckgo-search") from e

    region = settings.WEB_SEARCH_REGION or "cn-zh"
    out: list[dict] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=k, region=region):
                out.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": (r.get("body") or "")[:800],
                })
    except Exception as e:
        log.warning(f"DuckDuckGo 搜索失败: {e}")
        raise TimeoutError(f"DuckDuckGo 搜索失败: {e}") from e
    log.info(f"DuckDuckGo 返回 {len(out)} 条")
    return out


def _search_mock(query: str, k: int) -> list[dict]:
    return [
        {
            "title": f"[模拟] 关于「{query}」的结果 {i}",
            "url": f"https://example.com/mock/{i}",
            "snippet": f"这是演示用模拟摘要 #{i}，请配置真实搜索后端。",
        }
        for i in range(1, min(k, 3) + 1)
    ]
