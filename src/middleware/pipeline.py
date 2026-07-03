"""
src/middleware/pipeline.py — 工具调用中间件管线

【为什么要中间件？】把横切关注点（日志、重试、护栏、限流）从工具逻辑里剥离，
所有工具调用统一走管线，做到“一处实现，处处生效”。这也是 2026 年 Agent 框架
（LangGraph / OpenAI Agents SDK）都强调的 middleware 思想。

管线顺序（洋葱模型）：
    Guardrail(入参护栏) → Logging → Retry(含指数退避) → 真正调用 → Logging(出参)

【09｜工具调用失败时应该如何处理？】本管线给出可落地策略：
    - invalid_args：不重试，直接返回错误让 Agent 修正参数
    - timeout / upstream / runtime：可重试，指数退避，最多 MAX_TOOL_RETRY 次
    - 重试仍失败：返回结构化失败（降级），由 Agent 决定换工具或如实告知“无法获取”
"""
from __future__ import annotations

import time
from typing import Callable

from config.settings import settings
from src.tools.base import BaseTool, ToolResult
from src.utils.logger import get_logger

log = get_logger("middleware")

# 哪些错误类型允许重试
_RETRYABLE = {"timeout", "upstream", "runtime"}
# 简单的输入护栏：命中即拒绝（防注入/危险意图的最小演示）
_BLOCKLIST = ("rm -rf", "drop table", "delete from", "__import__", "os.system")


class MiddlewarePipeline:
    def __init__(self) -> None:
        self._call_count = 0

    def guardrail(self, tool: BaseTool, kwargs: dict) -> str | None:
        joined = " ".join(str(v).lower() for v in kwargs.values())
        for bad in _BLOCKLIST:
            if bad in joined:
                return f"护栏拦截：检测到危险内容「{bad}」"
        return None

    def invoke(self, tool: BaseTool, **kwargs) -> ToolResult:
        """通过中间件管线执行一次工具调用。"""
        self._call_count += 1

        # 1) 护栏
        blocked = self.guardrail(tool, kwargs)
        if blocked:
            log.warning(blocked)
            return ToolResult(ok=False, error=blocked, error_type="guardrail", tool=tool.name)

        # 2) 重试 + 指数退避
        attempt = 0
        last: ToolResult | None = None
        while attempt < settings.MAX_TOOL_RETRY:
            attempt += 1
            log.info(f"调用工具 {tool.name} (第{attempt}次) args={kwargs}")
            result = tool(**kwargs)  # BaseTool.__call__ 已兜住异常，返回 ToolResult
            result.meta["attempts"] = attempt

            if result.ok:
                log.info(f"工具 {tool.name} 成功，耗时 {result.latency_ms}ms")
                return result

            last = result
            # 不可重试错误：立即返回
            if result.error_type not in _RETRYABLE:
                log.warning(f"工具 {tool.name} 失败(不可重试/{result.error_type}): {result.error}")
                return result
            # 可重试：退避后再来
            backoff = 0.4 * (2 ** (attempt - 1))
            log.warning(f"工具 {tool.name} 失败({result.error_type})，{backoff:.1f}s 后重试: {result.error}")
            time.sleep(backoff)

        log.error(f"工具 {tool.name} 重试 {settings.MAX_TOOL_RETRY} 次仍失败，触发降级")
        if last is not None:
            last.meta["degraded"] = True
        return last or ToolResult(ok=False, error="未知失败", tool=tool.name)

    @property
    def call_count(self) -> int:
        return self._call_count


pipeline = MiddlewarePipeline()
