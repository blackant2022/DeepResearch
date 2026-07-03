"""
src/tools/base.py — Agent 工具系统（注册表 + Schema + 统一调用协议）

【如何设计 Agent 的工具系统？需要考虑哪些因素？】本文件给出工程化答案：

  1. 统一契约：每个工具都有 name / description / 参数 schema / run()，
     让 LLM 能“看懂”工具（description 即给模型的说明书）。
  2. 可发现：ToolRegistry 统一注册，向 LLM 暴露工具清单（function-calling 风格）。
  3. 安全与幂等：参数经 pydantic 校验；工具应无副作用或副作用可控。
  4. 可观测：每次调用产出结构化 ToolResult（成功/失败/耗时/错误类型）。
  5. 容错：调用失败不抛穿到 Agent，而是返回 ToolResult(ok=False)，由中间件决定重试/降级。

工具本身只管“做事”，重试、日志、护栏交给 middleware，实现关注点分离。
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    ok: bool
    output: Any = None
    error: str = ""
    error_type: str = ""       # 供中间件判断是否可重试：timeout / invalid_args / upstream ...
    latency_ms: float = 0.0
    tool: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):
    name: str = "base"
    description: str = ""
    # 参数 schema：{参数名: {"type","desc","required"}}，用于告诉 LLM 怎么调用
    schema: dict[str, dict[str, Any]] = {}

    @abstractmethod
    def _run(self, **kwargs: Any) -> Any:
        """真正干活的地方，子类实现；可自由抛异常，由 __call__ 统一兜住。"""

    def validate(self, kwargs: dict[str, Any]) -> str | None:
        """极简参数校验：检查必填项。返回错误信息或 None。"""
        for pname, spec in self.schema.items():
            if spec.get("required") and pname not in kwargs:
                return f"缺少必填参数: {pname}"
        return None

    def to_openai_schema(self) -> dict[str, Any]:
        """转为 OpenAI function calling JSON schema。"""
        type_map = {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}
        properties: dict[str, Any] = {}
        required: list[str] = []
        for pname, spec in self.schema.items():
            properties[pname] = {
                "type": type_map.get(spec.get("type", "str"), "string"),
                "description": spec.get("desc", ""),
            }
            if spec.get("required"):
                required.append(pname)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": properties, "required": required},
            },
        }

    def __call__(self, **kwargs: Any) -> ToolResult:
        t0 = time.time()
        err = self.validate(kwargs)
        if err:
            return ToolResult(ok=False, error=err, error_type="invalid_args", tool=self.name)
        try:
            out = self._run(**kwargs)
            return ToolResult(
                ok=True, output=out, tool=self.name, latency_ms=round((time.time() - t0) * 1000, 1)
            )
        except TimeoutError as e:
            return ToolResult(ok=False, error=str(e), error_type="timeout", tool=self.name)
        except Exception as e:  # noqa: BLE001
            return ToolResult(
                ok=False, error=str(e), error_type="runtime",
                tool=self.name, latency_ms=round((time.time() - t0) * 1000, 1)
            )


class ToolRegistry:
    """全局工具注册表：Agent 通过它发现和获取工具。"""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def all(self) -> list[BaseTool]:
        return list(self._tools.values())

    def manifest(self) -> str:
        """给 LLM 看的工具清单文本（function-calling 风格的自然语言版）。"""
        lines = []
        for t in self._tools.values():
            params = ", ".join(
                f"{k}:{v.get('type','str')}{'*' if v.get('required') else ''}"
                for k, v in t.schema.items()
            )
            lines.append(f"- {t.name}({params}): {t.description}")
        return "\n".join(lines)

    def openai_schemas(self) -> list[dict[str, Any]]:
        return [t.to_openai_schema() for t in self._tools.values()]


registry = ToolRegistry()
