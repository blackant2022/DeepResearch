"""
src/mcp/client.py — MCP Client 桥接（可选依赖）

支持两类 Server：
  1) 远程 URL（MCP Market 等）：{"url": "https://.../mcp/...", "prefix": "baidu"}
  2) 本地 stdio 进程：{"command": "python", "args": ["run_mcp_server.py"], "prefix": "dr"}

未安装 mcp 包时自动跳过。安装：pip install "mcp[cli]>=1.0.0"
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from config.settings import settings
from src.tools.base import BaseTool, registry
from src.utils.logger import get_logger

log = get_logger("mcp.client")

_registered_mcp: set[str] = set()


def mcp_available() -> bool:
    return importlib.util.find_spec("mcp") is not None


def _parse_server_specs() -> list[dict[str, Any]]:
    raw = (settings.MCP_SERVERS or "").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
        if not isinstance(data, list):
            log.warning("MCP_SERVERS 必须是 JSON 数组")
            return []
        return [s for s in data if isinstance(s, dict)]
    except json.JSONDecodeError as e:
        log.warning(f"MCP_SERVERS JSON 解析失败: {e}")
        return []


def _prefix_of(spec: dict[str, Any]) -> str:
    return (spec.get("prefix") or "mcp").strip().strip("_")


def _headers_of(spec: dict[str, Any]) -> dict[str, str]:
    headers = {str(k): str(v) for k, v in (spec.get("headers") or {}).items()}
    token = spec.get("token") or spec.get("api_key") or spec.get("auth_token")
    if token and "Authorization" not in headers:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _transport_kind(spec: dict[str, Any]) -> str:
    if spec.get("url") or spec.get("endpoint"):
        return "url"
    if spec.get("command"):
        return "stdio"
    return "unknown"


async def _connect(spec: dict[str, Any], stack: AsyncExitStack):
    """在给定 AsyncExitStack 上建立并 initialize ClientSession。"""
    from mcp import ClientSession, StdioServerParameters

    kind = _transport_kind(spec)
    if kind == "stdio":
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=spec["command"],
            args=spec.get("args") or [],
            env=spec.get("env"),
        )
        read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session

    if kind != "url":
        raise ValueError("MCP Server 配置需要 url 或 command 字段")

    url = str(spec.get("url") or spec.get("endpoint") or "").strip()
    if not url:
        raise ValueError("url 为空")
    headers = _headers_of(spec)
    prefer = (spec.get("transport") or "auto").lower()
    errors: list[str] = []

    async def via_http():
        from mcp.client.streamable_http import streamable_http_client
        from mcp.shared._httpx_utils import create_mcp_http_client

        http_client = create_mcp_http_client(headers=headers or None)
        read, write, _sid = await stack.enter_async_context(
            streamable_http_client(url, http_client=http_client)
        )
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session

    async def via_sse():
        from mcp.client.sse import sse_client

        read, write = await stack.enter_async_context(
            sse_client(url, headers=headers or None)
        )
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session

    if prefer in ("auto", "http", "streamable", "streamable-http", "streamable_http"):
        try:
            return await via_http()
        except Exception as e:  # noqa: BLE001
            errors.append(f"streamable-http: {e}")
            if prefer != "auto":
                raise

    if prefer in ("auto", "sse"):
        try:
            # auto 下 HTTP 已失败：必须用全新 stack，由调用方重建
            if errors:
                raise _NeedFreshStack("; ".join(errors))
            return await via_sse()
        except _NeedFreshStack:
            raise
        except Exception as e:  # noqa: BLE001
            errors.append(f"sse: {e}")
            raise RuntimeError("无法连接远程 MCP：" + " | ".join(errors)) from e

    raise RuntimeError("无法连接远程 MCP：" + (" | ".join(errors) if errors else "未知错误"))


class _NeedFreshStack(Exception):
    """HTTP 失败后提示调用方换新的 ExitStack 再试 SSE。"""


@asynccontextmanager
async def _open_session(spec: dict[str, Any]):
    stack = AsyncExitStack()
    await stack.__aenter__()
    try:
        try:
            session = await _connect(spec, stack)
        except _NeedFreshStack as first:
            await stack.__aexit__(None, None, None)
            stack = AsyncExitStack()
            await stack.__aenter__()
            # 强制走 SSE
            spec_sse = dict(spec)
            spec_sse["transport"] = "sse"
            try:
                session = await _connect(spec_sse, stack)
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(f"无法连接远程 MCP：{first} | sse: {e}") from e
        yield session
    finally:
        await stack.__aexit__(None, None, None)


def _tools_from_listed(listed: Any, spec: dict[str, Any]) -> list[dict[str, Any]]:
    prefix = _prefix_of(spec)
    tools: list[dict[str, Any]] = []
    for t in listed.tools:
        local_name = f"{prefix}_{t.name}" if prefix else t.name
        tools.append({
            "local_name": local_name,
            "remote_name": t.name,
            "description": t.description or t.name,
            "input_schema": t.inputSchema or {"type": "object", "properties": {}},
            "spec": spec,
        })
    return tools


async def _list_remote_tools(spec: dict[str, Any]) -> list[dict[str, Any]]:
    async with _open_session(spec) as session:
        listed = await session.list_tools()
        return _tools_from_listed(listed, spec)


async def _call_remote_tool(spec: dict[str, Any], remote_name: str, arguments: dict[str, Any]) -> Any:
    async with _open_session(spec) as session:
        result = await session.call_tool(remote_name, arguments or {})
        chunks: list[str] = []
        for block in result.content:
            text = getattr(block, "text", None)
            if text:
                chunks.append(text)
            else:
                data = getattr(block, "data", None)
                if data is not None:
                    chunks.append(json.dumps(data, ensure_ascii=False))
        return "\n".join(chunks) if chunks else str(result.content)


def _schema_from_json(input_schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    props = input_schema.get("properties") or {}
    required = set(input_schema.get("required") or [])
    out: dict[str, dict[str, Any]] = {}
    type_map = {
        "string": "str",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
        "array": "list",
        "object": "dict",
    }
    for pname, pspec in props.items():
        if not isinstance(pspec, dict):
            continue
        out[pname] = {
            "type": type_map.get(pspec.get("type", "string"), "str"),
            "desc": pspec.get("description", ""),
            "required": pname in required,
        }
    return out


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class MCPTool(BaseTool):
    """将远程 MCP 工具包装为本地 BaseTool。参数由对方 inputSchema 自动映射。"""

    def __init__(
        self,
        *,
        local_name: str,
        remote_name: str,
        description: str,
        input_schema: dict[str, Any],
        server_spec: dict[str, Any],
    ) -> None:
        self.name = local_name
        self.description = f"[MCP] {description}"
        self._remote_name = remote_name
        self._server_spec = server_spec
        self._input_schema = input_schema
        self.schema = _schema_from_json(input_schema)

    def _run(self, **kwargs: Any) -> Any:
        allowed = set(self.schema.keys()) if self.schema else set(kwargs.keys())
        args = {k: v for k, v in kwargs.items() if k in allowed}
        return _run_async(_call_remote_tool(self._server_spec, self._remote_name, args))


def register_mcp_tools() -> int:
    """发现并注册外部 MCP 工具，返回新增数量。"""
    if not settings.MCP_CLIENT_ENABLED:
        return 0
    if not mcp_available():
        log.info("未安装 mcp 包，跳过 MCP Client（可选：pip install \"mcp[cli]>=1.0.0\"）")
        return 0
    specs = _parse_server_specs()
    if not specs:
        log.info("MCP_CLIENT_ENABLED 但 MCP_SERVERS 为空，跳过")
        return 0

    added = 0
    for spec in specs:
        label = spec.get("url") or spec.get("endpoint") or spec.get("command") or "?"
        try:
            remote_tools = _run_async(_list_remote_tools(spec))
        except Exception as e:  # noqa: BLE001
            log.warning(f"连接 MCP Server 失败 ({label}): {e}")
            continue
        for meta in remote_tools:
            name = meta["local_name"]
            if name in _registered_mcp or registry.get(name) is not None:
                continue
            registry.register(MCPTool(
                local_name=name,
                remote_name=meta["remote_name"],
                description=meta["description"],
                input_schema=meta["input_schema"],
                server_spec=meta["spec"],
            ))
            _registered_mcp.add(name)
            added += 1
            params = ", ".join(
                f"{k}{'*' if v.get('required') else ''}:{v.get('type')}"
                for k, v in _schema_from_json(meta["input_schema"]).items()
            ) or "(无参数)"
            log.info(f"已注册 MCP 工具: {name} ← {meta['remote_name']}  params=[{params}]")
    return added


def describe_registered_mcp_tools() -> list[dict[str, Any]]:
    """调试用：列出已注册的 MCP 工具及参数。"""
    out = []
    for name in sorted(_registered_mcp):
        tool = registry.get(name)
        if tool is None:
            continue
        out.append({
            "name": name,
            "description": tool.description,
            "parameters": getattr(tool, "schema", {}),
        })
    return out
