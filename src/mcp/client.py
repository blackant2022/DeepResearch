"""
src/mcp/client.py — MCP Client 桥接（可选依赖）

未安装 mcp 包时自动跳过，不影响主流程运行。
安装：pip install "mcp[cli]>=1.0.0"
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
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
        return data if isinstance(data, list) else []
    except json.JSONDecodeError as e:
        log.warning(f"MCP_SERVERS JSON 解析失败: {e}")
        return []


async def _list_remote_tools(spec: dict[str, Any]) -> list[dict[str, Any]]:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    command = spec.get("command")
    if not command:
        return []
    args = spec.get("args") or []
    env = spec.get("env")
    prefix = (spec.get("prefix") or "mcp").strip("_")
    params = StdioServerParameters(command=command, args=args, env=env)

    tools: list[dict[str, Any]] = []
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
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


async def _call_remote_tool(spec: dict[str, Any], remote_name: str, arguments: dict[str, Any]) -> Any:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=spec["command"],
        args=spec.get("args") or [],
        env=spec.get("env"),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(remote_name, arguments)
            chunks: list[str] = []
            for block in result.content:
                text = getattr(block, "text", None)
                if text:
                    chunks.append(text)
            return "\n".join(chunks) if chunks else str(result.content)


def _schema_from_json(input_schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    props = input_schema.get("properties") or {}
    required = set(input_schema.get("required") or [])
    out: dict[str, dict[str, Any]] = {}
    type_map = {"string": "str", "integer": "int", "number": "float", "boolean": "bool"}
    for pname, spec in props.items():
        if not isinstance(spec, dict):
            continue
        out[pname] = {
            "type": type_map.get(spec.get("type", "string"), "str"),
            "desc": spec.get("description", ""),
            "required": pname in required,
        }
    return out


class MCPTool(BaseTool):
    """将远程 MCP 工具包装为本地 BaseTool。"""

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
        self.schema = _schema_from_json(input_schema)

    def _run(self, **kwargs: Any) -> Any:
        return asyncio.run(_call_remote_tool(self._server_spec, self._remote_name, kwargs))


def register_mcp_tools() -> int:
    """发现并注册外部 MCP 工具，返回新增数量。未安装 mcp 时安全跳过。"""
    if not settings.MCP_CLIENT_ENABLED:
        return 0
    if not mcp_available():
        log.info("未安装 mcp 包，跳过 MCP Client（可选：pip install \"mcp[cli]>=1.0.0\"）")
        return 0
    specs = _parse_server_specs()
    if not specs:
        return 0

    added = 0
    for spec in specs:
        try:
            remote_tools = asyncio.run(_list_remote_tools(spec))
        except Exception as e:
            log.warning(f"连接 MCP Server 失败 ({spec.get('command')}): {e}")
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
            log.info(f"已注册 MCP 工具: {name} ← {meta['remote_name']}")
    return added
