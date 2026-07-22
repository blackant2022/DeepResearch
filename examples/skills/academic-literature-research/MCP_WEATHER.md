# MCP 接入外部天气查询（说明稿 · 未改运行时默认）

> 配合 `academic-literature-research` Skill：天气属域外；只有用户明确要查天气时，才走 MCP/联网工具。

## 思路

DeepResearch 已有 **MCP Client**：把外部 MCP Server 的 tools 注册进本地 `registry`，Policy 就能像调 `knowledge_search` 一样调 `weather_xxx`。

```
外部天气 MCP Server  --stdio--> 本项目 MCP Client  --> registry
                                                      --> Policy / 工具调用
```

## 方式 A：接现成的天气 MCP（推荐理解路径）

1. 安装 MCP SDK（若尚未安装）：

```bash
pip install "mcp[cli]>=1.0.0"
```

2. 准备一个提供 `get_weather` / `weather` 一类工具的 MCP Server  
   （社区常见：包一层 OpenWeather / 和风 / wttr.in 的 stdio 服务。你自己写也可以，见方式 B。）

3. 在 `.env` 中打开 Client 并声明 Server，例如：

```env
MCP_CLIENT_ENABLED=true
MCP_SERVERS=[{"command":"python","args":["D:/path/to/weather_mcp_server.py"],"prefix":"weather"}]
```

- `command` + `args`：怎么启动那个 Server（stdio）
- `prefix`：本地工具名前缀，避免撞名；远程 `get_weather` → 本地 `weather_get_weather`

4. 启动主程序；日志里应出现：`已注册 MCP 工具: weather_...`

5. 提问：「用天气工具查一下北京今天天气」  
   Policy 在工具列表里看到 `[MCP] ...` 描述后即可调用。

`.env.example` 里已有同类示例（指向本仓库自己的 `run_mcp_server.py`）。

## 方式 B：自己写一个最小天气 MCP Server

新建例如 `examples/mcp_weather_server.py`（示意）：

```python
from mcp.server.fastmcp import FastMCP
import urllib.request

mcp = FastMCP("WeatherDemo")

@mcp.tool()
def get_weather(city: str) -> dict:
    """查询城市当前天气（演示：可用 wttr.in 等公开接口）。"""
    # 实际项目请换正式 API Key，并处理错误/超时
    url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1"
    ...
    return {"city": city, "summary": "...", "temp_c": ...}

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

然后：

```env
MCP_CLIENT_ENABLED=true
MCP_SERVERS=[{"command":"python","args":["D:/deepresearch-agent/examples/mcp_weather_server.py"],"prefix":"weather"}]
```

## 方式 C：不经过 MCP，直接做内置 Tool

若只有你自己用、不需要给 Cursor 等客户端共用：在 `src/tools/` 写 `weather_search` 并 `registry.register` 即可，**不必上 MCP**。

MCP 的价值是：**标准协议、可插拔、可被多个 Client 复用**。

## 和科研 Skill 的配合

| 用户问法 | Skill 判定 | 工具 |
|----------|------------|------|
| 高光谱估氮方法 | 文献问答 | `knowledge_search` |
| 今天北京天气 | 域外 | 默认拒答 / 提示无关 |
| 请调用天气工具查北京 | 域外但明确要工具 | `weather_get_weather`（MCP） |

不要把天气 MCP 默认开进「文献综述」路径，否则路由会脏。

## 排查

- Client 开了但没工具：检查 `MCP_SERVERS` JSON、路径、Server 是否能单独 `python xxx.py` 跑通
- 未装 `mcp`：Client 会安全跳过，主流程仍可用
- 工具名：看日志里的 `weather_*` 注册名，提问时用自然描述即可（靠 description）
