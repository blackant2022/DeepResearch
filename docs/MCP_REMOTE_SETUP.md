# 远程 / 本地 MCP 完整接入流程（DeepResearch）

## 一、你要配什么？

**参数不用你在 `.env` 里手写。**  
连上 MCP Server 后，Client 会 `list_tools`，自动拿到每个工具的：

- 名称（如 `baidu_web_search`）
- 描述（给 LLM 看，决定何时调用）
- **inputSchema**（`query` / `count` / `freshness` 等）→ 转成本地 `BaseTool.schema`

Agent 调工具时，由 **Function Calling 自动填参数**；你只需在对话里说清楚需求。

你要配置的只有：

1. 打开 Client  
2. 填 Server 的 **URL**（或本地 command）  
3. 可选：前缀、Token、请求头  

---

## 二、百度搜索（MCP Market）完整步骤

### 1. 安装依赖

```bash
pip install "mcp[cli]>=1.0.0"
```

### 2. 编辑 `.env`

```env
MCP_CLIENT_ENABLED=true
MCP_SERVERS=[{"url":"https://mcpmarket.cn/mcp/f06475ef3cc0f385b935c5d9","prefix":"baidu","transport":"auto"}]
```

说明：

| 字段 | 必填 | 含义 |
|------|------|------|
| `url` | 是* | 页面「连接信息」里的 MCP URL |
| `prefix` | 建议 | 本地名前缀，避免撞名 → `baidu_baidu_web_search` |
| `transport` | 否 | `auto`（先 HTTP 再 SSE）/ `http` / `sse` |
| `token` / `api_key` | 看平台 | 若 Market 要求鉴权，填 Bearer Token |
| `headers` | 否 | 额外 HTTP 头，如 `{"X-Api-Key":"..."}` |

\* 本地进程型 Server 用 `command`+`args`，不用 `url`。

### 3. 重启应用

启动 Streamlit / `run_agent` 时日志应出现类似：

```text
已注册 MCP 工具: baidu_baidu_web_search ← baidu_web_search  params=[query*:str, count:int, freshness:str]
```

### 4. 对话里怎么用（参数谁填？）

直接说自然语言即可，例如：

> 用百度搜索查一下「高光谱 氮含量 2024」，只要最近一年的结果，返回 5 条。

模型会大致调用：

```json
{
  "name": "baidu_baidu_web_search",
  "arguments": {
    "query": "高光谱 氮含量 2024",
    "count": 5,
    "freshness": "py"
  }
}
```

这些字段来自对方工具文档（你截图里的 `query` / `count` / `freshness`），**不是**你在 `.env` 里配死的。

### 5. 自测（可选）

```powershell
$env:PYTHONPATH="D:\deepresearch-agent"
py -c "from src.mcp.client import register_mcp_tools, describe_registered_mcp_tools; print(register_mcp_tools()); print(describe_registered_mcp_tools())"
```

---

## 三、配置模板

### 仅 URL（最常见）

```json
[{"url":"https://mcpmarket.cn/mcp/f06475ef3cc0f385b935c5d9","prefix":"baidu"}]
```

### 需要 Token

```json
[{"url":"https://mcpmarket.cn/mcp/...","prefix":"baidu","token":"你的密钥"}]
```

### 自定义 Header

```json
[{"url":"https://...","prefix":"baidu","headers":{"X-Api-Key":"xxx"}}]
```

### 强制 SSE（HTTP 不通时）

```json
[{"url":"https://...","prefix":"baidu","transport":"sse"}]
```

### 本地 stdio（自己的 Server）

```json
[{"command":"python","args":["D:/deepresearch-agent/run_mcp_server.py"],"prefix":"dr"}]
```

### 多个 Server

```json
[
  {"url":"https://mcpmarket.cn/mcp/百度搜索id","prefix":"baidu"},
  {"url":"https://mcpmarket.cn/mcp/文献评判id","prefix":"lit"}
]
```

写入 `.env` 时必须是**一行合法 JSON**（或注意转义）。PowerShell 里也可用：

```powershell
$env:MCP_CLIENT_ENABLED="true"
$env:MCP_SERVERS='[{"url":"https://mcpmarket.cn/mcp/f06475ef3cc0f385b935c5d9","prefix":"baidu"}]'
```

---

## 四、和 Cursor 安装命令的关系

页面上的：

```bash
uvx mcpstore-cli install "https://..." "百度搜索" --client cursor
```

只给 **Cursor IDE** 写配置，**不会**自动进 DeepResearch。  
在本项目里请用上面的 `MCP_SERVERS` 方式。

---

## 五、排错

| 现象 | 处理 |
|------|------|
| 没有任何 MCP 工具 | 确认 `MCP_CLIENT_ENABLED=true` 且 JSON 合法 |
| 连接失败 | 试 `"transport":"sse"`；检查 URL 是否完整；是否要 token |
| 有工具但不调用 | 提问里点名「用百度搜索」；看工具 description 是否注册成功 |
| 参数报错 | 看日志里的 `params=[...]`；必填项（带 `*`）必须由模型或你手动测通 |

---

## 六、流程小结

```text
MCP Market 复制 URL
    → .env: MCP_CLIENT_ENABLED + MCP_SERVERS=[{url, prefix}]
    → 启动 Agent → list_tools 自动导入参数 schema
    → 用户自然语言提问
    → Policy Function Calling 填 query/count/...
    → Client 按 URL 调用远端 baidu_web_search
```
