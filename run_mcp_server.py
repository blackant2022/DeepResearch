"""
run_mcp_server.py — 启动 DeepResearch MCP Server（stdio）

用法（供 Cursor / Claude Desktop 等配置）：
  command: python
  args: ["D:/deepresearch-agent/run_mcp_server.py"]
"""
from src.mcp.server import main

if __name__ == "__main__":
    main()
