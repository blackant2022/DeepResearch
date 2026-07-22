# DeepResearch — 学术文献多智能体系统

基于 **LangGraph** 的学术文献研究助手：本地 **RAG**、**Policy ReAct 工具调用**、**深度研究子图**、**双层置信度 / Grounding 幻觉治理**、**长期记忆**，以及可选 **MCP** 远程工具。支持 PDF / DOCX / TXT / MD 入库与 Streamlit 对话。

LLM 直连 [DeepSeek](https://platform.deepseek.com)（OpenAI 兼容协议）。向量库 **ChromaDB**；嵌入 **FastEmbed `BAAI/bge-small-zh-v1.5`**；重排默认 **`BAAI/bge-reranker-v2-m3`**。

给 AI 协作的约束见根目录 [`AGENTS.md`](./AGENTS.md)。

---

## 核心能力

| 能力 | 说明 |
|------|------|
| Workflow 分流 | 长期记忆召回 →（可选）Query 改写 → 多模态 → 路由 |
| 三路径 | `chat` 闲聊 / `policy` 文献与工具 / `deep_research` 综述管线 |
| Policy ReAct | **节点内**多轮工具调用（非整图 think↔tools 空转） |
| RAG | 结构化切块（摘要整块/表Markdown/参考文献处理）+ 语义滑窗、入库去重、宽召回、BGE Rerank |
| 幻觉治理 | 检索/回答双层门禁 + Critic Grounding（批量核验，默认关 per-claim） |
| 深度研究 | Planner → Researcher（可并行）→ Writer → Critic ↻ → Consolidate |
| 记忆 | 工作记忆（可序列化）+ LTM（跨会话；默认定稿后异步写入） |
| MCP | 可选：按 URL / stdio 接入远程工具（如 MCP Market 百度搜索） |
| 中间件 | 工具护栏、重试、降级 |

---

## 架构

```
用户问题
  │
  ▼
┌──────────── Workflow（确定性）────────────┐
│ LTM 召回 → Query 改写(可关) → 多模态 → Router │
└──────────────────┬───────────────────────┘
                   │
       ┌───────────┼───────────┐
       ▼           ▼           ▼
     chat       policy    deep_research
       │      （节点内 ReAct）   │
       │           │      plan → research → write
       │           │         → critic ↻ → consolidate
       └───────────┴───────────┘
                   ▼
              返回答案（+ 可选异步写 LTM）
```

**设计要点：** 同轮只走一条业务路径；ReAct / 修订环均有硬上限（`MAX_REACT_ITERATIONS`、`MAX_KNOWLEDGE_SEARCHES`、`MAX_REVISE` 等），避免多 Agent 指令冲突与无限循环。

---

## 技术栈

- Python 3.11+
- LangGraph 1.x（编排 + 检查点）
- ChromaDB + FastEmbed + transformers（BGE Reranker）
- OpenAI SDK v2 → DeepSeek
- Streamlit / Pydantic Settings
- 可选：`mcp[cli]`（MCP Client / 自建 Server）

---

## 快速开始

### 1. 克隆与依赖

```bash
git clone https://github.com/blackant2022/DeepResearch.git
cd DeepResearch
pip install -r requirements.txt
# 使用 MCP / BGE Reranker 时按需安装：
# pip install "mcp[cli]>=1.0.0" transformers torch
```

### 2. 配置（勿提交密钥）

```bash
cp .env.example .env
```

在 `.env` 填写：

```env
DEEPSEEK_API_KEY=你的真实Key
```

推送前建议：

```bash
python scripts/check_secrets.py
```

> 禁止把真实 Key 写进 `settings.py`、`.env.example`、README 或截图。

### 3. 文献入库

将文件放入 `data/docs/`：

```bash
python -m src.rag.ingest ./data/docs
```

入库支持**结构化切块**（摘要整块、表格转 Markdown、默认丢弃参考文献、过滤页眉页脚）以及**文件级 / 块级哈希去重**（同名覆盖更新）。

### 4. 运行

```bash
# CLI
python run_cli.py "知识库里有哪些关于高光谱的研究？"

# Web
streamlit run frontend/streamlit_app.py
```

浏览器：http://localhost:8501

### 5. 测试与评测

```bash
pytest tests/ -v

# 生成 / 使用约 2000 条代理测试集
python scripts/generate_rag_testset.py --n 2000
# Rerank A/B、门禁扫参等见 scripts/ 与 data/eval/
```

---

## 项目结构

```
deepresearch-agent/
├── AGENTS.md                 # AI / 协作者约束
├── config/settings.py        # 配置（读 .env）
├── frontend/streamlit_app.py
├── src/
│   ├── agents/               # router / policy / planner / researcher / writer / critic …
│   ├── orchestrator/         # graph.py + workflow.py
│   ├── rag/                  # ingest / chunking / retriever / rerank / grounding / confidence
│   ├── memory/               # 工作记忆 + LTM + embedding
│   ├── tools/                # 注册表与内置工具
│   ├── mcp/                  # MCP Server / Client（可选）
│   ├── middleware/
│   └── llm/
├── data/
│   ├── docs/                 # 原始文献（本地，通常不入库 Git）
│   ├── chroma/               # RAG 向量库
│   ├── ltm_chroma/           # 长期记忆
│   └── eval/                 # 评测集与报告
├── scripts/                  # 评测、扫参、生成测试集等
├── docs/                     # 如 MCP_REMOTE_SETUP.md
├── examples/                 # 示例 Skill 等（非运行时必选）
├── tests/
├── run_cli.py
├── run_mcp_server.py         # 对外暴露本仓库工具的 MCP Server
└── requirements.txt
```

---

## 常用环境变量

| 变量 | 默认（约） | 说明 |
|------|------------|------|
| `DEEPSEEK_API_KEY` | — | **必填** |
| `EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | 嵌入 |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | 重排 |
| `TOP_K` / `RETRIEVAL_RECALL_K` | `3` / `8` | 精排条数 / 宽召回 |
| `RETRIEVAL_THRESHOLD` | `0.55` | 检索门禁（vector_score） |
| `ANSWER_CONFIDENCE_THRESHOLD` | `0.45` | 回答门禁 |
| `FAST_MODE` | `true` | 快路径（路由与深研轻量策略） |
| `MAX_REACT_ITERATIONS` | `2` | Policy 最大轮次 |
| `QUERY_REWRITE_ENABLED` | `false` | Query 改写 |
| `MCP_CLIENT_ENABLED` | `false` | 是否注册远程 MCP |
| `MCP_SERVERS` | `[]` | JSON：`url` 或 `command`+`args` |

完整列表见 [`.env.example`](./.env.example)。远程 MCP 步骤见 [`docs/MCP_REMOTE_SETUP.md`](./docs/MCP_REMOTE_SETUP.md)。

---

## 内置工具

| 工具 | 功能 |
|------|------|
| `knowledge_search` | 本地知识库语义检索 |
| `kb_overview` | 知识库文档目录 |
| `calculator` | 安全表达式计算 |
| `web_search` | 联网（Tavily / DuckDuckGo·ddgs） |

扩展：在 `src/tools/` 注册，或开启 MCP Client 接入远程工具（自动下发参数 schema）。

---

## 设计说明

- **不走 LangChain ChatModel**：直接用 OpenAI SDK v2，减少版本摩擦。
- **工作记忆可序列化**：`wm_items` 兼容检查点。
- **置信度用 vector_score**：避免 Rerank CE 分虚高打穿拒答。
- **Grounding 控成本**：claim 封顶 + 批量核验；`GROUNDING_PER_CLAIM` 默认关闭。

---

## License

MIT
