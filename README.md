# DeepResearch — 学术文献超级智能体

基于 **LangGraph** 的多 Agent 学术文献研究系统：集成 **RAG 检索增强**、**ReAct 工具调用**、**深度研究子图**、**幻觉治理**与**长期记忆**，支持 PDF / Word 文献入库与交互式问答。

LLM 直连 [DeepSeek](https://platform.deepseek.com)（OpenAI 兼容协议），向量库使用 **ChromaDB**，嵌入模型 **FastEmbed（BGE 中文）**。

---

## 核心能力

| 能力 | 说明 |
|------|------|
| 超级智能体 ReAct | LLM 自主决定何时调用知识库检索、目录概览、计算器等工具 |
| 智能路由 | 自动分流：日常对话 / 超级智能体 / 深度研究 |
| RAG 知识库 | PDF、DOCX、TXT、Markdown 入库、切分、向量化、语义检索 |
| 深度研究链 | Planner → Researcher → Writer → Critic 多 Agent 协作 |
| 幻觉治理 | 生成前约束 + Grounding 事实核验 + 闭环修订 |
| 双层记忆 | 工作记忆（任务内）+ 长期记忆（Chroma 跨会话） |
| 工具中间件 | 护栏、指数退避重试、降级 |

---

## 架构

```
用户问题
  │
  ▼
recall_memory（长期记忆召回）
  │
  ▼
router（意图路由）
  ├─ chat            日常对话（快速路径）
  ├─ super_think     超级智能体 ReAct 环
  │     ↺ super_tools → super_think（自主调工具）
  └─ deep_research   深度研究子图
        plan → research → write → critic ↺ → consolidate
```

---

## 技术栈

- **Python 3.11+**
- **LangGraph 1.x** — 多 Agent 图编排 + 检查点
- **ChromaDB** — 向量存储（RAG + 长期记忆）
- **FastEmbed** — 本地 ONNX 嵌入（`BAAI/bge-small-zh-v1.5`）
- **OpenAI SDK v2** — 直连 DeepSeek，支持 Function Calling
- **Streamlit** — 对话前端
- **Pydantic Settings** — 配置管理

---

## 快速开始

### 1. 克隆 & 安装依赖

```bash
git clone https://github.com/blackant2022/DeepResearch.git
cd DeepResearch
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
```

编辑 `.env`，填入 DeepSeek API Key（[免费注册](https://platform.deepseek.com)）：

```env
DEEPSEEK_API_KEY=sk-your-key-here
```

### 3. 准备文献 & 构建知识库

将 PDF / Word / TXT 放入 `data/docs/`，然后：

```bash
python -m src.rag.ingest ./data/docs
```

### 4. 运行

**命令行：**

```bash
python run_cli.py "知识库里有哪些关于高光谱的研究？"
```

**Web 界面：**

```bash
streamlit run frontend/streamlit_app.py
```

浏览器打开 http://localhost:8501

### 5. 测试

```bash
pytest tests/ -v
```

---

## 项目结构

```
deepresearch-agent/
├── config/settings.py          # 全局配置（.env）
├── frontend/streamlit_app.py   # Streamlit 前端
├── src/
│   ├── agents/                 # 各类 Agent（router / super / planner …）
│   ├── orchestrator/graph.py   # LangGraph 主编排
│   ├── rag/                    # 入库、检索、幻觉治理
│   ├── memory/                 # 工作记忆 + 长期记忆
│   ├── tools/                  # 工具注册表 & 内置工具
│   ├── middleware/             # 工具调用中间件
│   └── llm/provider.py         # DeepSeek LLM 入口
├── data/
│   ├── docs/                   # 原始文献（本地）
│   ├── chroma/                 # RAG 向量库（本地生成）
│   └── ltm_chroma/             # 长期记忆向量库
├── tests/                      # 冒烟测试
├── run_cli.py                  # CLI 入口
└── requirements.txt
```

---

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DEEPSEEK_API_KEY` | — | **必填**，DeepSeek API 密钥 |
| `LLM_MODEL` | `deepseek-chat` | 模型名称 |
| `EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | 嵌入模型 |
| `MAX_REACT_ITERATIONS` | `8` | 超级智能体最大 ReAct 轮次 |
| `FAST_MODE` | `true` | 关闭后复杂问题走深度研究链 |
| `TOP_K` | `3` | RAG 检索条数 |
| `WEB_SEARCH_PROVIDER` | `auto` | 联网搜索：`auto` / `tavily` / `duckduckgo` / `mock` |
| `TAVILY_API_KEY` | — | Tavily API Key（可选，[tavily.com](https://tavily.com)） |

完整列表见 `.env.example`。

---

## 内置工具

| 工具 | 功能 |
|------|------|
| `knowledge_search` | 本地知识库语义检索 |
| `kb_overview` | 列出知识库全部文档 |
| `calculator` | 安全数学表达式计算 |
| `web_search` | 联网搜索（Tavily / DuckDuckGo） |

扩展方式：在 `src/tools/builtin_tools.py` 注册新工具，超级智能体自动获得调用能力。

---

## 部署到公网

推荐使用 **Docker + 云服务器**，需挂载 `data/` 目录以持久化知识库。详见项目 Issues 或自行配置：

```bash
streamlit run frontend/streamlit_app.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.headless true
```

> 公网部署务必做好访问鉴权，避免 API Key 被滥用。

---

## 设计说明

- **不走 LangChain ChatModel**：避免 langchain + openai 2.x 的 `proxies` 参数兼容问题，直接使用 OpenAI SDK v2。
- **工作记忆可序列化**：`wm_items` 替代不可序列化的 `WorkingMemory` 对象，兼容 LangGraph 检查点。
- **工具调用走中间件**：重试、护栏、降级与业务逻辑解耦。

---

## License

MIT
