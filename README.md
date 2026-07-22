# DeepResearch — 学术文献多智能体研究系统

面向**本地文献知识库**的 Deep Research 助手：用 **LangGraph** 做有界编排，用 **RAG + Grounding** 约束幻觉，用可选 **MCP** 接入外部工具。目标不是堆 Agent 数量，而是让回答**可追溯、有上限、缺证据可拒答**。

| 项 | 内容 |
|----|------|
| 仓库 | https://github.com/blackant2022/DeepResearch |
| LLM | DeepSeek（OpenAI 兼容 SDK，非 LangChain ChatModel） |
| 向量库 | ChromaDB |
| 嵌入 | FastEmbed `BAAI/bge-small-zh-v1.5` |
| 重排 | `BAAI/bge-reranker-v2-m3`（transformers 直载） |
| 前端 | Streamlit |
| AI 协作约束 | 必读 [`AGENTS.md`](./AGENTS.md) |

**一句话定位：** Workflow 确定性分流 + Policy 节点内 ReAct + 深研 Plan→Research→Write→Critic 管线；证据不足时提示上传文献，而不是编造综述。

---

## 核心能力（与当前代码一致）

| 能力 | 说明 |
|------|------|
| Workflow | LTM 召回 →（可选）Query 改写 → 多模态增强 → Router |
| 三路径互斥 | `chat` / `policy` / `deep_research`，**同轮只走一条** |
| Policy ReAct | **节点内**工具循环；上限 `MAX_REACT_ITERATIONS`、`MAX_KNOWLEDGE_SEARCHES` |
| 深研子图 | Planner → Researcher（可并行）→ Writer ⇄ Critic → Consolidate **或** Safe Fallback |
| RAG 入库 | 结构化切块（摘要整块 / 表转 Markdown / 参考文献默认丢弃 / 垃圾行过滤）+ 语义断点 + 滑窗；文件/块哈希去重 |
| 检索 | 领域词增强 → 宽召回 → BGE Rerank → `weight` 降权 → 双层置信度门禁（`vector_score`） |
| Grounding | Critic 批量拆论断核验；**纯拒答/无论断不得虚报 100%**；低支撑提示上传 PDF/数据 |
| 记忆 | 短期 Working Memory（`wm_items`）+ 长期 LTM（Chroma）；失败兜底**不写** LTM |
| Checkpoint | 主图 `MemorySaver` + `thread_id`；每轮 `Overwrite` 清空 messages/trace/wm |
| MCP | 可选 URL / stdio；工具 schema 远端下发 |
| 中间件 | 工具护栏、可重试错误指数退避、结构化降级 |

---

## 架构

```
用户问题 (+ 可选附件)
        │
        ▼
┌───────────── Workflow（确定性）─────────────┐
│ LTM.recall → QueryRewrite? → Vision? → Router │
└──────────────────────┬──────────────────────┘
                       │ 唯一路径
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
       chat         policy      deep_research
         │        节点内 ReAct         │
         │             │         plan → research
         │             │              → write ⇄ critic
         │             │         ┌────┴────┐
         │             │      pass      fallback
         │             │         ▼         ▼
         │             │   consolidate  safe_fallback
         │             │   (异步 LTM)   (提示上传/拒答)
         └─────────────┴─────────┴─────────┘
                       ▼
                  final_answer
```

**触发深研（`FAST_MODE=true` 时）：** 问题含「全面综述 / 系统梳理 / 综合分析 / 深入分析 / 多角度 / 方法与挑战」等，且足够长（强意图词约 ≥40 字）。短问默认走 `policy`。

---

## 技术栈

- Python 3.11+
- LangGraph 1.x + MemorySaver
- ChromaDB + FastEmbed + transformers/torch（BGE Reranker）
- OpenAI SDK v2 → DeepSeek
- Streamlit / Pydantic Settings / tenacity 风格自研中间件
- 可选 `mcp[cli]`

---

## 快速开始

### 1. 克隆与依赖

```bash
git clone https://github.com/blackant2022/DeepResearch.git
cd DeepResearch
pip install -r requirements.txt
# Rerank / MCP 按需：
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

```bash
python scripts/check_secrets.py
```

> 禁止把真实 Key 写进 `settings.py`、`.env.example`、README、测试或截图。

### 3. 文献入库

将 PDF / DOCX / TXT / MD 放入 `data/docs/`：

```bash
python -m src.rag.ingest ./data/docs
```

**切块策略变更或升级代码后请重新入库**，否则旧块缺少 `section` / `weight` 等 metadata。

### 4. 运行

```bash
python run_cli.py "知识库里有哪些关于高光谱的研究？"
streamlit run frontend/streamlit_app.py
```

浏览器：http://localhost:8501

### 5. 测试与评测

```bash
pytest tests/ -v
python scripts/generate_rag_testset.py --n 2000
```

Rerank A/B、门禁扫参等见 `scripts/`、`data/eval/`。无段落金标时**不要宣称正式 Recall@K / nDCG / RAGAS**。

---

## 项目结构

```
DeepResearch/
├── AGENTS.md                 # 给人类与 AI Agent 的硬约束（必读）
├── README.md
├── .env.example
├── config/settings.py
├── frontend/streamlit_app.py
├── src/
│   ├── agents/               # router / policy / planner / researcher / writer / critic / chat
│   ├── orchestrator/         # graph.py（主图+深研子图）+ workflow.py
│   ├── rag/                  # ingest / structure_chunk / chunking / retriever / rerank /
│   │                         # grounding / confidence
│   ├── memory/               # working_memory + long_term_memory + embedding + context_manager
│   ├── tools/                # 注册表与内置工具
│   ├── mcp/                  # Client（URL/stdio）+ 可选 Server
│   ├── middleware/           # 护栏 / 重试
│   ├── planning/             # planner + evaluator
│   └── llm/
├── data/docs|chroma|ltm_chroma|eval/
├── scripts/                  # 评测、扫参、测试集生成
├── docs/MCP_REMOTE_SETUP.md
├── examples/skills/          # 示例 Skill（非运行时必选）
├── tests/
├── run_cli.py
└── run_mcp_server.py
```

---

## 常用环境变量

| 变量 | 默认（约） | 说明 |
|------|------------|------|
| `DEEPSEEK_API_KEY` | — | **必填** |
| `EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | 嵌入 |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | 重排 |
| `INGEST_STRUCTURED_CHUNK` | `true` | 结构化切块总开关 |
| `INGEST_DROP_REFERENCES` | `true` | 默认不入库参考文献 |
| `INGEST_FILTER_GARBAGE` | `true` | 过滤页码/页眉页脚等 |
| `TOP_K` / `RETRIEVAL_RECALL_K` | `3` / `8` | 精排 / 宽召回 |
| `RETRIEVAL_THRESHOLD` | `0.55` | 检索门禁（vector_score） |
| `ANSWER_CONFIDENCE_THRESHOLD` | `0.45` | 回答门禁 |
| `GROUNDING_THRESHOLD` | `0.6` | Critic 支撑率通过线 |
| `GROUNDING_FALLBACK_MIN_SUPPORT` | `0.4` | 修订达上限后的部分支撑分界 |
| `MAX_REVISE` | `2` | Writer⇄Critic 最多再写轮数 |
| `MAX_REACT_ITERATIONS` | `2` | Policy ReAct 轮次 |
| `FAST_MODE` | `true` | 路由与深研快路径 |
| `PARALLEL_RESEARCH` | `true` | 深研子任务并行 |
| `LTM_ASYNC_CONSOLIDATE` | `true` | 定稿后异步写 LTM |
| `QUERY_REWRITE_ENABLED` | `false` | Query 改写 |
| `MCP_CLIENT_ENABLED` | `false` | 远程 MCP |
| `MCP_SERVERS` | `[]` | JSON：`url` 或 `command`+`args` |

完整列表见 [`.env.example`](./.env.example)。MCP：[docs/MCP_REMOTE_SETUP.md](./docs/MCP_REMOTE_SETUP.md)。

---

## 内置工具

| 工具 | 功能 |
|------|------|
| `knowledge_search` | 本地知识库语义检索 |
| `kb_overview` | 知识库文档目录 |
| `calculator` | 安全表达式计算 |
| `web_search` | 联网（Tavily / DuckDuckGo·ddgs） |

扩展：在 `src/tools/` 注册，或开启 MCP Client（如百度搜索 `baidu_*`）。

---

## 设计原则（实现约束）

1. **证据优先于文笔**：无支撑则拒答或提示上传，禁止用「无法确定」虚报支撑率 100%。
2. **有界循环**：ReAct / 修订 / 检索次数均有硬上限；达上限必须走 fallback。
3. **确定性与策略分离**：路由、阈值、预算在 Workflow/配置；策略决策在 Policy/深研节点。
4. **置信度用 `vector_score`**：不用 Cross-Encoder 原始分打穿拒答门。
5. **消息链完整性**：截断 Context 时整组丢弃 `assistant`+`tool`，避免 API 400。
6. **评测诚实**：无金标不做正式 Recall@K 宣传。

---

## License

MIT
