# AGENTS.md — DeepResearch 项目约束（人类 + 大模型必读）

本文档约束在本仓库协作的 **人类开发者与 AI Agent**（Cursor / Claude Code / 其它 Coding Agent）。

**优先级：** 用户当次明确指令 > 本文档 > 一般编程习惯。  
**例外：** 安全底线（密钥、破坏性 Git、编造评测分数）不可被用户口头「随便写写」覆盖到提交物里。

修改任何代码前先读完本文；不确定时先问用户，不要猜着大改。

---

## 0. 给大模型的硬指令（照办）

1. **先读后改**：改编排 / RAG / 消息链前，先打开 `src/orchestrator/graph.py`、`config/settings.py`、相关测试，对齐现状再动手。
2. **最小改动**：只改任务需要的文件；禁止顺手重构、无关重命名、全库格式化、「顺便优化」。
3. **禁止恢复已否决架构**：不要把 ReAct 做回图级 `think ↔ tools` 空转；不要并行多条业务路径抢写 `final_answer`。
4. **默认值变更必须三处同步**：`config/settings.py`、`.env.example`、`README.md`（必要时本文件）。
5. **不擅自**：提交 Git、推远程、改 `git config`、`--force` 推送、`--no-verify`，除非用户明确要求。
6. **密钥零容忍**：真实 Key 只许出现在本地 `.env`；禁止写入 README、`.env.example`、测试断言、注释、截图说明、commit message。
7. **诚实表达**：无段落金标不写正式 Recall@K / nDCG / RAGAS；未实现的能力（如完整 PDF Layout 解析、图 VLM 入库）不得写成「已支持」。
8. **改完要验证**：至少跑相关 `pytest`；延迟相关改动用同一条样例对比，禁止「应该没问题」。
9. **用户沟通默认简体中文**；结论先行；数字与边界要可核对。
10. **以仓库代码为准**：文档与代码冲突时，以代码与测试行为为准，并应修正文档，而不是臆造功能。

---

## 1. 项目是什么

- **定位**：学术文献向的 LangGraph 多智能体 Deep Research 系统。
- **主链路**：Workflow（LTM → 可选改写 → 多模态 → Router）→ 三选一  
  `chat` | `policy`（节点内 ReAct）| `deep_research`（Planner→Researcher→Writer⇄Critic→Consolidate/SafeFallback）。
- **核心价值**：本地 RAG + 双层置信度 + Grounding 拒答/上传提示 + 可选 MCP；**不是**堆 Agent 数量。
- **主领域语料**：高光谱 / 作物氮素等；闲聊走 `chat`，域外无证据应拒答，禁止硬检索胡编。

仓库：https://github.com/blackant2022/DeepResearch

---

## 2. 架构红线（禁止轻易改回）

1. **ReAct 只在 Policy 节点内**，受 `MAX_REACT_ITERATIONS`、`MAX_KNOWLEDGE_SEARCHES` 约束。禁止图级 think↔tools 空转。
2. **同轮只走一条路由路径**（chat / policy / deep_research），禁止并行业务边抢写 `final_answer`。
3. **深研修订环必须有出口**：`MAX_REVISE`；未通过且达上限 → `safe_fallback`，不得无限 rewrite。
4. **Grounding**：默认批量核验；`GROUNDING_PER_CLAIM` 默认 **false**（打开可导致分钟级延迟）。  
   **纯拒答 / 论断数为 0 → `grounded=false`、`support_rate=0`，禁止虚报 100% 并 consolidate。**  
   `no_claims` / `refusal` 应直接 fallback，不要无意义修订空转。
5. **低支撑兜底**：支撑率过低须提示用户上传 PDF/DOCX/TXT/MD 或实验数据；失败路径**不写 LTM**。
6. **置信度门禁用 `vector_score`**（及权重调整后的检索分），禁止用 Cross-Encoder sigmoid 原始分做拒答依据。
7. **Context / 消息截断必须整组丢弃** `assistant`+`tool` 配对，禁止拆断 tool 链导致 API 400。
8. **入库**：结构化切块（`structure_chunk`）+ 去重为默认路径；改切块策略后须提示用户**重新入库**。不要把参考文献噪声默认灌进主检索集（`INGEST_DROP_REFERENCES` 默认 true）。
9. **主图 Checkpoint 用可序列化 State**；`wm_items` 为 Working Memory 快照。深研子图当前是主节点内一次 invoke，不要假装已支持子图节点级断点续跑，除非真正接上同一 checkpointer 与 resume 语义。

---

## 3. 改动范围与风格

- **最小改动**；一个 PR / 一次任务一个焦点。
- **依赖**：新增包需充分理由并更新 `requirements.txt`；优先 Chroma / FastEmbed / transformers，不加无关重框架。
- **注释**：只解释非显而易见约束；禁止叙事性废注释与大段教程式注释。
- **测试**：新行为先补/改测试再改实现（尤其 Grounding、消息链、入库、路由）。
- **文档**：用户未要求时不新增长篇 markdown；但**改了默认行为必须更新** README / `.env.example` / 本文件相关条款。

---

## 4. RAG / 评测诚实性

- 可报告：关键词命中、文件名覆盖、延迟、支撑率、拒答率等**代理指标**。
- **不可宣称**：无人工段落金标时的正式 Recall@K / nDCG / RAGAS「SOTA」分数。
- 调参与测试集：`scripts/`、`data/eval/`；全量 2k 检索很慢，优先抽样。
- 模型名与阈值以 `config/settings.py` + `.env.example` 为准。

当前入库要点（实现见 `src/rag/structure_chunk.py`、`ingest.py`）：

- 摘要整块、`section` / `weight` metadata  
- 表格转 Markdown 同块  
- 垃圾行过滤；参考文献默认丢弃或低权  
- 文件级 / 块级 SHA256 去重；同名覆盖先删旧块  

检索：`retriever` 对 `weight<1` 降权后再 Rerank / 门禁。

---

## 5. MCP

- 支持 **`url`（远程 HTTP/SSE）** 与 **`command`+`args`（stdio）**。
- 参数 schema 由远端 `list_tools` 下发；**禁止**在 `.env` 手写完整工具参数表。
- `MCP_CLIENT_ENABLED` 默认关闭；主路径不得强依赖外网 MCP。
- 用户提到「百度搜索」时优先 MCP `baidu_*`，避免误走缺依赖的 `web_search`。
- 说明：`docs/MCP_REMOTE_SETUP.md`。

---

## 6. 记忆与状态（勿混淆）

| 概念 | 作用 | 注意 |
|------|------|------|
| Working Memory / `wm_items` | 单轮任务草稿纸（fact/scratch/observation/message） | 每轮入口可 Overwrite 清空 |
| LTM | 跨会话任务摘要向量召回 → `memory_hint` | 仅成功定稿写入；fallback 不写 |
| Checkpoint / MemorySaver | 按 `thread_id` 保存图状态 | ≠ LTM；进程内 MemorySaver 重启即丢 |
| RAG Chroma | 文献知识库 | ≠ LTM |

Agent 间正式协作靠 **State 字段**（plan/findings/draft/revise_note/grounding）；`say()` 写入的 WM message 目前主要是日志向，不要假设有消费者。

---

## 7. 测试与验证

改编排 / 消息链 / Grounding / 入库后，至少覆盖相关用例，例如：

- `tests/test_smoke.py`（含三档 fallback、拒答非 100%）  
- `tests/test_tool_message_chain.py`  
- `tests/test_ingest_dedup.py` / `tests/test_structure_chunk.py` / `tests/test_chunking.py`  
- `tests/test_rag_quality.py`  

禁止提交：`data/chroma/`、`data/ltm_chroma/`、用户 PDF、`.env`、密钥、大型缓存。

---

## 8. 目录约定

| 用途 | 位置 |
|------|------|
| 运行配置 | `config/settings.py`、`.env` |
| AI / 协作者约束 | 本文件 `AGENTS.md`；细规则可加 `.cursor/rules/*.mdc` |
| 示例 Skill（非运行时） | `examples/skills/` |
| 评测 | `data/eval/`、`scripts/` |
| MCP 接入 | `docs/MCP_REMOTE_SETUP.md` |

---

## 9. 提交与推送（仅当用户明确要求）

1. `git status` / `diff` / `log`  
2. 暂存时排除密钥与向量库  
3. 提交信息写清「为什么」  
4. 推送仅 `git push` 到用户指定远程；禁止擅自改 remote、改 config、强推 main  

---

## 10. 一句话原则

**可观测、有上限、可降级、证据优先于文笔；Agent 加速实现，验收靠测试与诚实指标；文档与代码必须同步，大模型不得臆造能力。**
