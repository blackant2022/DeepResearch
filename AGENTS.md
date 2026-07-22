# AGENTS.md — DeepResearch 项目约束

本文档约束在本仓库协作的 **人类开发者与 AI Agent**（Cursor / Claude Code 等）。修改代码前请先阅读；与用户临时指示冲突时，以用户当次明确指令为准，但不得违反安全底线。

---

## 1. 项目是什么

- 学术文献向的 **LangGraph 多智能体**：Workflow 分流 → `chat` | `policy` | `deep_research`。
- 核心价值：本地 RAG + 可配置拒答/Grounding + 可选 MCP，而不是堆砌 Agent 数量。
- 主领域语料：高光谱 / 作物氮素等文献；通用闲聊走 `chat`，勿强行检索胡编。

---

## 2. 架构红线（禁止轻易改回）

1. **禁止恢复图级 `think ↔ tools` 空转。** ReAct 必须留在 Policy **节点内**，并受 `MAX_REACT_ITERATIONS` / `MAX_KNOWLEDGE_SEARCHES` 约束。
2. **同轮只走一条路由路径**（chat / policy / deep_research），不要并行多条业务边抢写 `final_answer`。
3. **深研修订环必须有出口**：`MAX_REVISE`；Grounding 默认 **批量核验**，`GROUNDING_PER_CLAIM` 默认 **false**（打开会导致分钟级延迟）。
4. **置信度门禁用 `vector_score`**，不要用 Cross-Encoder sigmoid 原始分做拒答依据。
5. **Context / 消息截断必须整组丢弃** `assistant`+`tool` 配对，禁止拆断 tool 链导致 API 400。

---

## 3. 改动范围与风格

- **最小改动**：只改任务需要的文件；禁止顺手大重构、无关重命名、批量「美化」。
- **不擅自**：新增文档（用户未要求时）、提交 Git、推远程、改 git config、强推、`--no-verify`。
- **密钥**：永不写入 README / `.env.example` / 测试断言 / 截图说明；真实密钥只在本地 `.env`。
- **依赖**：新增包需有充分理由，并更新 `requirements.txt`；能用已有栈（Chroma / FastEmbed / transformers）则不加新框架。
- **注释**：少而准，解释非显而易见的约束；不写叙事性废注释。
- **用户沟通**：默认简体中文；回答简洁，数字与能力边界要诚实（无标注集勿吹 Recall@K）。

---

## 4. RAG / 评测诚实性

- 代理指标（关键词、文件名、延迟、支撑率）可以写进报告；**没有段落金标就不要宣称正式 Recall@K / nDCG / RAGAS 分数**。
- 调参脚本与测试集：`scripts/`、`data/eval/`；全量 2k 检索很慢，优先抽样。
- 入库去重、分块、Rerank 模型名以 `config/settings.py` 与 `.env.example` 为准；改默认值需同步示例与 README。

---

## 5. MCP

- Client 支持 **`url`（远程）** 与 **`command`+`args`（stdio）**；参数 schema 由远端 `list_tools` 下发，**不要**在 `.env` 手写工具参数表。
- `MCP_CLIENT_ENABLED` 默认应可关闭；主流程不得强依赖外网 MCP。
- 用户说「百度搜索」时优先 MCP `baidu_*` 工具，避免误走缺依赖的 `web_search`。
- 接入说明：`docs/MCP_REMOTE_SETUP.md`。

---

## 6. 测试与验证

- 改编排 / 消息链 / RAG 门禁后：至少跑相关 `pytest`（如 `tests/test_smoke.py`、消息链、去重、chunking）。
- 延迟相关改动：应用同一条深研或检索样例做前后对比，避免只改代码不测。
- 不要提交 `data/chroma/`、`data/ltm_chroma/`、用户 PDF、`.env`。

---

## 7. 推荐目录约定

| 用途 | 位置 |
|------|------|
| 运行配置 | `config/settings.py`、`.env` |
| AI 项目约束 | 本文件 `AGENTS.md`；细规则可加 `.cursor/rules/*.mdc` |
| 示例 Skill（非运行时） | `examples/skills/` |
| 评测数据 | `data/eval/` |

---

## 8. 提交信息（仅当用户明确要求提交时）

- 先 `git status` / `diff` / `log` 再提交；信息说明「为什么」；不把密钥与大体积向量库加入暂存区。

---

## 9. 一句话原则

**可观测、有上限、可降级、证据优先于文笔；Agent 加速实现，验收靠测试与诚实指标。**
