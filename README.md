# 🧠 DeepResearch — 多 Agent 智能研究系统

> 可写进简历核心项目的**多 Agent（Multi-Agent）系统**：基于 **LangGraph 1.x** 编排，集成
> **RAG 检索增强**、**幻觉治理**、**工作记忆 + 长期记忆**、**工具系统 + 中间件**、
> **任务规划 + 规划质量评估**，前端 **Streamlit**。LLM 直连 **DeepSeek**（OpenAI 兼容协议）。
> 面向 2026 主流栈：langgraph / openai v2 / pydantic v2 / chromadb v1。

## 一句话价值
用户提一个问题，系统像一个研究小组一样工作：**先规划、再分工取证、然后成文、最后事实核查**，
把幻觉挡在返回之前，并把经验沉淀进长期记忆，越用越聪明。

## 架构总览
```
用户问题
   │
① recall_memory   长期记忆召回（内置，不在前端展示）
② plan            规划Agent：任务分解 → 规划自评 → 不达标则重规划
③ research        研究员Agent：按子任务调用工具取证（统一走中间件：重试/护栏/降级）
④ write           写作Agent：综合成答案（prompt反幻觉约束 = 第一道防线）
⑤ critic          批评家Agent：grounding事实核查（= 第二道防线）
        ┌─ 通过 → consolidate 定稿 + 写入长期记忆 → END
        └─ 疑似幻觉 → 回到④带修订说明重写（闭环纠错，限次防死循环）
```
编排引擎是 LangGraph 的 StateGraph：各 Agent 是图上的节点，共享一份 AgentState 在节点间流转，
构成多 Agent 的“协作总线”；节点内部再用 WorkingMemory 做消息级通信。

## 需求点 → 代码位置对照
| 能力 | 说明 | 文件 |
|------|------|------|
| RAG 检索增强 | 入库/切分/向量化/检索 | src/rag/ingest.py, retriever.py |
| 解决幻觉 | 生成前约束 + 生成后grounding核查 + 闭环重写 | src/rag/grounding.py, agents/critic_agent.py, graph.py |
| 工作记忆(内置) | 短期scratchpad，分区+预算，不展示前端 | src/memory/working_memory.py |
| 长期记忆(内置) | ChromaDB持久化，跨会话召回/巩固 | src/memory/long_term_memory.py |
| 工具系统设计 | 统一契约+注册表+可发现+可观测 | src/tools/base.py, builtin_tools.py |
| 工具失败处理 | 错误分类+指数退避重试+降级+护栏 | src/middleware/pipeline.py |
| 中间件 | 洋葱模型横切：护栏→日志→重试→调用 | src/middleware/pipeline.py |
| 任务规划 | Plan-and-Execute分解式规划 | src/planning/planner.py |
| 规划质量评估&改进 | 四维打分+重规划闭环 | src/planning/evaluator.py |
| 多Agent系统 | Planner/Researcher/Writer/Critic | src/agents/* |
| Agent间协作通信 | 共享State + WorkingMemory消息 | src/orchestrator/state.py, graph.py |
| 多Agent优化 | 记忆先验/规划自评/幻觉闭环/工具容错 | src/orchestrator/graph.py |

## 快速开始
```bash
# 1. 装依赖
pip install -r requirements.txt
# 2. 配置 Key
cp .env.example .env   # 填入 DEEPSEEK_API_KEY（platform.deepseek.com 免费注册）
# 3. 构建知识库（已内置示例文档）
python -m src.rag.ingest ./data/docs
# 4a. 命令行验证整条链路
python run_cli.py "多Agent系统如何优化？"
# 4b. 前端
streamlit run frontend/streamlit_app.py   # http://localhost:8501
# 5. 测试
pytest tests/ -v
```
Windows PowerShell 同上（命令一致）。

## 为什么这套选型稳（踩坑说明）
- **不走 langchain 的 init_chat_model**：langchain 1.x + openai 2.x 下它会把 proxies 透传给 httpx，
  触发 `Client.__init__() got an unexpected keyword argument 'proxies'`。本项目在 src/llm/provider.py
  里**直接用 openai SDK v2 构造 client**（只传 api_key/base_url），彻底绕开。
- **LangGraph 节点是纯函数**，不依赖 langchain chat model 抽象，不受其版本变动影响。

## 简历写法示例
DeepResearch 多Agent智能研究系统（个人项目，Python / LangGraph / DeepSeek / ChromaDB）
- 基于 LangGraph 设计 Planner-Researcher-Writer-Critic 四角色多Agent协作管线，共享State+工作记忆实现Agent通信；
- 实现 RAG 与两级幻觉治理（生成前提示约束 + 生成后grounding事实核查闭环重写），以答案支撑率为质量门禁；
- 设计统一工具系统与中间件（注册表+参数校验+分类错误重试/降级/安全护栏），提升工具调用鲁棒性；
- 内置工作记忆+长期记忆（ChromaDB持久化），跨会话沉淀经验实现规划复用；
- 规划层做四维质量评估+自动重规划，保证复杂任务分解质量。
