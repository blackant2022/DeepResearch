"""
config/settings.py — 全局配置

敏感信息（API Key）一律：
  1. 只从 .env / 环境变量读取
  2. 使用 pydantic SecretStr，禁止打印/序列化明文
  3. 代码与 .env.example 中不得出现真实 Key
"""
from __future__ import annotations

import os
from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _abs_path(value: str) -> str:
    p = Path(value)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return str(p.resolve())


def secret_value(value: SecretStr | str | None) -> str:
    """安全取出密钥明文（仅供 SDK 调用，勿写入日志）。"""
    if value is None:
        return ""
    if isinstance(value, SecretStr):
        return (value.get_secret_value() or "").strip()
    return str(value).strip()


class Settings(BaseSettings):
    # ---- LLM（Key 只来自 .env）----
    DEEPSEEK_API_KEY: SecretStr = SecretStr("")
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    LLM_MODEL: str = "deepseek-chat"
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 2048

    # ---- 多模态（空则复用 DeepSeek）----
    MULTIMODAL_API_KEY: SecretStr = SecretStr("")
    MULTIMODAL_BASE_URL: str = ""
    MULTIMODAL_MODEL: str = ""
    MULTIMODAL_MAX_TOKENS: int = 2048
    MULTIMODAL_ENABLED: bool = True
    CHAT_ATTACHMENT_MAX_DOC_CHARS: int = 12000

    # ---- 嵌入 ----
    EMBEDDING_MODEL: str = "BAAI/bge-small-zh-v1.5"

    # ---- 路径 ----
    RAG_CHROMA_DIR: str = "data/chroma"
    LTM_CHROMA_DIR: str = "data/ltm_chroma"
    DOCS_DIR: str = "data/docs"
    RAG_COLLECTION: str = "kb_documents"
    LTM_COLLECTION: str = "long_term_memory"

    # ---- RAG 分块（语义相似度断点 + 长段动态滑动窗口）----
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 60
    CHUNK_OVERLAP_RATIO: float = 0.12  # 动态重叠下限比例（与 CHUNK_OVERLAP 取较大）
    CHUNK_MIN_SIZE: int = 40  # 过短块合并阈值
    SEMANTIC_SPLIT_THRESHOLD: float = 0.55  # 相邻句余弦相似度低于此值则断段
    # ---- 结构化入库（摘要整块 / 表 Markdown / 参考文献 / 垃圾行）----
    INGEST_STRUCTURED_CHUNK: bool = True
    INGEST_FILTER_GARBAGE: bool = True  # 过滤页码、页眉页脚、下载声明等
    INGEST_DROP_REFERENCES: bool = True  # True=不入库参考文献；False=低权重入库
    INGEST_REFERENCE_WEIGHT: float = 0.35  # 保留参考文献时的检索权重
    INGEST_ABSTRACT_MAX_CHARS: int = 3000  # 摘要超过此长度才二次切分
    # ---- 入库去重 ----
    INGEST_DEDUP_ENABLED: bool = True
    INGEST_DEDUP_BY_FILE: bool = True  # 全文哈希相同则跳过（跨文件名）
    INGEST_DEDUP_BY_CHUNK: bool = True  # 块内容哈希已存在则跳过
    TOP_K: int = 3
    RETRIEVAL_RECALL_K: int = 8  # 初筛宽召回（降延迟）

    # ---- 重排序 ----
    RERANK_ENABLED: bool = True
    RERANK_BACKEND: str = "auto"  # auto | cross_encoder | bge | dense
    RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANK_USE_FP16: bool = False  # GPU 可开；CPU 建议 false
    RERANK_MIN_SCORE: float = 0.35  # 仅 dense 后端参考；CE 排序后按 vector 门禁

    # ---- 领域词典 / 向量增强 ----
    DOMAIN_LEXICON_ENABLED: bool = True
    DOMAIN_LEXICON_PATH: str = "data/domain_lexicon.json"
    DOMAIN_EMBED_BLEND: float = 0.15  # 查询向量与术语向量混合比例

    # ---- 问答置信度兜底 ----
    ANSWER_CONFIDENCE_ENABLED: bool = True
    ANSWER_CONFIDENCE_THRESHOLD: float = 0.45  # 基于 vector_score

    # ---- Agent ----
    FAST_MODE: bool = True
    MAX_REPLAN: int = 0
    MAX_TOOL_RETRY: int = 3
    MAX_REACT_ITERATIONS: int = 2  # 减少重复搜库
    MAX_KNOWLEDGE_SEARCHES: int = 2  # 单轮问答最多 knowledge_search 次数
    MAX_REVISE: int = 2  # Critic 未通过时最多再写两轮，达到上限后必须退出
    GROUNDING_THRESHOLD: float = 0.6
    GROUNDING_FALLBACK_MIN_SUPPORT: float = 0.4  # 达修订上限后的中/低风险分界
    GROUNDING_MAX_CLAIMS: int = 6  # 论断封顶（合并核验后仍控制 token）
    GROUNDING_EVIDENCE_CHARS: int = 4000
    GROUNDING_PER_CLAIM: bool = False  # 默认关：打开会导致深研分钟级
    GROUNDING_PER_CLAIM_MAX: int = 3
    CRITIC_RAG_K: int = 3  # WM 证据充足时会自动跳过补检
    CRITIC_SKIP_RAG_IF_WM: int = 4  # WM facts ≥ 此值则不再搜同一 query
    RETRIEVAL_THRESHOLD: float = 0.55
    PARALLEL_TOOLS: bool = True
    PARALLEL_TOOL_WORKERS: int = 4
    PARALLEL_RESEARCH: bool = True  # 深研子任务并行（墙钟≈max而非sum）

    # ---- MCP ----
    MCP_CLIENT_ENABLED: bool = False
    MCP_SERVERS: str = ""
    MAX_SUBTASKS: int = 6
    FAST_MAX_SUBTASKS: int = 3  # FAST_MODE 下规划子任务上限
    CHECKPOINT_DIR: str = "data/checkpoints"
    QUERY_REWRITE_ENABLED: bool = False
    LTM_ASYNC_CONSOLIDATE: bool = True  # 定稿后异步写 LTM，不阻塞返回答案

    # ---- 联网搜索 ----
    WEB_SEARCH_PROVIDER: str = "auto"
    TAVILY_API_KEY: SecretStr = SecretStr("")
    TAVILY_SEARCH_DEPTH: str = "basic"
    WEB_SEARCH_MAX_RESULTS: int = 5
    WEB_SEARCH_REGION: str = "cn-zh"

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("RAG_CHROMA_DIR", "LTM_CHROMA_DIR", "DOCS_DIR", "CHECKPOINT_DIR", mode="before")
    @classmethod
    def _resolve_data_paths(cls, v: str) -> str:
        return _abs_path(v) if v else v

    def masked_summary(self) -> dict[str, str]:
        """可安全打印的配置摘要（密钥仅显示是否已配置）。"""
        return {
            "DEEPSEEK_API_KEY": "set" if secret_value(self.DEEPSEEK_API_KEY) else "missing",
            "MULTIMODAL_API_KEY": "set" if secret_value(self.MULTIMODAL_API_KEY) else "missing",
            "TAVILY_API_KEY": "set" if secret_value(self.TAVILY_API_KEY) else "missing",
            "DEEPSEEK_BASE_URL": self.DEEPSEEK_BASE_URL,
            "LLM_MODEL": self.LLM_MODEL,
            "FAST_MODE": str(self.FAST_MODE),
        }


settings = Settings()

for _p in (settings.RAG_CHROMA_DIR, settings.LTM_CHROMA_DIR, settings.DOCS_DIR, settings.CHECKPOINT_DIR):
    os.makedirs(_p, exist_ok=True)
