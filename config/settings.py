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
    TOP_K: int = 3
    RETRIEVAL_RECALL_K: int = 16  # 初筛宽召回条数，再交重排

    # ---- 重排序 ----
    RERANK_ENABLED: bool = True
    RERANK_BACKEND: str = "auto"  # auto | cross_encoder | dense
    RERANK_MODEL: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    RERANK_MIN_SCORE: float = 0.35  # 重排后低于此分过滤（cross-encoder 经 sigmoid）

    # ---- 领域词典 / 向量增强 ----
    DOMAIN_LEXICON_ENABLED: bool = True
    DOMAIN_LEXICON_PATH: str = "data/domain_lexicon.json"
    DOMAIN_EMBED_BLEND: float = 0.15  # 查询向量与术语向量混合比例

    # ---- 问答置信度兜底 ----
    ANSWER_CONFIDENCE_ENABLED: bool = True
    ANSWER_CONFIDENCE_THRESHOLD: float = 0.45  # 低于则输出兜底提示

    # ---- Agent ----
    FAST_MODE: bool = True
    MAX_REPLAN: int = 0
    MAX_TOOL_RETRY: int = 3
    MAX_REACT_ITERATIONS: int = 4
    GROUNDING_THRESHOLD: float = 0.6
    RETRIEVAL_THRESHOLD: float = 0.55
    PARALLEL_TOOLS: bool = True
    PARALLEL_TOOL_WORKERS: int = 4

    # ---- MCP ----
    MCP_CLIENT_ENABLED: bool = False
    MCP_SERVERS: str = ""
    MAX_SUBTASKS: int = 6
    CHECKPOINT_DIR: str = "data/checkpoints"
    QUERY_REWRITE_ENABLED: bool = False

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
