"""
config/settings.py — 全局配置
用 pydantic-settings 从 .env 读取，所有参数集中管理，避免硬编码。
路径一律解析为相对【项目根目录】的绝对路径，避免 Streamlit/IDE 从子目录启动时找不到向量库。
"""
from __future__ import annotations

import os
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（config/ 的上一级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _abs_path(value: str) -> str:
    p = Path(value)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return str(p.resolve())


class Settings(BaseSettings):
    # ---- LLM (DeepSeek, OpenAI 兼容协议) ----
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    LLM_MODEL: str = "deepseek-chat"
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 2048

    # ---- 嵌入模型（FastEmbed / ONNX，比 sentence-transformers 轻，首次检索时自动下载）----
    EMBEDDING_MODEL: str = "BAAI/bge-small-zh-v1.5"

    # ---- 向量库路径（相对项目根，加载后转为绝对路径）----
    RAG_CHROMA_DIR: str = "data/chroma"
    LTM_CHROMA_DIR: str = "data/ltm_chroma"
    DOCS_DIR: str = "data/docs"
    RAG_COLLECTION: str = "kb_documents"
    LTM_COLLECTION: str = "long_term_memory"

    # ---- RAG 参数 ----
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 60
    TOP_K: int = 3

    # ---- Agent 运行参数 ----
    FAST_MODE: bool = True           # 默认快速模式（检索+一次生成，跳过完整多Agent链）
    MAX_REPLAN: int = 0
    MAX_TOOL_RETRY: int = 3
    MAX_REACT_ITERATIONS: int = 8    # 超级智能体 ReAct 最大轮次
    GROUNDING_THRESHOLD: float = 0.6
    MAX_SUBTASKS: int = 6
    CHECKPOINT_DIR: str = "data/checkpoints"

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("RAG_CHROMA_DIR", "LTM_CHROMA_DIR", "DOCS_DIR", "CHECKPOINT_DIR", mode="before")
    @classmethod
    def _resolve_data_paths(cls, v: str) -> str:
        return _abs_path(v) if v else v


settings = Settings()

# 确保目录存在
for _p in (settings.RAG_CHROMA_DIR, settings.LTM_CHROMA_DIR, settings.DOCS_DIR, settings.CHECKPOINT_DIR):
    os.makedirs(_p, exist_ok=True)
