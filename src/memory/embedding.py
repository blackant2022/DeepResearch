"""
src/memory/embedding.py — 嵌入服务（FastEmbed / ONNX，轻量、延迟加载）
"""
from __future__ import annotations

from config.settings import settings
from src.utils.logger import get_logger

log = get_logger("embedding")

_model = None


def _get_model():
    global _model
    if _model is None:
        from fastembed import TextEmbedding

        log.info(f"加载嵌入模型 {settings.EMBEDDING_MODEL}（FastEmbed ONNX，首次约 90MB）…")
        _model = TextEmbedding(model_name=settings.EMBEDDING_MODEL)
        log.info("嵌入模型就绪")
    return _model


def encode_docs(texts: list[str]) -> list[list[float]]:
    """文档/段落向量化（入库用）。"""
    return [vec.tolist() for vec in _get_model().embed(texts)]


def encode_query(text: str) -> list[float]:
    """查询向量化（检索用）。"""
    return encode_docs([text])[0]
