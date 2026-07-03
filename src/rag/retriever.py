"""
src/rag/retriever.py — 语义检索器
"""
from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import settings
from src.memory.embedding import encode_query
from src.rag.kb_utils import normalize_chunk
from src.utils.logger import get_logger

log = get_logger("rag.retriever")

_META = {"hnsw:space": "cosine"}


class Retriever:
    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(
            path=settings.RAG_CHROMA_DIR, settings=ChromaSettings(anonymized_telemetry=False)
        )
        n = self.col.count()
        log.info(f"知识库路径={settings.RAG_CHROMA_DIR}，共 {n} 块")

    @property
    def col(self):
        return self.client.get_or_create_collection(name=settings.RAG_COLLECTION, metadata=_META)

    def count(self) -> int:
        return self.col.count()

    def stats(self) -> tuple[int, int]:
        """返回 (文本块数, 文档数)，仅读 metadata，不加载正文。"""
        chunks = self.count()
        if chunks == 0:
            return 0, 0
        metas = self.col.get(include=["metadatas"])["metadatas"]
        docs = len({m.get("filename", "?") for m in metas})
        return chunks, docs

    def search(self, query: str, k: int | None = None) -> list[dict]:
        k = k or settings.TOP_K
        if self.count() == 0:
            log.warning(
                f"知识库为空（path={settings.RAG_CHROMA_DIR}），"
                "请运行: python -m src.rag.ingest ./data/docs"
            )
            return []
        res = self.col.query(
            query_embeddings=[encode_query(query)],
            n_results=min(k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            out.append(
                {
                    "content": normalize_chunk(doc),
                    "filename": meta.get("filename", "?"),
                    "score": round(1 - dist / 2, 4),
                }
            )
        return out

    def list_sources(self, sample_chars: int = 150) -> list[dict]:
        n = self.count()
        if n == 0:
            return []
        data = self.col.get(include=["metadatas", "documents"], limit=n)
        buckets: dict[str, list[str]] = {}
        for doc, meta in zip(data["documents"], data["metadatas"]):
            fn = meta.get("filename", "?")
            buckets.setdefault(fn, []).append(doc or "")
        out = []
        for fn in sorted(buckets):
            docs = buckets[fn]
            sample = normalize_chunk(docs[0].replace("\n", " ").strip())[:sample_chars]
            out.append({"filename": fn, "chunks": len(docs), "sample": sample})
        return out


retriever = Retriever()
