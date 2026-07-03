"""
src/memory/long_term_memory.py — 长期记忆（持久 / 跨会话）
"""
from __future__ import annotations

import time
import uuid

import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import settings
from src.memory.embedding import encode_docs, encode_query
from src.utils.logger import get_logger

log = get_logger("long_term_mem")

_META = {"hnsw:space": "cosine"}


class LongTermMemory:
    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(
            path=settings.LTM_CHROMA_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        n = self.col.count()
        log.info(f"长期记忆就绪，现有 {n} 条记忆")

    @property
    def col(self):
        return self.client.get_or_create_collection(name=settings.LTM_COLLECTION, metadata=_META)

    def remember(self, text: str, tags: str = "", source_query: str = "") -> str:
        mem_id = str(uuid.uuid4())
        self.col.add(
            ids=[mem_id],
            embeddings=[encode_docs([text])[0]],
            documents=[text],
            metadatas=[{"tags": tags, "source_query": source_query, "ts": time.time()}],
        )
        return mem_id

    def consolidate(self, summary: str, source_query: str) -> str:
        log.info("巩固长期记忆：写入本轮任务摘要")
        return self.remember(summary, tags="task_summary", source_query=source_query)

    def recall(self, query: str, k: int = 3) -> list[dict]:
        if self.col.count() == 0:
            return []
        res = self.col.query(
            query_embeddings=[encode_query(query)],
            n_results=min(k, self.col.count()),
            include=["documents", "metadatas", "distances"],
        )
        out = []
        docs = res.get("documents") or [[]]
        for doc, meta, dist in zip(docs[0], res["metadatas"][0], res["distances"][0]):
            out.append({"text": doc, "score": round(1 - dist / 2, 4), "meta": meta})
        return out

    def stats(self) -> dict:
        return {"total": self.col.count()}


ltm = LongTermMemory()
