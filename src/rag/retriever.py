"""
src/rag/retriever.py — 语义检索器

流程：领域词增强查询向量 → 宽召回 → 重排序 → 阈值过滤 → Top-K
"""
from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import settings
from src.rag.confidence import confidence_report, is_low_confidence
from src.rag.domain_lexicon import enhance_query_embedding, expand_query_text
from src.rag.kb_utils import normalize_chunk
from src.rag.rerank import rerank_hits
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
        """
        检索并重排。返回项含 content / filename / score，
        另附 _meta（不写入 Chroma）时由调用方剥离。
        """
        k = k or settings.TOP_K
        if self.count() == 0:
            log.warning(
                f"知识库为空（path={settings.RAG_CHROMA_DIR}），"
                "请运行: python -m src.rag.ingest ./data/docs"
            )
            return []

        expanded = expand_query_text(query)
        q_vec = enhance_query_embedding(query)
        recall_k = max(k, settings.RETRIEVAL_RECALL_K)
        recall_k = min(recall_k, self.count())

        res = self.col.query(
            query_embeddings=[q_vec],
            n_results=recall_k,
            include=["documents", "metadatas", "distances"],
        )
        candidates = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            vec_score = round(1 - dist / 2, 4)
            try:
                weight = float(meta.get("weight", 1.0) or 1.0)
            except (TypeError, ValueError):
                weight = 1.0
            weight = max(0.05, min(1.0, weight))
            # 低权重块（如保留的参考文献）降权，减少噪声抢占 Top-K
            adj = round(vec_score * weight, 4)
            candidates.append(
                {
                    "content": normalize_chunk(doc),
                    "filename": meta.get("filename", "?"),
                    "section": meta.get("section", "body"),
                    "weight": weight,
                    "vector_score": adj,
                    "score": adj,
                    "raw_vector_score": vec_score,
                }
            )

        ranked = rerank_hits(
            expanded or query,
            candidates,
            top_n=max(k * 2, k),  # 先多取一点再按文件去重压到 k
            min_score=settings.RETRIEVAL_THRESHOLD,
        )
        ranked = ranked[:k]
        report = confidence_report(ranked, query=query)
        log.info(
            f"检索完成 query={query[:36]!r} recall={len(candidates)} "
            f"out={len(ranked)} conf={report['confidence']:.3f} "
            f"pass={report['pass']} ood={report.get('ood_query')}"
        )
        return ranked

    def search_with_confidence(self, query: str, k: int | None = None) -> dict:
        """供工具层使用：hits + 置信度报告。"""
        hits = self.search(query, k=k)
        report = confidence_report(hits, query=query)
        return {
            "hits": hits,
            "confidence": report,
            "low_confidence": is_low_confidence(hits, query=query),
            "expanded_query": expand_query_text(query),
        }

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
