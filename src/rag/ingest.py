"""
src/rag/ingest.py — 文档入库（RAG 知识库构建）
支持 PDF / DOCX / TXT / MD，切分后向量化写入 ChromaDB。
运行：python -m src.rag.ingest ./data/docs
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from src.memory.embedding import encode_docs
from src.utils.logger import get_logger

log = get_logger("rag.ingest")

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP,
    separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
)


def _read(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n\n".join((p.extract_text() or "") for p in reader.pages)
    if ext == ".docx":
        import docx
        return "\n\n".join(p.text for p in docx.Document(str(path)).paragraphs if p.text.strip())
    for enc in ("utf-8", "gbk", "gb2312"):
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, LookupError):
            continue
    raise RuntimeError(f"无法读取 {path}")


def _collection():
    client = chromadb.PersistentClient(
        path=settings.RAG_CHROMA_DIR, settings=ChromaSettings(anonymized_telemetry=False)
    )
    return client.get_or_create_collection(
        name=settings.RAG_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def _delete_file_chunks(col, filename: str) -> None:
    """同一文件重复入库时先删旧块，避免检索噪音。"""
    try:
        batch = col.get(where={"filename": filename}, include=[])
        ids = batch.get("ids") or []
        if ids:
            col.delete(ids=ids)
            log.info(f"清除旧块 {filename} → {len(ids)} 块")
    except Exception as e:  # noqa: BLE001
        log.warning(f"清除 {filename} 旧块失败: {e}")


_SUPPORTED = {".pdf", ".docx", ".txt", ".md"}


def ingest_file(path: str | Path) -> dict:
    """入库单个文件，返回结构化结果。"""
    f = Path(path)
    if f.suffix.lower() not in _SUPPORTED:
        return {"ok": False, "filename": f.name, "chunks": 0, "error": "不支持的格式"}
    col = _collection()
    try:
        text = _read(f)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "filename": f.name, "chunks": 0, "error": str(e)}
    chunks = [c.strip() for c in _splitter.split_text(text) if len(c.strip()) >= 10]
    if not chunks:
        return {"ok": False, "filename": f.name, "chunks": 0, "error": "未能提取有效文本"}
    _delete_file_chunks(col, f.name)
    doc_id = str(uuid.uuid4())[:8]
    col.add(
        ids=[f"{doc_id}_{i}" for i in range(len(chunks))],
        embeddings=encode_docs(chunks),
        documents=chunks,
        metadatas=[{"filename": f.name, "chunk": i} for i in range(len(chunks))],
    )
    log.info(f"入库 {f.name} → {len(chunks)} 块")
    return {"ok": True, "filename": f.name, "chunks": len(chunks), "error": ""}


def ingest_uploads(files: list[tuple[str, bytes]]) -> list[dict]:
    """保存上传文件到 docs 目录并入库。"""
    docs_dir = Path(settings.DOCS_DIR)
    docs_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name, content in files:
        dest = docs_dir / Path(name).name  # 防止路径注入
        dest.write_bytes(content)
        results.append(ingest_file(dest))
    return results


def ingest_path(path: str) -> int:
    col = _collection()
    root = Path(path)
    files = [root] if root.is_file() else [f for f in root.rglob("*") if f.is_file()]
    total = 0
    for f in files:
        if f.suffix.lower() not in _SUPPORTED:
            continue
        r = ingest_file(f)
        if r["ok"]:
            total += r["chunks"]
    log.info(f"完成，共 {total} 块，知识库现有 {col.count()} 块")
    return total


if __name__ == "__main__":
    ingest_path(sys.argv[1] if len(sys.argv) > 1 else settings.DOCS_DIR)
