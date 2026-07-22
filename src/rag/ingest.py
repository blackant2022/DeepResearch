"""
src/rag/ingest.py — 文档入库（RAG 知识库构建）
支持 PDF / DOCX / TXT / MD，切分后向量化写入 ChromaDB。
分块策略：结构化切块（摘要整块 / 表格 Markdown / 参考文献处理）
         + 语义相似度划分 + 长段落动态滑动窗口（见 structure_chunk / chunking）。

去重（可配置）：
  1. 文件级：全文规范化哈希；跨文件名内容相同则跳过入库
  2. 块级：块内容哈希；与库内已有块相同则跳过（含本文件内重复）
  3. 同名文件：仍先删旧块再写入（更新语义）

运行：python -m src.rag.ingest ./data/docs
"""
from __future__ import annotations

import hashlib
import re
import sys
import uuid
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import settings
from src.memory.embedding import encode_docs
from src.rag.kb_utils import normalize_chunk
from src.rag.structure_chunk import StructuredChunk, structure_document
from src.utils.logger import get_logger

log = get_logger("rag.ingest")

_WS = re.compile(r"\s+")


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


def content_fingerprint(text: str) -> str:
    """规范化后 SHA256，用于文件/块精确去重。"""
    cleaned = normalize_chunk(text or "")
    cleaned = _WS.sub("", cleaned).strip().lower()
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()


def _delete_file_chunks(col, filename: str) -> int:
    """同一文件重复入库时先删旧块，避免检索噪音。返回删除条数。"""
    try:
        batch = col.get(where={"filename": filename}, include=[])
        ids = batch.get("ids") or []
        if ids:
            col.delete(ids=ids)
            log.info(f"清除旧块 {filename} → {len(ids)} 块")
            return len(ids)
    except Exception as e:  # noqa: BLE001
        log.warning(f"清除 {filename} 旧块失败: {e}")
    return 0


def _load_dedup_index(col) -> tuple[set[str], dict[str, str]]:
    """
    扫描库内 metadata，返回：
      - chunk_hashes: 已有 content_hash
      - file_hash → 任一已有 filename
    无 hash 的旧数据忽略（不影响新入库去重）。
    """
    chunk_hashes: set[str] = set()
    file_owners: dict[str, str] = {}
    try:
        batch = col.get(include=["metadatas"])
    except Exception as e:  # noqa: BLE001
        log.warning(f"加载去重索引失败: {e}")
        return chunk_hashes, file_owners

    for meta in batch.get("metadatas") or []:
        if not isinstance(meta, dict):
            continue
        ch = meta.get("content_hash")
        if isinstance(ch, str) and ch:
            chunk_hashes.add(ch)
        fh = meta.get("file_hash")
        fn = meta.get("filename")
        if isinstance(fh, str) and fh and isinstance(fn, str) and fn and fh not in file_owners:
            file_owners[fh] = fn
    return chunk_hashes, file_owners


def _dedupe_structured_local(
    chunks: list[StructuredChunk],
) -> tuple[list[StructuredChunk], int]:
    """本文件内块级去重，保序。"""
    seen: set[str] = set()
    out: list[StructuredChunk] = []
    skipped = 0
    for c in chunks:
        h = content_fingerprint(c.text)
        if h in seen:
            skipped += 1
            continue
        seen.add(h)
        out.append(c)
    return out, skipped


_SUPPORTED = {".pdf", ".docx", ".txt", ".md"}


def ingest_file(path: str | Path) -> dict:
    """入库单个文件，返回结构化结果（含去重统计）。"""
    f = Path(path)
    if f.suffix.lower() not in _SUPPORTED:
        return {
            "ok": False, "filename": f.name, "chunks": 0,
            "skipped_chunks": 0, "duplicate_of": "", "error": "不支持的格式",
        }
    col = _collection()
    try:
        text = _read(f)
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False, "filename": f.name, "chunks": 0,
            "skipped_chunks": 0, "duplicate_of": "", "error": str(e),
        }

    file_hash = content_fingerprint(text)
    dedup_on = bool(getattr(settings, "INGEST_DEDUP_ENABLED", True))
    by_file = dedup_on and bool(getattr(settings, "INGEST_DEDUP_BY_FILE", True))
    by_chunk = dedup_on and bool(getattr(settings, "INGEST_DEDUP_BY_CHUNK", True))

    existing_chunks: set[str] = set()
    file_owners: dict[str, str] = {}
    if by_file or by_chunk:
        existing_chunks, file_owners = _load_dedup_index(col)

    # 跨文件全文重复：跳过（同名文件除外，同名走「删旧写新」更新）
    if by_file and file_hash in file_owners and file_owners[file_hash] != f.name:
        dup = file_owners[file_hash]
        log.info(f"跳过全文重复 {f.name}（与 {dup} 内容相同）")
        return {
            "ok": True,
            "filename": f.name,
            "chunks": 0,
            "skipped_chunks": 0,
            "duplicate_of": dup,
            "error": "",
            "skipped_as_file_dup": True,
        }

    structured = [
        c for c in structure_document(text)
        if len((c.text or "").strip()) >= 10
    ]
    if not structured:
        return {
            "ok": False, "filename": f.name, "chunks": 0,
            "skipped_chunks": 0, "duplicate_of": "", "error": "未能提取有效文本",
        }

    local_skip = 0
    if by_chunk:
        structured, local_skip = _dedupe_structured_local(structured)

    # 同名更新：先删旧块，并从索引中去掉该文件留下的 hash（避免「自己挡自己」）
    _delete_file_chunks(col, f.name)
    if by_file or by_chunk:
        existing_chunks, file_owners = _load_dedup_index(col)

    to_add: list[StructuredChunk] = []
    hashes: list[str] = []
    cross_skip = 0
    for c in structured:
        h = content_fingerprint(c.text)
        if by_chunk and h in existing_chunks:
            cross_skip += 1
            continue
        if h in hashes:  # 同批再保险
            cross_skip += 1
            continue
        to_add.append(c)
        hashes.append(h)
        existing_chunks.add(h)

    skipped_total = local_skip + cross_skip
    if not to_add:
        log.info(f"入库 {f.name} → 0 块（全部被去重，跳过 {skipped_total}）")
        return {
            "ok": True,
            "filename": f.name,
            "chunks": 0,
            "skipped_chunks": skipped_total,
            "duplicate_of": "",
            "error": "",
            "skipped_as_file_dup": False,
        }

    doc_id = str(uuid.uuid4())[:8]
    docs = [c.text for c in to_add]
    col.add(
        ids=[f"{doc_id}_{i}" for i in range(len(to_add))],
        embeddings=encode_docs(docs),
        documents=docs,
        metadatas=[
            {
                "filename": f.name,
                "chunk": i,
                "content_hash": hashes[i],
                "file_hash": file_hash,
                "section": to_add[i].section,
                "weight": float(to_add[i].weight),
            }
            for i in range(len(to_add))
        ],
    )
    log.info(
        f"入库 {f.name} → {len(to_add)} 块"
        + (f"（去重跳过 {skipped_total}）" if skipped_total else "")
    )
    return {
        "ok": True,
        "filename": f.name,
        "chunks": len(to_add),
        "skipped_chunks": skipped_total,
        "duplicate_of": "",
        "error": "",
        "skipped_as_file_dup": False,
    }


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
    skipped = 0
    for f in files:
        if f.suffix.lower() not in _SUPPORTED:
            continue
        r = ingest_file(f)
        if r["ok"]:
            total += r["chunks"]
            skipped += int(r.get("skipped_chunks") or 0)
            if r.get("skipped_as_file_dup"):
                skipped += 1
    log.info(
        f"完成，共写入 {total} 块，去重相关跳过约 {skipped}，知识库现有 {col.count()} 块"
    )
    return total


if __name__ == "__main__":
    ingest_path(sys.argv[1] if len(sys.argv) > 1 else settings.DOCS_DIR)
