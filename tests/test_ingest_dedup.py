"""tests/test_ingest_dedup.py — 入库精确去重（临时 Chroma，假 embedding）"""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def isolated_kb(tmp_path, monkeypatch):
    chroma = tmp_path / "chroma"
    docs = tmp_path / "docs"
    chroma.mkdir()
    docs.mkdir()
    monkeypatch.setattr("config.settings.settings.RAG_CHROMA_DIR", str(chroma))
    monkeypatch.setattr("config.settings.settings.DOCS_DIR", str(docs))
    monkeypatch.setattr("config.settings.settings.RAG_COLLECTION", "test_dedup_kb")
    monkeypatch.setattr("config.settings.settings.INGEST_DEDUP_ENABLED", True)
    monkeypatch.setattr("config.settings.settings.INGEST_DEDUP_BY_FILE", True)
    monkeypatch.setattr("config.settings.settings.INGEST_DEDUP_BY_CHUNK", True)
    # 去重测试用简单按段切分，避免结构化切块干扰计数
    monkeypatch.setattr("config.settings.settings.INGEST_STRUCTURED_CHUNK", False)

    def fake_encode(texts):
        dim = 32
        out = []
        for t in texts:
            seed = sum(ord(c) for c in (t or "")[:40]) or 1
            out.append([((seed * (i + 1)) % 97) / 97.0 for i in range(dim)])
        return out

    monkeypatch.setattr("src.rag.ingest.encode_docs", fake_encode)
    monkeypatch.setattr(
        "src.rag.chunking.hybrid_chunk_text",
        lambda text, **kwargs: [p.strip() for p in str(text).split("\n\n") if len(p.strip()) >= 10],
    )
    monkeypatch.setattr(
        "src.rag.structure_chunk.hybrid_chunk_text",
        lambda text, **kwargs: [p.strip() for p in str(text).split("\n\n") if len(p.strip()) >= 10],
    )
    return docs


def test_content_fingerprint_stable():
    from src.rag.ingest import content_fingerprint

    a = content_fingerprint("高光谱  氮素\n监测")
    b = content_fingerprint("高光谱氮素监测")
    assert a == b
    assert a != content_fingerprint("别的内容")


def test_skip_duplicate_file_different_name(isolated_kb):
    from src.rag.ingest import ingest_file, _collection

    docs = isolated_kb
    body = "Paragraph one about hyperspectral remote sensing.\n\nParagraph two about nitrogen monitoring accuracy."
    f1 = docs / "paper_a.txt"
    f2 = docs / "paper_a_copy.txt"
    f1.write_text(body, encoding="utf-8")
    f2.write_text(body, encoding="utf-8")

    r1 = ingest_file(f1)
    r2 = ingest_file(f2)
    assert r1["ok"] and r1["chunks"] == 2
    assert r2["ok"] and r2.get("skipped_as_file_dup") is True
    assert r2["duplicate_of"] == "paper_a.txt"
    assert r2["chunks"] == 0
    assert _collection().count() == 2


def test_skip_duplicate_chunks_across_files(isolated_kb, monkeypatch):
    from src.rag.ingest import ingest_file, _collection

    # 关闭文件级去重，只测块级（两篇有一段重叠）
    monkeypatch.setattr("config.settings.settings.INGEST_DEDUP_BY_FILE", False)
    docs = isolated_kb
    shared = "Shared nitrogen estimation paragraph used by both documents."
    (docs / "doc1.txt").write_text(
        f"{shared}\n\nOnly document one unique paragraph content here.", encoding="utf-8"
    )
    (docs / "doc2.txt").write_text(
        f"{shared}\n\nOnly document two unique paragraph content here.", encoding="utf-8"
    )

    r1 = ingest_file(docs / "doc1.txt")
    r2 = ingest_file(docs / "doc2.txt")
    assert r1["chunks"] == 2
    assert r2["chunks"] == 1
    assert r2["skipped_chunks"] >= 1
    assert _collection().count() == 3


def test_same_filename_replace(isolated_kb):
    from src.rag.ingest import ingest_file, _collection

    docs = isolated_kb
    path = docs / "update_me.txt"
    path.write_text(
        "Old version paragraph alpha content.\n\nOld version paragraph beta content.",
        encoding="utf-8",
    )
    r0 = ingest_file(path)
    assert r0["ok"] and r0["chunks"] == 2, r0
    path.write_text("New version single paragraph only.", encoding="utf-8")
    r = ingest_file(path)
    assert r["ok"] and r["chunks"] == 1, r
    assert _collection().count() == 1
