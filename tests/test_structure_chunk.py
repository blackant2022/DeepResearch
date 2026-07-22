"""tests/test_structure_chunk.py — 摘要整块 / 表格 / 参考文献 / 垃圾过滤"""
from __future__ import annotations


def test_filter_garbage_drops_page_numbers():
    from src.rag.structure_chunk import filter_garbage_lines

    text = "正文第一段关于氮素监测。\n12\nPage 3 of 10\n第二段继续讨论高光谱。\nDownloaded from example.com"
    out = filter_garbage_lines(text)
    assert "氮素" in out and "高光谱" in out
    assert "Page 3" not in out
    assert "Downloaded" not in out
    assert "\n12\n" not in f"\n{out}\n"


def test_abstract_kept_as_single_chunk(monkeypatch):
    from config.settings import settings
    from src.rag.structure_chunk import structure_document

    monkeypatch.setattr(settings, "INGEST_STRUCTURED_CHUNK", True)
    monkeypatch.setattr(settings, "INGEST_FILTER_GARBAGE", True)
    monkeypatch.setattr(settings, "INGEST_DROP_REFERENCES", True)

    def fake_embed(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    doc = """
Title of the paper about crop nitrogen

Abstract
This study proposes a hyperspectral method for estimating leaf nitrogen content in maize.
The model achieves high accuracy on field datasets.

Introduction
Nitrogen is essential for crop growth and yield prediction in precision agriculture.

Methods
We collected canopy reflectance and used partial least squares regression.

References
[1] Smith et al. Remote Sensing of Environment, 2020.
[2] Zhang et al. Computers and Electronics in Agriculture, 2021.
""".strip()

    chunks = structure_document(doc, embed_fn=fake_embed)
    abstracts = [c for c in chunks if c.section == "abstract"]
    assert len(abstracts) == 1
    assert "hyperspectral" in abstracts[0].text
    assert "Introduction" not in abstracts[0].text or abstracts[0].text.lower().startswith("abstract")
    assert not any(c.section == "reference" for c in chunks)
    assert any(c.section in ("intro", "method", "body") for c in chunks)


def test_table_converted_to_markdown(monkeypatch):
    from config.settings import settings
    from src.rag.structure_chunk import extract_table_blocks, structure_document

    monkeypatch.setattr(settings, "INGEST_STRUCTURED_CHUNK", True)

    block = """
结果如下所示。

Table 1 Accuracy of nitrogen estimation models
Model  RMSE  R2
PLS  0.12  0.85
RF  0.10  0.88

后续讨论模型泛化能力。
""".strip()

    tables, rest = extract_table_blocks(block)
    assert tables
    assert "|" in tables[0]
    assert "PLS" in tables[0]
    assert "Accuracy" in tables[0]

    def fake_embed(texts: list[str]) -> list[list[float]]:
        return [[0.0, 1.0] for _ in texts]

    chunks = structure_document(block, embed_fn=fake_embed)
    assert any(c.section == "table" for c in chunks)


def test_keep_references_with_low_weight(monkeypatch):
    from config.settings import settings
    from src.rag.structure_chunk import structure_document

    monkeypatch.setattr(settings, "INGEST_STRUCTURED_CHUNK", True)
    monkeypatch.setattr(settings, "INGEST_DROP_REFERENCES", False)
    monkeypatch.setattr(settings, "INGEST_REFERENCE_WEIGHT", 0.35)

    def fake_embed(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    doc = """
Abstract
A short abstract about maize nitrogen.

References
[1] Alpha Beta, 2019, Journal of Remote Sensing.
""".strip()
    chunks = structure_document(doc, embed_fn=fake_embed)
    refs = [c for c in chunks if c.section == "reference"]
    assert refs
    assert all(c.weight == 0.35 for c in refs)
