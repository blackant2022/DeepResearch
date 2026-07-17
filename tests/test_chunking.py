"""tests/test_chunking.py — 语义分块 + 滑动窗口（无需真实 Embedding 模型）"""
from __future__ import annotations

from src.rag.chunking import (
    _cosine,
    _dynamic_overlap,
    hybrid_chunk_text,
    sliding_window_chunks,
    split_sentences,
)


def test_split_sentences_zh():
    text = "第一句关于氮含量。第二句讲高光谱。第三句是结论！"
    sents = split_sentences(text)
    assert len(sents) >= 3
    assert "氮含量" in sents[0]


def test_dynamic_overlap_grows_with_ratio():
    base = _dynamic_overlap(500, 60, 0.12, 500)
    assert base >= 60
    # 更高比例应不小于基础
    higher = _dynamic_overlap(500, 60, 0.3, 500)
    assert higher >= base
    # 不超过半窗
    assert higher <= 250


def test_sliding_window_respects_overlap():
    text = "A" * 1200
    chunks = sliding_window_chunks(
        text, chunk_size=500, chunk_overlap=60, overlap_ratio=0.12, min_chunk=40
    )
    assert len(chunks) >= 3
    assert all(len(c) <= 500 for c in chunks[:-1])
    # 相邻块应有重叠（字符级）
    assert chunks[0][-40:] == chunks[1][:40] or chunks[0][-50:] in chunks[1]


def test_hybrid_semantic_break_with_fake_embeddings():
    """相邻句向量正交 → 应断成多段；同向 → 合并。"""
    sents = [
        "玉米叶片氮含量反演方法研究。",
        "本文提出基于高光谱的估测模型。",
        "天气预报显示明天有雨。",
        "请携带雨伞出门。",
    ]
    text = "".join(sents)

    # 前两句同向，后两句同向，两组正交 → 两个语义段
    def fake_embed(texts: list[str]) -> list[list[float]]:
        vecs = []
        for t in texts:
            if "氮" in t or "高光谱" in t:
                vecs.append([1.0, 0.0, 0.0])
            else:
                vecs.append([0.0, 1.0, 0.0])
        return vecs

    chunks = hybrid_chunk_text(
        text,
        embed_fn=fake_embed,
        chunk_size=500,
        chunk_overlap=60,
        similarity_threshold=0.5,
        overlap_ratio=0.12,
        min_chunk=10,
    )
    assert len(chunks) >= 2
    assert any("氮" in c for c in chunks)
    assert any("雨" in c for c in chunks)
    # 主题不应严重串段：含氮的块不应主要谈雨
    nitrogen = next(c for c in chunks if "氮" in c)
    assert "雨伞" not in nitrogen


def test_hybrid_long_para_uses_sliding():
    long_topic = ("高光谱遥感用于估算作物氮含量。" * 40)  # 远超 500
    weather = "今日晴转多云，适宜田间作业。"

    def fake_embed(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "高光谱" in t or "氮" in t else [0.0, 1.0] for t in texts]

    chunks = hybrid_chunk_text(
        long_topic + weather,
        embed_fn=fake_embed,
        chunk_size=200,
        chunk_overlap=40,
        similarity_threshold=0.5,
        overlap_ratio=0.15,
        min_chunk=20,
    )
    assert len(chunks) >= 2
    assert all(len(c) <= 220 for c in chunks)  # 允许尾部轻微合并余量


def test_cosine_basic():
    assert _cosine([1, 0], [1, 0]) > 0.99
    assert abs(_cosine([1, 0], [0, 1])) < 0.01
