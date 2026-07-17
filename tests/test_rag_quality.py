"""tests/test_rag_quality.py — 重排 / 领域词典 / 置信度门禁"""
from __future__ import annotations

from src.rag.confidence import (
    FALLBACK_MESSAGE,
    apply_answer_confidence_gate,
    confidence_report,
    hits_confidence,
    is_low_confidence,
)
from src.rag.domain_lexicon import expand_query_text, match_terms
from src.rag.rerank import rerank_hits


def test_domain_lexicon_expands_nitrogen_terms(monkeypatch):
    from config import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "DOMAIN_LEXICON_ENABLED", True)
    # 清缓存
    from src.rag import domain_lexicon as dl

    dl.load_lexicon.cache_clear()
    expanded = expand_query_text("玉米叶片氮含量怎么估？")
    assert "LNC" in expanded or "氮" in expanded
    assert match_terms("高光谱反演")


def test_rerank_filters_low_scores(monkeypatch):
    from config import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "RERANK_ENABLED", True)
    monkeypatch.setattr(settings_mod.settings, "RERANK_BACKEND", "dense")
    monkeypatch.setattr(settings_mod.settings, "RERANK_MIN_SCORE", 0.99)
    monkeypatch.setattr(settings_mod.settings, "TOP_K", 2)

    hits = [
        {"content": "完全无关的天气预报", "filename": "a.pdf", "score": 0.9},
        {"content": "高光谱氮含量反演方法", "filename": "b.pdf", "score": 0.8},
    ]

    def fake_encode_query(q: str):
        return [1.0, 0.0]

    def fake_encode_docs(texts: list[str]):
        out = []
        for t in texts:
            if "氮" in t or "高光谱" in t:
                out.append([1.0, 0.0])
            else:
                out.append([0.0, 1.0])
        return out

    monkeypatch.setattr("src.memory.embedding.encode_query", fake_encode_query)
    monkeypatch.setattr("src.memory.embedding.encode_docs", fake_encode_docs)
    # min_score 极高时仍至少保留 1 条供置信度判断
    ranked = rerank_hits("氮含量反演", hits, top_n=2, min_score=0.99)
    assert len(ranked) == 1
    assert "氮" in ranked[0]["content"] or ranked[0]["score"] >= 0.99


def test_confidence_gate_triggers_fallback(monkeypatch):
    from config import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "ANSWER_CONFIDENCE_ENABLED", True)
    monkeypatch.setattr(settings_mod.settings, "ANSWER_CONFIDENCE_THRESHOLD", 0.8)

    weak = [{"content": "x", "score": 0.2}]
    assert is_low_confidence(weak)
    assert hits_confidence(weak) < 0.8
    ans, report = apply_answer_confidence_gate("胡编答案", weak, used_knowledge=True)
    assert report["fallback"] is True
    assert "置信度不足" in ans
    assert ans == FALLBACK_MESSAGE


def test_confidence_pass_keeps_answer(monkeypatch):
    from config import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "ANSWER_CONFIDENCE_ENABLED", True)
    monkeypatch.setattr(settings_mod.settings, "ANSWER_CONFIDENCE_THRESHOLD", 0.4)

    strong = [{"content": "证据", "score": 0.9}, {"content": "证据2", "score": 0.85}]
    ans, report = apply_answer_confidence_gate("可靠答案", strong, used_knowledge=True)
    assert report["pass"] is True
    assert ans == "可靠答案"
    assert confidence_report(strong)["confidence"] >= 0.4
