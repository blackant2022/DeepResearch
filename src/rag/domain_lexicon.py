"""
src/rag/domain_lexicon.py — 领域词词典与查询向量增强

对专业术语做同义词/中英对照扩展，并可选将术语向量与查询向量混合，
提升口语问法与文献表述之间的对齐。
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from config.settings import PROJECT_ROOT, settings
from src.utils.logger import get_logger

log = get_logger("rag.domain_lexicon")

_DEFAULT_LEXICON: dict[str, list[str]] = {
    "高光谱": ["Hyperspectral", "hyperspectral imaging", "HSI", "光谱成像"],
    "氮含量": ["叶氮含量", "LNC", "leaf nitrogen content", "氮素", "氮浓度"],
    "反演": ["估测", "估算", "retrieval", "estimation", "定量反演"],
    "深度学习": ["Deep Learning", "DL", "神经网络", "CNN", "Transformer"],
    "支持向量回归": ["SVR", "Support Vector Regression", "支持向量机回归"],
    "偏最小二乘": ["PLSR", "PLS", "Partial Least Squares"],
    "随机森林": ["Random Forest", "RF"],
    "SHAP": ["可解释性", "特征贡献", "SHapley"],
    "无人机": ["UAV", "植保无人机", "drone"],
    "遥感": ["remote sensing", "RS"],
    "玉米": ["corn", "maize", "Zea mays"],
    "叶片": ["leaf", "foliar"],
}


@lru_cache(maxsize=1)
def load_lexicon() -> dict[str, list[str]]:
    """加载领域词典：默认内置 + 可选 JSON 覆盖/合并。"""
    lex = {k: list(v) for k, v in _DEFAULT_LEXICON.items()}
    path = Path(settings.DOMAIN_LEXICON_PATH)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, list):
                        merged = list(dict.fromkeys([*(lex.get(k) or []), *[str(x) for x in v]]))
                        lex[str(k)] = merged
                    elif isinstance(v, str):
                        lex[str(k)] = list(dict.fromkeys([*(lex.get(k) or []), v]))
            log.info(f"已加载领域词典 {path}，共 {len(lex)} 组术语")
        except Exception as e:  # noqa: BLE001
            log.warning(f"领域词典读取失败，使用内置词表: {e}")
    return lex


def match_terms(query: str) -> list[tuple[str, list[str]]]:
    """返回命中的 (规范词, 别名列表)。"""
    q = (query or "").strip()
    if not q:
        return []
    q_lower = q.lower()
    hits: list[tuple[str, list[str]]] = []
    for canonical, aliases in load_lexicon().items():
        keys = [canonical, *aliases]
        if any(k.lower() in q_lower or k in q for k in keys if k):
            hits.append((canonical, aliases))
    return hits


def expand_query_text(query: str) -> str:
    """
    文本侧增强：把命中术语的中英同义词追加到查询，供检索/改写使用。
    例：「氮怎么估」→ 追加「氮含量 LNC 反演 Hyperspectral …」中与命中相关的扩展。
    """
    q = (query or "").strip()
    if not q or not settings.DOMAIN_LEXICON_ENABLED:
        return q
    matched = match_terms(q)
    if not matched:
        # 弱匹配：氮/光谱等单字启发
        weak = []
        if "氮" in q and not any(m[0] == "氮含量" for m in matched):
            weak.append(("氮含量", load_lexicon().get("氮含量", [])))
        if ("光谱" in q or "高光" in q) and not any(m[0] == "高光谱" for m in matched):
            weak.append(("高光谱", load_lexicon().get("高光谱", [])))
        matched = weak
    if not matched:
        return q

    extras: list[str] = []
    for canonical, aliases in matched:
        extras.append(canonical)
        extras.extend(aliases[:4])
    # 去重且避免把整句再拼一遍
    seen = set(q.lower().split())
    add = []
    for t in extras:
        key = t.lower()
        if key not in seen and t not in q:
            add.append(t)
            seen.add(key)
    if not add:
        return q
    expanded = f"{q}（{' '.join(add[:12])}）"
    log.debug(f"领域词扩展: {q[:40]} → +{len(add)} terms")
    return expanded


def enhance_query_embedding(query: str, base_vec: list[float] | None = None) -> list[float]:
    """
    向量侧增强：查询向量与命中术语向量加权混合。
    blend = (1-α)*q + α*mean(term_vecs)
    """
    from src.memory.embedding import encode_docs, encode_query

    q = (query or "").strip()
    expanded = expand_query_text(q)
    q_vec = base_vec if base_vec is not None else encode_query(expanded)

    if not settings.DOMAIN_LEXICON_ENABLED or settings.DOMAIN_EMBED_BLEND <= 0:
        return q_vec

    matched = match_terms(q) or match_terms(expanded)
    if not matched:
        return q_vec

    term_texts = []
    for canonical, aliases in matched[:6]:
        term_texts.append(canonical)
        term_texts.extend(aliases[:2])
    term_texts = list(dict.fromkeys(term_texts))[:8]
    try:
        term_vecs = encode_docs(term_texts)
    except Exception as e:  # noqa: BLE001
        log.warning(f"术语向量化失败，跳过混合: {e}")
        return q_vec

    if not term_vecs:
        return q_vec

    dim = len(q_vec)
    mean = [0.0] * dim
    for v in term_vecs:
        if len(v) != dim:
            continue
        for i, x in enumerate(v):
            mean[i] += x
    n = max(1, len(term_vecs))
    mean = [x / n for x in mean]
    alpha = float(settings.DOMAIN_EMBED_BLEND)
    alpha = max(0.0, min(0.5, alpha))
    blended = [(1 - alpha) * a + alpha * b for a, b in zip(q_vec, mean)]
    # L2 归一化，保持与 cosine 空间一致
    norm = sum(x * x for x in blended) ** 0.5
    if norm > 1e-12:
        blended = [x / norm for x in blended]
    return blended
