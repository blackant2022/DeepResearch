"""
scripts/compare_rerank_ab.py — 有 Rerank vs 无 Rerank 对照实验

同一批查询、同一知识库、同一 TOP_K / recall_k，仅切换 RERANK_ENABLED。

指标（无段落标注时的代理）：
  - 延迟 mean / p50
  - 域内召回代理（关键词 + 期望文件名）
  - Top1 向量分、平均分
  - Top1 结果是否因 Rerank 发生变化（排序改变率）
  - 域外低置信拦截率

用法：
  set PYTHONPATH=项目根
  python scripts/compare_rerank_ab.py
  python scripts/compare_rerank_ab.py --set data/eval/latency_relevance_set.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SET = ROOT / "data" / "eval" / "latency_relevance_set.json"
OUT_JSON = ROOT / "data" / "eval" / "rerank_ab_report.json"
OUT_MD = ROOT / "data" / "eval" / "rerank_ab_report.md"


def _kw_hit(text: str, keywords: list[str]) -> bool:
    if not keywords:
        return True
    t = text.lower()
    return any(k.lower() in t for k in keywords)


def _fn_hit(filename: str, prefers: list[str]) -> bool:
    if not prefers:
        return True
    fn = filename.lower()
    return any(p.lower() in fn for p in prefers)


def _recall_proxy(case: dict, hits: list[dict]) -> bool | None:
    must = case.get("must_keywords") or []
    prefer = case.get("prefer_filename") or []
    if not must and not prefer:
        return None
    blob = "\n".join(str(h.get("content") or "") for h in hits)
    files = [str(h.get("filename") or "") for h in hits]
    parts = []
    if must:
        parts.append(_kw_hit(blob, must))
    if prefer:
        parts.append(any(_fn_hit(f, prefer) for f in files))
    return all(parts) if parts else None


def _run_arm(cases: list[dict], *, rerank: bool, k: int) -> list[dict]:
    from config.settings import settings
    from src.rag.confidence import confidence_report
    from src.rag.retriever import retriever

    settings.RERANK_ENABLED = rerank
    rows = []
    for case in cases:
        if case.get("skip_retrieval"):
            continue
        q = case["question"]
        t0 = time.perf_counter()
        hits = retriever.search(q, k=k)
        elapsed = time.perf_counter() - t0
        conf = confidence_report(hits, query=q)
        scores = [float(h.get("vector_score") or h.get("score") or 0) for h in hits]
        files = [str(h.get("filename") or "") for h in hits]
        top_content = (hits[0].get("content") or "")[:120] if hits else ""
        rows.append({
            "id": case["id"],
            "question": q,
            "category": case.get("category"),
            "latency_s": round(elapsed, 4),
            "hits": len(hits),
            "top_score": round(max(scores), 4) if scores else 0.0,
            "avg_score": round(statistics.mean(scores), 4) if scores else 0.0,
            "confidence": conf.get("confidence"),
            "low_confidence": conf.get("fallback"),
            "ood_query": conf.get("ood_query"),
            "sources": files,
            "top1_file": files[0] if files else "",
            "top1_snippet": top_content,
            "recall_proxy": _recall_proxy(case, hits),
            "expect_low_confidence": bool(case.get("expect_low_confidence")),
        })
    return rows


def _summarize(rows: list[dict]) -> dict:
    lat = [r["latency_s"] for r in rows]
    proxy = [r["recall_proxy"] for r in rows if r.get("recall_proxy") is not None]
    in_dom = [
        r for r in rows
        if r.get("category") not in ("ood", "chitchat") and r.get("recall_proxy") is not None
    ]
    in_proxy = [r["recall_proxy"] for r in in_dom]
    ood = [r for r in rows if r.get("expect_low_confidence")]
    ood_ok = [r for r in ood if r.get("low_confidence")]
    tops = [r["top_score"] for r in rows if r.get("hits")]
    return {
        "n": len(rows),
        "latency_mean_s": round(statistics.mean(lat), 4) if lat else None,
        "latency_p50_s": round(statistics.median(lat), 4) if lat else None,
        "latency_max_s": round(max(lat), 4) if lat else None,
        "top_score_mean": round(statistics.mean(tops), 4) if tops else None,
        "recall_proxy_all": round(sum(1 for x in proxy if x) / len(proxy), 4) if proxy else None,
        "recall_proxy_in_domain": (
            round(sum(1 for x in in_proxy if x) / len(in_proxy), 4) if in_proxy else None
        ),
        "n_in_domain_labeled": len(in_proxy),
        "ood_reject_rate": round(len(ood_ok) / len(ood), 4) if ood else None,
        "n_ood": len(ood),
    }


def _pairwise(with_r: list[dict], no_r: list[dict]) -> dict:
    by_id_w = {r["id"]: r for r in with_r}
    by_id_n = {r["id"]: r for r in no_r}
    ids = [i for i in by_id_w if i in by_id_n]
    top1_changed = 0
    order_changed = 0
    deltas = []
    changed_cases = []
    for i in ids:
        a, b = by_id_w[i], by_id_n[i]
        if a["top1_file"] != b["top1_file"] or a["top1_snippet"] != b["top1_snippet"]:
            top1_changed += 1
            changed_cases.append({
                "id": i,
                "question": a["question"][:60],
                "with_rerank_top1": a["top1_file"],
                "no_rerank_top1": b["top1_file"],
                "with_recall": a["recall_proxy"],
                "no_recall": b["recall_proxy"],
            })
        if a["sources"] != b["sources"]:
            order_changed += 1
        deltas.append(a["latency_s"] - b["latency_s"])

    # 仅看域内：rerank 是否修好/弄坏代理召回
    fixed = missed = same_ok = same_bad = 0
    for i in ids:
        a, b = by_id_w[i], by_id_n[i]
        if a.get("recall_proxy") is None:
            continue
        if a["recall_proxy"] and not b["recall_proxy"]:
            fixed += 1
        elif (not a["recall_proxy"]) and b["recall_proxy"]:
            missed += 1
        elif a["recall_proxy"] and b["recall_proxy"]:
            same_ok += 1
        else:
            same_bad += 1

    return {
        "n_paired": len(ids),
        "top1_change_rate": round(top1_changed / len(ids), 4) if ids else 0.0,
        "list_change_rate": round(order_changed / len(ids), 4) if ids else 0.0,
        "latency_delta_mean_s": round(statistics.mean(deltas), 4) if deltas else 0.0,
        "rerank_fixed_recall": fixed,
        "rerank_broke_recall": missed,
        "both_ok": same_ok,
        "both_bad": same_bad,
        "changed_top1_examples": changed_cases[:12],
    }


def _to_md(report: dict) -> str:
    w = report["with_rerank"]["summary"]
    n = report["no_rerank"]["summary"]
    p = report["pairwise"]
    lines = [
        "# Rerank A/B 对照实验报告",
        "",
        f"- 生成时间：{report['generated_at']}",
        f"- 评测集：`{report['set_path']}`（n={report['n_cases']}，跳过 skip_retrieval）",
        f"- TOP_K={report['top_k']}，RETRIEVAL_RECALL_K={report['recall_k']}",
        f"- 说明：无段落级标注，召回为关键词/文件名**代理指标**，非真正 Recall@K",
        "",
        "## 汇总对比",
        "",
        "| 指标 | 有 Rerank | 无 Rerank | 差值(有-无) |",
        "|------|-----------|-----------|-------------|",
        f"| 延迟 mean (s) | {w['latency_mean_s']} | {n['latency_mean_s']} | "
        f"{round((w['latency_mean_s'] or 0) - (n['latency_mean_s'] or 0), 4)} |",
        f"| 延迟 p50 (s) | {w['latency_p50_s']} | {n['latency_p50_s']} | "
        f"{round((w['latency_p50_s'] or 0) - (n['latency_p50_s'] or 0), 4)} |",
        f"| 域内召回代理 | {w['recall_proxy_in_domain']} | {n['recall_proxy_in_domain']} | "
        f"{round((w['recall_proxy_in_domain'] or 0) - (n['recall_proxy_in_domain'] or 0), 4)} |",
        f"| 全量召回代理 | {w['recall_proxy_all']} | {n['recall_proxy_all']} | "
        f"{round((w['recall_proxy_all'] or 0) - (n['recall_proxy_all'] or 0), 4)} |",
        f"| Top1 向量分均值 | {w['top_score_mean']} | {n['top_score_mean']} | "
        f"{round((w['top_score_mean'] or 0) - (n['top_score_mean'] or 0), 4)} |",
        f"| 域外拒答率 | {w['ood_reject_rate']} | {n['ood_reject_rate']} | "
        f"{round((w['ood_reject_rate'] or 0) - (n['ood_reject_rate'] or 0), 4)} |",
        "",
        "## 排序改变（Rerank 是否真的改结果）",
        "",
        f"- Top1 改变率：**{p['top1_change_rate']:.1%}**",
        f"- Top-K 列表改变率：**{p['list_change_rate']:.1%}**",
        f"- Rerank 修好代理召回：{p['rerank_fixed_recall']} 条",
        f"- Rerank 弄坏代理召回：{p['rerank_broke_recall']} 条",
        f"- 两边都对 / 都错：{p['both_ok']} / {p['both_bad']}",
        "",
        "## 结论（自动）",
        "",
    ]
    # auto conclusion
    lat_cost = (w["latency_mean_s"] or 0) - (n["latency_mean_s"] or 0)
    rec_gain = (w["recall_proxy_in_domain"] or 0) - (n["recall_proxy_in_domain"] or 0)
    if p["top1_change_rate"] < 0.05 and abs(rec_gain) < 0.02:
        lines.append(
            "- 本次集上 Rerank **几乎不改变 Top1/代理召回**，主要带来延迟开销；"
            "可考虑默认关 CE，或仅对低分宽召回启用。"
        )
    elif rec_gain > 0.02:
        lines.append(
            f"- Rerank 提升域内代理召回约 **{rec_gain:.1%}**，额外延迟约 **{lat_cost:.3f}s/query**；"
            "质量优先场景建议保留。"
        )
    elif rec_gain < -0.02:
        lines.append(
            f"- Rerank 使域内代理召回下降约 **{abs(rec_gain):.1%}**，需检查 CE 模型与中文领域是否匹配。"
        )
    else:
        lines.append(
            f"- 代理召回接近持平（Δ={rec_gain:+.1%}），延迟差约 **{lat_cost:.3f}s**；"
            "按延迟预算决定是否开启。"
        )
    if p["changed_top1_examples"]:
        lines.extend(["", "## Top1 变化样例", ""])
        for ex in p["changed_top1_examples"][:8]:
            lines.append(
                f"- `{ex['id']}` {ex['question']}…  \n"
                f"  有Rerank→ `{ex['with_rerank_top1']}`；无→ `{ex['no_rerank_top1']}`"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", type=str, default=str(DEFAULT_SET))
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    set_path = Path(args.set)
    cases = json.loads(set_path.read_text(encoding="utf-8"))
    usable = [c for c in cases if not c.get("skip_retrieval")]

    from config.settings import settings
    from src.memory.embedding import encode_query
    from src.rag.retriever import retriever

    print(f"加载 {len(usable)} 条（跳过 skip_retrieval），kb={retriever.count()}")
    print("预热 embedding / rerank…")
    encode_query("warmup")
    # 预热 CE：先开 rerank 搜一次
    settings.RERANK_ENABLED = True
    retriever.search("高光谱氮含量", k=args.k)

    print("\n=== Arm A: RERANK_ENABLED=true ===")
    with_rows = _run_arm(usable, rerank=True, k=args.k)
    print(f"  done n={len(with_rows)}")

    print("\n=== Arm B: RERANK_ENABLED=false ===")
    no_rows = _run_arm(usable, rerank=False, k=args.k)
    print(f"  done n={len(no_rows)}")

    # 恢复默认，避免污染后续进程（同进程内）
    settings.RERANK_ENABLED = True

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "set_path": str(set_path),
        "n_cases": len(usable),
        "top_k": args.k,
        "recall_k": settings.RETRIEVAL_RECALL_K,
        "with_rerank": {"summary": _summarize(with_rows), "rows": with_rows},
        "no_rerank": {"summary": _summarize(no_rows), "rows": no_rows},
        "pairwise": _pairwise(with_rows, no_rows),
    }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(_to_md(report), encoding="utf-8")

    print("\n======== 对比摘要 ========")
    print(_to_md(report))
    print(f"JSON: {OUT_JSON}")
    print(f"Markdown: {OUT_MD}")


if __name__ == "__main__":
    main()
