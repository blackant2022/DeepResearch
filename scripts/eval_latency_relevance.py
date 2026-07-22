"""
scripts/eval_latency_relevance.py — 批量评测检索延迟与相关度/召回代理指标

用法：
  set PYTHONPATH=项目根
  python scripts/eval_latency_relevance.py
  python scripts/eval_latency_relevance.py --e2e 5   # 额外跑 N 条端到端 Agent

说明：
  - 无人工标注段落时，用「关键词命中率 + 期望文件名命中」作为召回代理
  - 相关度用检索 score / 置信度
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SET_PATH = ROOT / "data" / "eval" / "latency_relevance_set.json"
OUT_PATH = ROOT / "data" / "eval" / "latency_relevance_report.json"


def _kw_hit(text: str, keywords: list[str]) -> bool:
    if not keywords:
        return True
    t = text.lower()
    return any(k.lower() in t for k in keywords)


def _filename_hit(filename: str, prefers: list[str]) -> bool:
    if not prefers:
        return True
    fn = filename.lower()
    return any(p.lower() in fn for p in prefers)


def eval_retrieval(cases: list[dict]) -> list[dict]:
    from src.memory.embedding import encode_query
    from src.rag.confidence import confidence_report, is_low_confidence
    from src.rag.retriever import retriever

    print("预热嵌入…")
    t0 = time.perf_counter()
    encode_query("warmup")
    warm_s = time.perf_counter() - t0
    print(f"  warmup={warm_s:.2f}s  kb_chunks={retriever.count()}")

    rows: list[dict] = []
    for i, case in enumerate(cases, 1):
        if case.get("skip_retrieval"):
            rows.append({
                **{k: case.get(k) for k in ("id", "question", "category")},
                "skipped": True,
            })
            continue

        q = case["question"]
        t0 = time.perf_counter()
        hits = retriever.search(q, k=3)
        elapsed = time.perf_counter() - t0
        conf = confidence_report(hits, query=q)
        scores = [float(h.get("vector_score") or h.get("score") or 0) for h in hits]
        blob = "\n".join(str(h.get("content") or "") for h in hits)
        files = [str(h.get("filename") or "") for h in hits]

        must = case.get("must_keywords") or []
        prefer = case.get("prefer_filename") or []
        kw_ok = _kw_hit(blob, must) if must else None
        fn_ok = any(_filename_hit(f, prefer) for f in files) if prefer else None

        # 召回代理：有 must_keywords 时要求正文命中；有 prefer_filename 时要求文件名命中
        recall_parts = []
        if must:
            recall_parts.append(bool(kw_ok))
        if prefer:
            recall_parts.append(bool(fn_ok))
        recall_proxy = all(recall_parts) if recall_parts else None

        row = {
            "id": case["id"],
            "question": q,
            "category": case.get("category"),
            "latency_s": round(elapsed, 3),
            "hits": len(hits),
            "scores": [round(s, 4) for s in scores],
            "avg_score": round(statistics.mean(scores), 4) if scores else 0.0,
            "top_score": round(max(scores), 4) if scores else 0.0,
            "confidence": conf.get("confidence"),
            "low_confidence": conf.get("fallback"),
            "ood_query": conf.get("ood_query"),
            "n_unique_files": len(set(files)),
            "sources": files,
            "keyword_hit": kw_ok,
            "filename_hit": fn_ok,
            "recall_proxy": recall_proxy,
            "expect_low_confidence": case.get("expect_low_confidence", False),
            "ood_gate_ok": (
                bool(conf.get("fallback"))
                if case.get("expect_low_confidence")
                else (not conf.get("fallback") if scores else None)
            ),
        }
        rows.append(row)
        flag = "OK" if recall_proxy is not False else "MISS"
        print(
            f"[{i:02d}/{len(cases)}] {case['id']} {elapsed:.2f}s "
            f"top={row['top_score']:.3f} conf={row['confidence']} {flag}"
        )
    return rows


def eval_e2e(cases: list[dict], n: int) -> list[dict]:
    from src.orchestrator.graph import run_agent

    # 选有代表性的：闲聊、文献、含糊、OOD、深研各尽量覆盖
    prefer_ids = ["q39", "q01", "q23", "q36", "q40", "q22", "q04", "q12"]
    by_id = {c["id"]: c for c in cases}
    chosen = [by_id[i] for i in prefer_ids if i in by_id][:n]
    while len(chosen) < n and len(chosen) < len(cases):
        for c in cases:
            if c not in chosen:
                chosen.append(c)
            if len(chosen) >= n:
                break

    rows = []
    for i, case in enumerate(chosen, 1):
        q = case["question"]
        print(f"E2E [{i}/{len(chosen)}] {case['id']} {q[:40]}…")
        t0 = time.perf_counter()
        try:
            result = run_agent(q)
            elapsed = time.perf_counter() - t0
            rows.append({
                "id": case["id"],
                "question": q,
                "latency_s": round(elapsed, 3),
                "route": result.get("route"),
                "react_iterations": result.get("react_iterations"),
                "tool_calls": (result.get("metrics") or {}).get("tool_calls"),
                "answer_chars": len(result.get("answer") or ""),
                "primary": (result.get("metrics") or {}).get("primary_display"),
                "ok": True,
            })
            print(f"  → {elapsed:.1f}s route={result.get('route')}")
        except Exception as e:  # noqa: BLE001
            rows.append({
                "id": case["id"],
                "question": q,
                "error": str(e),
                "latency_s": round(time.perf_counter() - t0, 3),
                "ok": False,
            })
            print(f"  → FAIL {e}")
    return rows


def summarize(retrieval_rows: list[dict], e2e_rows: list[dict] | None) -> dict:
    scored = [r for r in retrieval_rows if not r.get("skipped") and r.get("hits", 0) >= 0]
    lat = [r["latency_s"] for r in scored]
    avg_scores = [r["avg_score"] for r in scored if r.get("hits")]
    top_scores = [r["top_score"] for r in scored if r.get("hits")]
    proxy = [r["recall_proxy"] for r in scored if r.get("recall_proxy") is not None]
    ood = [r for r in scored if r.get("expect_low_confidence")]
    ood_ok = [r for r in ood if r.get("ood_gate_ok")]

    in_domain = [r for r in scored if r.get("category") not in ("ood", "chitchat")]
    in_domain_proxy = [r["recall_proxy"] for r in in_domain if r.get("recall_proxy") is not None]

    summary = {
        "n_retrieval": len(scored),
        "latency_s": {
            "mean": round(statistics.mean(lat), 3) if lat else None,
            "p50": round(statistics.median(lat), 3) if lat else None,
            "max": round(max(lat), 3) if lat else None,
            "min": round(min(lat), 3) if lat else None,
        },
        "relevance": {
            "avg_score_mean": round(statistics.mean(avg_scores), 4) if avg_scores else None,
            "top_score_mean": round(statistics.mean(top_scores), 4) if top_scores else None,
            "frac_top_ge_0_55": round(
                sum(1 for s in top_scores if s >= 0.55) / len(top_scores), 3
            ) if top_scores else None,
            "frac_top_ge_0_70": round(
                sum(1 for s in top_scores if s >= 0.70) / len(top_scores), 3
            ) if top_scores else None,
        },
        "recall_proxy": {
            "all_labeled": round(sum(1 for x in proxy if x) / len(proxy), 3) if proxy else None,
            "n_labeled": len(proxy),
            "in_domain": round(
                sum(1 for x in in_domain_proxy if x) / len(in_domain_proxy), 3
            ) if in_domain_proxy else None,
            "n_in_domain": len(in_domain_proxy),
        },
        "ood_gate": {
            "n": len(ood),
            "correctly_low_conf": len(ood_ok),
            "rate": round(len(ood_ok) / len(ood), 3) if ood else None,
        },
    }
    if e2e_rows:
        ok = [r for r in e2e_rows if r.get("ok")]
        elat = [r["latency_s"] for r in ok]
        summary["e2e"] = {
            "n": len(e2e_rows),
            "ok": len(ok),
            "latency_mean_s": round(statistics.mean(elat), 3) if elat else None,
            "latency_max_s": round(max(elat), 3) if elat else None,
            "rows": e2e_rows,
        }
    return summary


def recommend(summary: dict) -> list[str]:
    tips: list[str] = []
    lat = (summary.get("latency_s") or {}).get("mean") or 0
    if lat > 3:
        tips.append(
            f"检索均耗时 {lat}s 偏高：优先 RERANK_BACKEND=dense 或 RERANK_ENABLED=false，"
            f"RETRIEVAL_RECALL_K 从 16 降到 8，并做启动预热。"
        )
    rp = (summary.get("recall_proxy") or {}).get("in_domain")
    if rp is not None and rp < 0.75:
        tips.append(
            f"领域问召回代理仅 {rp:.0%}：加强领域词典、检查切分是否切断方法段、"
            f"对失败 case 做 Query Rewrite 或提高宽召回后再精排。"
        )
    elif rp is not None and rp >= 0.85:
        tips.append(f"领域召回代理 {rp:.0%} 尚可，优先优化延迟而非盲目加大模型。")

    frac55 = (summary.get("relevance") or {}).get("frac_top_ge_0_55")
    if frac55 is not None and frac55 < 0.8:
        tips.append(
            f"Top 分≥0.55 仅占 {frac55:.0%}：检查重排分数尺度是否与门禁不匹配，"
            f"或下调 RETRIEVAL_THRESHOLD / 换 dense 相关度。"
        )

    ood_rate = (summary.get("ood_gate") or {}).get("rate")
    if ood_rate is not None and ood_rate < 0.67:
        tips.append(
            f"域外问题低置信拦截率 {ood_rate:.0%} 不足：提高 ANSWER_CONFIDENCE_THRESHOLD "
            f"或对明显闲聊/无关路由到 chat/拒答。"
        )
    elif ood_rate is not None and ood_rate >= 1.0:
        tips.append("域外低置信门禁表现正常，保持即可。")

    e2e = summary.get("e2e") or {}
    e2e_lat = e2e.get("latency_mean_s")
    if e2e_lat and e2e_lat > 25:
        tips.append(
            f"端到端均耗时 {e2e_lat}s：限制 MAX_REACT_ITERATIONS=2、禁止同问重复 knowledge_search、"
            f"工具默认 k=3。"
        )
    if not tips:
        tips.append("当前指标较均衡，建议补人工标注段落以计算真 Recall@K，再微调阈值。")
    return tips


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2e", type=int, default=5, help="端到端 Agent 条数，0 表示跳过")
    parser.add_argument("--set", type=str, default=str(SET_PATH))
    args = parser.parse_args()

    cases = json.loads(Path(args.set).read_text(encoding="utf-8"))
    print(f"加载 {len(cases)} 条评测问")

    retrieval_rows = eval_retrieval(cases)
    e2e_rows = eval_e2e(cases, args.e2e) if args.e2e > 0 else None
    summary = summarize(retrieval_rows, e2e_rows)
    summary["recommendations"] = recommend(summary)

    report = {
        "summary": summary,
        "retrieval_rows": retrieval_rows,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n======== SUMMARY ========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n改进建议：")
    for t in summary["recommendations"]:
        print(f"- {t}")
    print(f"\n完整报告: {OUT_PATH}")


if __name__ == "__main__":
    main()
