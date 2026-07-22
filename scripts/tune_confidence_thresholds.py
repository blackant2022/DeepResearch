"""
scripts/tune_confidence_thresholds.py — 双层 RAG 门禁网格搜索

用三套代理测试集（域内 / 域外 / 边界）扫：
  RETRIEVAL_THRESHOLD × ANSWER_CONFIDENCE_THRESHOLD

无段落级标注时的目标：
  - 域内：期望有命中 + 关键词/文件名代理召回 + 回答门禁通过
  - 域外：期望回答门禁拒绝（fallback）
  - 边界：不强制，仅作稳定性观察

用法：
  set PYTHONPATH=项目根
  python scripts/tune_confidence_thresholds.py
  python scripts/tune_confidence_thresholds.py --apply-best
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TUNE_DIR = ROOT / "data" / "eval" / "threshold_tune"
OUT_PATH = ROOT / "data" / "eval" / "threshold_tune_report.json"

# 检索门禁候选、回答门禁候选
RET_GRID = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
ANS_GRID = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def _load_cases() -> list[dict]:
    cases: list[dict] = []
    for name in ("in_domain.json", "ood.json", "borderline.json"):
        path = TUNE_DIR / name
        cases.extend(json.loads(path.read_text(encoding="utf-8")))
    return cases


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


def _score_pair(
    cases: list[dict],
    *,
    ret_thr: float,
    ans_thr: float,
) -> dict:
    from config.settings import settings
    from src.rag.confidence import confidence_report
    from src.rag.retriever import retriever

    settings.RETRIEVAL_THRESHOLD = ret_thr
    settings.ANSWER_CONFIDENCE_THRESHOLD = ans_thr

    in_pass = in_total = 0
    in_recall = in_recall_n = 0
    in_hits = in_hits_n = 0
    ood_reject = ood_total = 0
    bd_pass = bd_total = 0
    latencies: list[float] = []

    for case in cases:
        q = case["question"]
        t0 = time.perf_counter()
        hits = retriever.search(q, k=3)
        latencies.append(time.perf_counter() - t0)
        report = confidence_report(hits, query=q)
        passed = bool(report.get("pass"))
        s = case.get("set")

        if s == "in_domain":
            in_total += 1
            if case.get("expect_hits"):
                in_hits_n += 1
                if hits:
                    in_hits += 1
            if case.get("expect_answer_pass") is True and passed:
                in_pass += 1
            rp = _recall_proxy(case, hits)
            if rp is not None:
                in_recall_n += 1
                if rp:
                    in_recall += 1
        elif s == "ood":
            ood_total += 1
            # 期望拒绝：pass=False
            if case.get("expect_answer_pass") is False and not passed:
                ood_reject += 1
        else:
            bd_total += 1
            if passed:
                bd_pass += 1

    in_pass_rate = in_pass / in_total if in_total else 0.0
    in_recall_rate = in_recall / in_recall_n if in_recall_n else 0.0
    in_hit_rate = in_hits / in_hits_n if in_hits_n else 0.0
    ood_reject_rate = ood_reject / ood_total if ood_total else 0.0
    bd_pass_rate = bd_pass / bd_total if bd_total else 0.0

    # 综合分：域内通过与召回、域外拒绝为主；边界轻微参考
    # 过严导致域内无命中会在 in_hit_rate / in_pass_rate 上惩罚
    score = (
        0.35 * in_pass_rate
        + 0.30 * in_recall_rate
        + 0.15 * in_hit_rate
        + 0.35 * ood_reject_rate
        + 0.05 * bd_pass_rate
    )
    # 轻度归一：权重和>1，再压到约 0~1
    score = round(score / 1.20, 4)

    return {
        "retrieval_threshold": ret_thr,
        "answer_confidence_threshold": ans_thr,
        "score": score,
        "in_domain_pass_rate": round(in_pass_rate, 4),
        "in_domain_recall_proxy": round(in_recall_rate, 4),
        "in_domain_hit_rate": round(in_hit_rate, 4),
        "ood_reject_rate": round(ood_reject_rate, 4),
        "borderline_pass_rate": round(bd_pass_rate, 4),
        "mean_latency_s": round(statistics.mean(latencies), 4) if latencies else 0.0,
        "n_cases": len(cases),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply-best",
        action="store_true",
        help="把最优阈值写入 data/eval/threshold_tune_best.env（不改 .env）",
    )
    parser.add_argument("--ret", nargs="*", type=float, default=None)
    parser.add_argument("--ans", nargs="*", type=float, default=None)
    args = parser.parse_args()

    ret_grid = args.ret or RET_GRID
    ans_grid = args.ans or ANS_GRID
    cases = _load_cases()

    from src.memory.embedding import encode_query
    from src.rag.retriever import retriever

    print(f"预热… kb={retriever.count()} cases={len(cases)}")
    encode_query("warmup")

    results = []
    total = len(ret_grid) * len(ans_grid)
    i = 0
    for ret in ret_grid:
        for ans in ans_grid:
            i += 1
            row = _score_pair(cases, ret_thr=ret, ans_thr=ans)
            results.append(row)
            print(
                f"[{i}/{total}] ret={ret:.2f} ans={ans:.2f} "
                f"score={row['score']:.3f} "
                f"in_pass={row['in_domain_pass_rate']:.2f} "
                f"recall={row['in_domain_recall_proxy']:.2f} "
                f"ood_rej={row['ood_reject_rate']:.2f}"
            )

    results.sort(key=lambda r: (-r["score"], -r["ood_reject_rate"], -r["in_domain_pass_rate"]))
    best = results[0]
    # 在接近最优的方案里，优先 ood 拒绝高且域内通过不太差的
    near = [r for r in results if r["score"] >= best["score"] - 0.02]
    near.sort(
        key=lambda r: (
            -r["ood_reject_rate"],
            -r["in_domain_pass_rate"],
            -r["in_domain_recall_proxy"],
            abs(r["retrieval_threshold"] - 0.55) + abs(r["answer_confidence_threshold"] - 0.45),
        )
    )
    recommended = near[0]

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "无段落标注；用域内代理召回 + 域外拒答率调参，非真正 Recall@K",
        "grids": {"retrieval": ret_grid, "answer_confidence": ans_grid},
        "best_by_score": best,
        "recommended": recommended,
        "top10": results[:10],
        "all": results,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== 推荐 ===")
    print(json.dumps(recommended, ensure_ascii=False, indent=2))
    print(f"\n报告已写: {OUT_PATH}")

    if args.apply_best:
        env_path = TUNE_DIR / "recommended_thresholds.env"
        env_path.write_text(
            (
                f"# 由 tune_confidence_thresholds.py 生成，可手工合并进 .env\n"
                f"RETRIEVAL_THRESHOLD={recommended['retrieval_threshold']}\n"
                f"ANSWER_CONFIDENCE_THRESHOLD={recommended['answer_confidence_threshold']}\n"
            ),
            encoding="utf-8",
        )
        print(f"已写出: {env_path}")


if __name__ == "__main__":
    main()
