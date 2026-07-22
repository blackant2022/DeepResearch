"""scripts/latency_probe.py — 实测端到端回答耗时（需 .env 中 DEEPSEEK_API_KEY）

用法：
  python scripts/latency_probe.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _warm_embedding() -> float:
    from src.memory.embedding import encode_query

    t0 = time.perf_counter()
    encode_query("预热嵌入模型")
    return time.perf_counter() - t0


def _bench_retrieval(query: str) -> dict:
    from src.rag.retriever import retriever

    t0 = time.perf_counter()
    hits = retriever.search(query, k=3)
    elapsed = time.perf_counter() - t0
    scores = [h.get("score", 0) for h in hits]
    return {
        "elapsed_s": round(elapsed, 3),
        "hits": len(hits),
        "top_score": round(max(scores), 4) if scores else None,
    }


def _bench_agent(question: str) -> dict:
    from src.orchestrator.graph import run_agent

    t0 = time.perf_counter()
    result = run_agent(question)
    elapsed = time.perf_counter() - t0
    return {
        "question": question,
        "elapsed_s": round(elapsed, 3),
        "route": result.get("route"),
        "react_iterations": result.get("react_iterations"),
        "answer_chars": len(result.get("answer") or ""),
        "trace_steps": len(result.get("trace") or []),
        "metrics": {
            "primary": (result.get("metrics") or {}).get("primary_display"),
            "tool_calls": (result.get("metrics") or {}).get("tool_calls"),
        },
    }


def main() -> None:
    report: dict = {"scenarios": []}

    print("=== 1) 嵌入模型预热 ===")
    warm = _warm_embedding()
    report["embedding_warmup_s"] = round(warm, 3)
    print(f"  warmup: {warm:.2f}s")

    print("=== 2) 纯检索（含领域增强+重排，无 LLM）===")
    for q in ("玉米叶片氮含量反演方法", "高光谱 LNC 估算精度"):
        r = _bench_retrieval(q)
        r["query"] = q
        report.setdefault("retrieval", []).append(r)
        print(f"  {q[:24]}… → {r['elapsed_s']}s  hits={r['hits']} top={r['top_score']}")

    print("=== 3) 端到端 Agent ===")
    cases = [
        ("chat", "你好，你是谁？"),
        ("policy_rag", "知识库里关于玉米叶片氮含量反演的主要方法有哪些？"),
    ]
    for tag, q in cases:
        print(f"  running [{tag}] {q[:40]}…")
        try:
            r = _bench_agent(q)
            r["tag"] = tag
            report["scenarios"].append(r)
            print(
                f"    → {r['elapsed_s']}s  route={r['route']}  "
                f"react={r['react_iterations']}  tools={r['metrics'].get('tool_calls')}"
            )
        except Exception as e:  # noqa: BLE001
            err = {"tag": tag, "question": q, "error": str(e)}
            report["scenarios"].append(err)
            print(f"    → FAILED: {e}")

    out = ROOT / "data" / "eval" / "latency_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写: {out}")


if __name__ == "__main__":
    main()
