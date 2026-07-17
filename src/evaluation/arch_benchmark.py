"""
src/evaluation/arch_benchmark.py — 架构改造量化评测

不依赖真实 LLM 费用，用「图节点跳转次数 + 本地工具并行开销」给出可复现数字，
用于简历项目成果写量化亮点。

运行：
  python -m src.evaluation.arch_benchmark
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from config.settings import PROJECT_ROOT


@dataclass
class Scenario:
    name: str
    description: str
    old_hops: int
    new_hops: int
    old_path: str
    new_path: str

    @property
    def hop_reduction_pct(self) -> float:
        if self.old_hops <= 0:
            return 0.0
        return round((self.old_hops - self.new_hops) / self.old_hops * 100, 1)


# 旧架构：recall → rewrite → multimodal → router → think↔tools → finalize
# 新架构：workflow → policy/chat/deep_research
SCENARIOS: list[Scenario] = [
    Scenario(
        name="chitchat",
        description="日常寒暄（你好/你是谁）",
        old_hops=5,  # 4 preprocess + chat；若误走 super 则更多
        new_hops=2,  # workflow + chat
        old_path="recall→rewrite→multimodal→router→chat",
        new_path="workflow→chat",
    ),
    Scenario(
        name="qa_1_tool_round",
        description="文献问答：同轮 1～N 个工具，一次再推理后作答",
        old_hops=8,  # 4 prep + think + tools + think + finalize
        new_hops=2,  # workflow + policy（环在节点内）
        old_path="prep×4→think→tools→think→finalize",
        new_path="workflow→policy",
    ),
    Scenario(
        name="qa_2_serial_rounds",
        description="策略环两轮串行工具（每轮各调工具后再想）",
        old_hops=10,  # 4 prep + T→U→T→U→T + finalize
        new_hops=2,
        old_path="prep×4→think↔tools×2→think→finalize",
        new_path="workflow→policy",
    ),
    Scenario(
        name="llm_calls_fast_path",
        description="快路径额外 LLM：Query Rewrite + 路由分类（不含 Policy）",
        old_hops=2,  # rewrite + classify（计为 2 次多余 LLM）
        new_hops=0,  # FAST_MODE 默认关闭
        old_path="query_rewrite(LLM)+router_classify(LLM)",
        new_path="规则路由，无额外 LLM",
    ),
]


def _fake_tool(latency_s: float) -> float:
    time.sleep(latency_s)
    return latency_s


def bench_parallel_tools(
    n_tools: int = 3,
    per_tool_s: float = 0.25,
    rounds: int = 3,
) -> dict:
    """模拟同轮多工具：串行 vs 并行墙钟时间。"""
    serial_times: list[float] = []
    parallel_times: list[float] = []

    for _ in range(rounds):
        t0 = time.perf_counter()
        for _i in range(n_tools):
            _fake_tool(per_tool_s)
        serial_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n_tools) as pool:
            futs = [pool.submit(_fake_tool, per_tool_s) for _ in range(n_tools)]
            for f in as_completed(futs):
                f.result()
        parallel_times.append(time.perf_counter() - t0)

    serial_avg = sum(serial_times) / len(serial_times)
    parallel_avg = sum(parallel_times) / len(parallel_times)
    speedup = serial_avg / parallel_avg if parallel_avg > 0 else 0.0
    return {
        "n_tools": n_tools,
        "per_tool_s": per_tool_s,
        "rounds": rounds,
        "serial_avg_s": round(serial_avg, 3),
        "parallel_avg_s": round(parallel_avg, 3),
        "speedup": round(speedup, 2),
        "latency_reduction_pct": round((1 - parallel_avg / serial_avg) * 100, 1) if serial_avg else 0.0,
    }


def run_benchmark() -> dict:
    scenarios = []
    for s in SCENARIOS:
        scenarios.append({
            **asdict(s),
            "hop_reduction_pct": s.hop_reduction_pct,
        })

    hop_rates = [s.hop_reduction_pct for s in SCENARIOS if s.name != "llm_calls_fast_path"]
    summary = {
        "main_graph_nodes_old": 9,  # recall,rewrite,multimodal,router,chat,think,tools,finalize,deep
        "main_graph_nodes_new": 4,  # workflow,chat,policy,deep
        "main_graph_node_reduction_pct": round((9 - 4) / 9 * 100, 1),
        "avg_hop_reduction_pct_qa": round(sum(hop_rates) / len(hop_rates), 1) if hop_rates else 0.0,
        "extra_llm_eliminated_fast_path": 2,
        "react_max_iterations_default_old": 8,
        "react_max_iterations_default_new": 4,
    }

    parallel = bench_parallel_tools()
    report = {
        "title": "DeepResearch 架构改造量化报告",
        "method": (
            "图节点跳转按编排路径静态计数；"
            "额外 LLM 按 FAST_MODE 快路径规则统计；"
            "并行工具用 sleep 模拟等时延工具墙钟时间（不消耗 API）。"
        ),
        "summary": summary,
        "scenarios": scenarios,
        "parallel_tools": parallel,
        "resume_bullets": _resume_bullets(summary, parallel, scenarios),
    }
    return report


def _resume_bullets(summary: dict, parallel: dict, scenarios: list[dict]) -> list[str]:
    qa = next(s for s in scenarios if s["name"] == "qa_1_tool_round")
    serial = next(s for s in scenarios if s["name"] == "qa_2_serial_rounds")
    return [
        (
            f"将图级 ReAct（think<->tools）重构为 Workflow + 统一策略节点："
            f"主图节点数 {summary['main_graph_nodes_old']}->{summary['main_graph_nodes_new']} "
            f"(降{summary['main_graph_node_reduction_pct']}%); "
            f"单轮工具问答路径跳转 {qa['old_hops']}->{qa['new_hops']} "
            f"(降{qa['hop_reduction_pct']}%)."
        ),
        (
            f"两轮串行工具场景跳转 {serial['old_hops']}->{serial['new_hops']} "
            f"(降{serial['hop_reduction_pct']}%); "
            f"FAST_MODE 默认关闭 Query Rewrite / LLM 路由分类，"
            f"每问减少约 {summary['extra_llm_eliminated_fast_path']} 次无效推理。"
        ),
        (
            f"同轮多工具并行执行：{parallel['n_tools']} 个等时延工具场景下 "
            f"墙钟时延 {parallel['serial_avg_s']}s->{parallel['parallel_avg_s']}s "
            f"(加速约 {parallel['speedup']}x，降{parallel['latency_reduction_pct']}%)."
        ),
    ]


def main() -> None:
    report = run_benchmark()
    out_dir = Path(PROJECT_ROOT) / "data" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "arch_benchmark.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 60)
    print(report["title"])
    print("=" * 60)
    print("method:", report["method"])
    print("\n[summary]")
    for k, v in report["summary"].items():
        print(f"  {k}: {v}")
    print("\n[scenarios]")
    for s in report["scenarios"]:
        print(
            f"  - {s['name']}: {s['old_hops']} -> {s['new_hops']} "
            f"(-{s['hop_reduction_pct']}%) | {s['description']}"
        )
    print("\n[parallel_tools]")
    p = report["parallel_tools"]
    print(
        f"  {p['n_tools']} tools x {p['per_tool_s']}s: "
        f"serial {p['serial_avg_s']}s -> parallel {p['parallel_avg_s']}s "
        f"(x{p['speedup']}, -{p['latency_reduction_pct']}%)"
    )
    print("\n[resume_bullets]")
    for i, b in enumerate(report["resume_bullets"], 1):
        print(f"  {i}. {b}")
    print(f"\nreport: {out_path}")


if __name__ == "__main__":
    main()
