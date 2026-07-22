"""
scripts/generate_rag_testset.py — 生成 DeepResearch RAG 评测集（约 2000 条）

覆盖：方法/概念/对比/模型/应用/流程/精度/预处理/目录/综述/含糊/域外/英文等。
无人工段落标注；must_keywords + prefer_filename 作召回代理。

用法：
  set PYTHONPATH=项目根
  python scripts/generate_rag_testset.py
  python scripts/generate_rag_testset.py --n 2000 --out data/eval/rag_testset_2k.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "eval" / "rag_testset_2k.json"

CROPS = ["玉米", "小麦", "水稻", "茶树", "大豆", "作物", "粮食作物", "冠层"]
SENSORS = ["高光谱", "无人机高光谱", "近地高光谱", "航空高光谱", "多光谱", "遥感"]
METRICS = ["氮含量", "叶氮含量", "LNC", "叶绿素", "LAI", "蛋白质含量", "氮素", "叶片氮"]
METHODS = [
    "PLSR", "SVR", "随机森林", "CNN", "1D-CNN", "深度学习", "机器学习",
    "偏最小二乘", "支持向量回归", "植被指数", "全光谱建模", "SPA", "CARS",
    "Atrous-CDAE-1DCNN", "超参数优化", "迁移学习",
]
INDICES = ["NDVI", "VARI", "NDRE", "GNDVI", "EVI", "红边指数"]
TASKS = ["反演", "估测", "预测", "监测", "估算", "定量反演", "建模"]
ASPECTS = ["方法", "精度", "流程", "挑战", "应用", "优缺点", "数据预处理", "特征选择"]

FN_HS = ["高光谱", "hyperspectral", "氮", "nitrogen", "玉米"]
FN_UAV = ["无人机", "高光谱", "玉米", "氮"]
FN_LAI = ["LAI", "氮", "高光谱"]
FN_CNN = ["CNN", "Atrous", "学习", "Deep", "氮"]
FN_REVIEW = ["review", "hyperspectral", "crops", "高光谱"]

OOD_TEMPLATES = [
    "今天{city}天气怎么样？",
    "帮我写一首关于{topic}的诗",
    "{topic}的最新八卦是什么？",
    "推荐几部{topic}电影",
    "比特币现在适合买入吗？",
    "今晚吃什么外卖比较好？",
    "足球比赛比分是多少？",
    "帮我作一首七言绝句",
    "量子计算在游戏里怎么用？",
    "{city}明天会下雨吗？",
    "写一段笑话逗我开心",
    "股票{topic}怎么看？",
]

CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "南京"]
TOPICS = ["春天", "科幻", "爱情", "校园", "科技", "美食", "旅行", "音乐"]


def _item(
    qid: str,
    question: str,
    category: str,
    *,
    must_keywords: list[str] | None = None,
    prefer_filename: list[str] | None = None,
    expect_low_confidence: bool = False,
    expect_route: str | None = None,
    skip_retrieval: bool = False,
    expect_answer_pass: bool | None = None,
) -> dict:
    row: dict = {
        "id": qid,
        "question": question,
        "category": category,
        "must_keywords": must_keywords or [],
        "prefer_filename": prefer_filename or [],
    }
    if expect_low_confidence:
        row["expect_low_confidence"] = True
    if expect_route:
        row["expect_route"] = expect_route
    if skip_retrieval:
        row["skip_retrieval"] = True
    if expect_answer_pass is not None:
        row["expect_answer_pass"] = expect_answer_pass
    return row


def generate(n: int = 2000, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    items: list[dict] = []
    i = 0

    def add(row: dict) -> None:
        nonlocal i
        i += 1
        row["id"] = f"rag_{i:04d}"
        items.append(row)

    # ---- 种子高质量模板（显式展开）----
    seeds = [
        {"q": "玉米叶片氮含量高光谱反演常用哪些方法？", "cat": "method", "kw": ["氮", "高光谱"], "fn": FN_HS},
        {"q": "什么是叶氮含量 LNC？如何用遥感估算？", "cat": "concept", "kw": ["氮", "LNC"], "fn": ["氮", "nitrogen"]},
        {"q": "PLSR 和 SVR 在氮含量反演中有什么区别？", "cat": "compare", "kw": ["氮"], "fn": FN_HS},
        {"q": "深度学习如何用于高光谱估测叶片氮？", "cat": "method", "kw": ["氮"], "fn": FN_CNN},
        {"q": "Atrous-CDAE-1DCNN 模型是做什么的？", "cat": "model", "kw": ["氮"], "fn": FN_CNN},
        {"q": "无人机高光谱影像估测玉米叶氮的流程？", "cat": "workflow", "kw": ["氮"], "fn": FN_UAV},
        {"q": "高光谱图像如何同时估算 LAI 和氮含量？", "cat": "method", "kw": ["LAI", "氮"], "fn": FN_LAI},
        {"q": "植被指数 NDVI 与氮含量关系如何？", "cat": "index", "kw": ["氮"], "fn": FN_HS},
        {"q": "高光谱数据预处理一般有哪些步骤？", "cat": "preprocess", "kw": ["高光谱"], "fn": ["高光谱", "hyperspectral"]},
        {"q": "对比传统植被指数与全光谱建模估氮的优劣", "cat": "compare", "kw": ["氮"], "fn": FN_HS},
        {"q": "知识库里有哪些文档？", "cat": "catalog", "kw": [], "fn": [], "skip": True},
        {
            "q": "全面综述高光谱遥感在作物氮素监测中的方法与挑战",
            "cat": "deep", "kw": ["高光谱", "氮"], "fn": FN_REVIEW, "route": "deep_research",
        },
    ]
    for s in seeds:
        add(_item(
            "", s["q"], s["cat"],
            must_keywords=s["kw"], prefer_filename=s["fn"],
            expect_route=s.get("route"),
            skip_retrieval=bool(s.get("skip")),
            expect_answer_pass=False if s.get("skip") else True,
        ))

    # ---- 组合生成：方法类 ----
    for crop, sensor, metric, method, task in itertools.product(
        CROPS[:6], SENSORS[:4], METRICS[:5], METHODS[:10], TASKS[:4]
    ):
        templates = [
            f"{crop}{metric}的{sensor}{task}常用哪些{method}相关方法？",
            f"如何用{sensor}结合{method}进行{crop}{metric}{task}？",
            f"{method}在{crop}{sensor}{metric}{task}中的效果如何？",
            f"请介绍基于{sensor}的{crop}{metric}{task}研究进展",
        ]
        q = rng.choice(templates)
        kw = [x for x in [metric[:1] if metric.startswith("氮") else None, "氮" if "氮" in metric or metric == "LNC" else None,
                          "高光谱" if "高光谱" in sensor else None, "LAI" if metric == "LAI" else None] if x]
        if "氮" in metric or metric in ("LNC", "氮素", "叶氮含量", "叶片氮", "氮含量"):
            if "氮" not in kw:
                kw.append("氮")
        if "高光谱" in sensor and "高光谱" not in kw:
            kw.append("高光谱")
        if not kw:
            kw = ["光谱"] if "光谱" in sensor else ["遥感"]
        fn = list(FN_HS)
        if "无人机" in sensor:
            fn = list(FN_UAV)
        if metric == "LAI":
            fn = list(FN_LAI)
        if method in ("CNN", "1D-CNN", "深度学习", "Atrous-CDAE-1DCNN"):
            fn = list(FN_CNN)
        add(_item("", q, "method", must_keywords=kw[:3], prefer_filename=fn, expect_answer_pass=True))
        if len(items) >= n * 0.55:
            break

    # ---- 对比类 ----
    pairs = list(itertools.combinations(
        ["PLSR", "SVR", "随机森林", "CNN", "植被指数", "全光谱建模", "深度学习"], 2
    ))
    for (a, b), crop, metric in itertools.product(pairs, CROPS[:4], METRICS[:4]):
        q = rng.choice([
            f"{a} 和 {b} 在{crop}{metric}反演中有什么区别？",
            f"对比{a}与{b}用于{sensor if False else ''}{crop}{metric}估测的优劣",
            f"{crop}氮素监测中，{a}相对{b}的优势是什么？" if "氮" in metric or metric == "LNC"
            else f"{crop}{metric}估测中{a}与{b}如何选择？",
        ])
        kw = ["氮"] if ("氮" in metric or metric in ("LNC", "氮素")) else [metric[:2] if len(metric) >= 2 else metric]
        if metric == "LAI":
            kw = ["LAI"]
        add(_item("", q, "compare", must_keywords=kw, prefer_filename=FN_HS, expect_answer_pass=True))
        if len(items) >= n * 0.70:
            break

    # ---- 指数 / 精度 / 流程 ----
    for idx, crop, metric in itertools.product(INDICES, CROPS[:5], METRICS[:4]):
        add(_item(
            "", f"植被指数 {idx} 与{crop}{metric}的关系如何？", "index",
            must_keywords=["氮"] if "氮" in metric or metric == "LNC" else [idx[:2]],
            prefer_filename=FN_HS, expect_answer_pass=True,
        ))
        add(_item(
            "", f"如何提高{crop}{metric}高光谱反演精度？", "improve",
            must_keywords=["氮"] if "氮" in metric or metric == "LNC" else ["高光谱"],
            prefer_filename=FN_HS, expect_answer_pass=True,
        ))
        add(_item(
            "", f"无人机高光谱估测{crop}{metric}的典型流程是什么？", "workflow",
            must_keywords=["氮"] if "氮" in metric or metric == "LNC" else ["高光谱"],
            prefer_filename=FN_UAV, expect_answer_pass=True,
        ))
        if len(items) >= n * 0.82:
            break

    # ---- 概念 / 应用 / 预处理 ----
    for metric, sensor, aspect in itertools.product(METRICS, SENSORS[:3], ASPECTS):
        add(_item(
            "", f"{sensor}在{metric}{aspect}方面有哪些要点？", "application",
            must_keywords=["氮"] if "氮" in metric or metric == "LNC" else ["高光谱"],
            prefer_filename=FN_HS, expect_answer_pass=True,
        ))
        if len(items) >= n * 0.88:
            break

    # ---- 英文问 ----
    en_qs = [
        ("What methods are used for leaf nitrogen content estimation with hyperspectral data?",
         ["nitrogen", "hyperspectral"], ["nitrogen", "hyperspectral"]),
        ("How does PLSR perform in hyperspectral nitrogen inversion?",
         ["nitrogen", "PLSR"], FN_HS),
        ("UAV hyperspectral imaging for corn nitrogen estimation workflow",
         ["nitrogen", "hyperspectral"], FN_UAV),
        ("Difference between vegetation indices and full-spectrum models for LNC",
         ["LNC", "nitrogen"], FN_HS),
        ("Deep learning CNN for hyperspectral chlorophyll or nitrogen prediction",
         ["nitrogen"], FN_CNN),
        ("What is LAI and nitrogen joint estimation from hyperspectral imagery?",
         ["LAI", "nitrogen"], FN_LAI),
        ("Review of hyperspectral remote sensing of crops",
         ["hyperspectral"], FN_REVIEW),
        ("Protein content prediction in grain using hyperspectral deep learning",
         ["hyperspectral"], ["Protein", "hyperspectral", "Grain"]),
    ]
    for _ in range(40):
        q, kw, fn = rng.choice(en_qs)
        # 轻微改写后缀
        suffix = rng.choice(["", "?", " please summarize", " with key metrics"])
        add(_item("", q + suffix if not q.endswith("?") else q[:-1] + suffix + ("?" if "?" not in suffix else ""),
                  "english", must_keywords=kw, prefer_filename=fn, expect_answer_pass=True))

    # ---- 目录 / 元问题 ----
    catalog_qs = [
        "知识库里有哪些文档？",
        "列出当前知识库中的论文文件名",
        "库里有多少篇文献？",
        "有哪些高光谱相关文档？",
        "上传了哪些关于氮含量的资料？",
        "介绍一下知识库里有什么内容",
        "现有文档目录是什么？",
    ]
    for q in catalog_qs * 8:
        add(_item("", q, "catalog", must_keywords=[], prefer_filename=[],
                  skip_retrieval=False, expect_answer_pass=True))

    # ---- 深研意图（每条措辞略有差异，避免去重）----
    deep_bases = [
        "全面综述高光谱遥感在作物氮素监测中的方法、精度与挑战",
        "系统对比传统统计模型与深度学习在叶氮估测中的差异并给出展望",
        "请做一份关于无人机高光谱估氮的研究综述",
        "从数据、方法、精度三方面综述作物冠层氮含量遥感反演",
        "写一篇短综述：高光谱氮素反演的关键技术路线与瓶颈",
        "综述高光谱影像用于玉米叶片氮含量反演的国内外进展",
        "请归纳高光谱氮监测中特征选择与建模方法的发展脉络",
        "对比植被指数法与全光谱机器学习在氮素估测上的适用边界",
    ]
    deep_tails = [
        "", "，并列出主要挑战", "，侧重深度学习", "，侧重无人机平台",
        "，要求有方法对照", "（面向研究生开题）", "，控制在要点级",
        "，强调精度评价指标", "，补充数据预处理环节", "，给出研究空白",
    ]
    for base, tail in itertools.product(deep_bases, deep_tails):
        q = base + tail
        add(_item("", q, "deep", must_keywords=["高光谱", "氮"], prefer_filename=FN_REVIEW,
                  expect_route="deep_research", expect_answer_pass=True))

    # ---- 含糊 / 边界 ----
    vague = [
        "光谱是什么？", "精度怎么提高？", "遥感能干什么？", "机器学习模型有哪些？",
        "CNN 和随机森林哪个好？", "氮肥施多了会怎样？", "这个方法靠谱吗？",
        "有什么最新进展？", "怎么做实验？", "数据怎么处理？",
        "模型过拟合怎么办？", "样本不够怎么建模？", "波段怎么选？",
        "要不要做一阶导？", "交叉验证怎么做？",
    ]
    for q, crop in itertools.product(vague, CROPS[:5]):
        add(_item("", f"{q}（结合{crop}高光谱场景）", "borderline",
                  must_keywords=[], prefer_filename=FN_HS, expect_answer_pass=None))

    # ---- 域外 OOD（强制唯一）----
    ood_n = 0
    while ood_n < 250:
        tmpl = rng.choice(OOD_TEMPLATES)
        q = tmpl.format(city=rng.choice(CITIES), topic=rng.choice(TOPICS))
        q = f"{q}（无关测试#{ood_n+1}）"
        add(_item("", q, "ood", must_keywords=[], prefer_filename=[],
                  expect_low_confidence=True, expect_answer_pass=False))
        ood_n += 1

    # ---- 闲聊 ----
    chitchat = ["你好", "你是谁？", "谢谢", "帮我介绍一下你自己", "你会做什么？", "在吗", "早上好"]
    for idx, q in enumerate(chitchat * 12):
        add(_item("", f"{q}" if idx < len(chitchat) else f"{q}#{idx}", "chitchat",
                  skip_retrieval=True, expect_answer_pass=None))

    # 去重（同问题保留先出现的），再补齐到 n
    seen: set[str] = set()
    unique: list[dict] = []
    for row in items:
        key = row["question"].strip()
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)

    # 不足则用改写补齐
    fillers = []
    base_pool = [r for r in unique if r["category"] in ("method", "compare", "application")]
    while len(unique) + len(fillers) < n and base_pool:
        src = rng.choice(base_pool)
        prefix = rng.choice(["请简要说明：", "请从文献角度回答：", "基于知识库：", "科研场景下：", ""])
        suffix = rng.choice(["", "（给出要点即可）", "，并注明可能的局限", "？"])
        q = f"{prefix}{src['question'].rstrip('？?')}{suffix}"
        if q in seen:
            continue
        seen.add(q)
        row = dict(src)
        row["question"] = q
        row["category"] = src["category"] + "_var"
        fillers.append(row)

    unique.extend(fillers)

    # 截断或若仍不足再随机组合
    while len(unique) < n:
        crop, sensor, metric, task = rng.choice(CROPS), rng.choice(SENSORS), rng.choice(METRICS), rng.choice(TASKS)
        q = f"{sensor}用于{crop}{metric}{task}时需要注意什么？"
        if q in seen:
            q = q + f"（案例{len(unique)}）"
        seen.add(q)
        unique.append(_item(
            "", q, "method",
            must_keywords=["氮"] if "氮" in metric or metric == "LNC" else ["高光谱"],
            prefer_filename=FN_HS, expect_answer_pass=True,
        ))

    unique = unique[:n]
    # 重新编号
    for idx, row in enumerate(unique, 1):
        row["id"] = f"rag_{idx:04d}"
    return unique


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = parser.parse_args()

    rows = generate(n=args.n, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    # 统计
    from collections import Counter
    cats = Counter(r["category"] for r in rows)
    ood = sum(1 for r in rows if r.get("expect_low_confidence"))
    deep = sum(1 for r in rows if r.get("expect_route") == "deep_research")
    print(f"写出 {len(rows)} 条 → {out}")
    print(f"域外标记 {ood}，深研路由标记 {deep}")
    print("类别分布（前15）:")
    for k, v in cats.most_common(15):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
