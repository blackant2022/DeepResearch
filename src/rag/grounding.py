"""
src/rag/grounding.py — 幻觉治理（Grounding / 事实核验）

【这是"解决幻觉"的核心模块】。思路是双层防护：

  1) 生成前约束（在 writer 的 prompt 里强制"只用证据作答/未知就说不知道"）。
  2) 生成后核验（本模块）：把答案拆成若干原子论断（claims），逐条让 LLM 判断
     "该论断是否被检索到的证据支撑"，算出 grounding 支撑率。
     - 支撑率 ≥ 阈值 → 通过
     - 支撑率 <  阈值 → 判定可能幻觉，返回未被支撑的论断，交给 Critic 触发修订

【关键优化 v2】对首次核验未被支撑的论断，做针对性 RAG 再检索（per-claim fallback），
显著降低因"初次检索范围不够精准"导致的误判。
"""
from __future__ import annotations

import re

from config.settings import settings
from src.llm.provider import llm
from src.utils.logger import get_logger

log = get_logger("grounding")

_CLAIM_PROMPT = """把下面这段回答拆分成若干条独立、可验证的原子论断（claim）。

重要规则：
- 只提取"关于事实的断言"（如"X 是 Y"，"X 达到 Z 值"）
- 如果某句话表达的是"证据不足/无法确定/没有找到相关描述"等元认知结论，
  这说明作者诚实地承认了知识局限，不要把它拆成论断——因为它是关于"知识的状态"，
  而非关于"事实的断言"，核验它会错误地判定为幻觉。
- 跳过所有"根据现有资料无法确定……"、"证据不足……"、"未找到……"这类表述

只输出 JSON：{{"claims": ["论断1", "论断2", ...]}}，不要多余文字。

回答：
{answer}"""

_VERIFY_PROMPT = """你是严格的事实核查员。请判断【论断】是否能被【证据】直接支撑。
只输出 JSON：{{"supported": true/false, "reason": "简短理由"}}。
判断标准：证据必须明确支持该论断，模糊或无关都算 false。

【证据】
{evidence}

【论断】
{claim}"""


class GroundingChecker:
    def check(self, answer: str, evidence_chunks: list[dict], question: str = "") -> dict:
        """
        参数:
          answer:         待核验的回答
          evidence_chunks: 证据块列表，每项含 {"content", "filename", "score"}
          question:       原始用户问题（用于 per-claim 兜底检索）

        返回:
          {
            "grounded": bool,          # 是否整体通过
            "support_rate": float,     # 支撑率
            "unsupported": [str, ...], # 未被支撑的论断（用于修订）
            "claims_total": int
          }
        """
        evidence = "\n\n".join(f"- {c['content']}" for c in evidence_chunks) or "（无证据）"

        # ---- 第一步：拆解原子论断 ----
        claims = llm.chat_json(
            [{"role": "user", "content": _CLAIM_PROMPT.format(answer=answer)}]
        ).get("claims", [])

        # ---- 过滤器：排除"无法确定"类元认知语句（即使 LLM 没听话，这里也兜住） ----
        _UNCERTAINTY_PATTERNS = re.compile(
            r"(根据(现有资料|当前资料|已有资料)?(无法|难以|暂未|不能)(确定|判断|证实|找到|获取|提供|描述|说明))|"
            r"(未(找到|发现|提及|涉及|描述|说明|提供|有))|"
            r"(没有(找到|发现|提及|描述|说明|提供))|"
            r"(证据不(足|够|充分))|"
            r"(不(清楚|确定|知道|明确))"
        )
        filtered = [c for c in claims if not _UNCERTAINTY_PATTERNS.search(c)]
        if filtered != claims:
            log.info(f"过滤掉 {len(claims) - len(filtered)} 条元认知语句（如'无法确定'），保留 {len(filtered)} 条事实论断")
            claims = filtered

        if not claims:
            # 拆不出论断（答案可能是"未找到信息"），视为通过，避免误杀
            return {"grounded": True, "support_rate": 1.0, "unsupported": [], "claims_total": 0}

        # ---- 第二步：用已有证据逐一核验 ----
        supported = []
        pending = []  # 第一轮未通过的，进入第二轮
        for c in claims:
            verdict = llm.chat_json(
                [{"role": "user", "content": _VERIFY_PROMPT.format(evidence=evidence, claim=c)}]
            )
            if verdict.get("supported") is True:
                supported.append(c)
            else:
                pending.append(c)

        # ---- 第三步：对首次未通过的论断做针对性 RAG 再检索（per-claim fallback） ----
        if pending and question:
            newly_supported = []
            # 延迟导入避免循环依赖，且只在必要时加载
            from src.rag.retriever import retriever as _retriever
            for c in pending:
                try:
                    # 从论断中提取关键名词短语作为搜索词，而不是用全句
                    # 去掉"根据资料"、"无法确定"、"没有"等修饰词，只保留核心实体
                    search_terms = re.sub(
                        r"(根据(现有资料|当前资料|已有资料))|"
                        r"((无法|难以|暂未|不能)(确定|判断|证实|找到|获取|提供|描述|说明))|"
                        r"(未(找到|发现|提及|涉及|描述|说明|提供|有))|"
                        r"(没有(找到|发现|提及|描述|说明|提供))",
                        "", c
                    ).strip()
                    if not search_terms or len(search_terms) < 4:
                        search_terms = c  # 提取失败，回退用原始论断
                    hits = _retriever.search(search_terms, k=3)
                    if hits:
                        extra_evidence = "\n\n".join(f"- {h['content']}" for h in hits)
                        verdict = llm.chat_json(
                            [{"role": "user",
                              "content": _VERIFY_PROMPT.format(evidence=extra_evidence, claim=c)}]
                        )
                        if verdict.get("supported") is True:
                            supported.append(c)
                            newly_supported.append(c)
                except Exception as e:
                    log.debug(f"per-claim RAG 检索失败(c={c[:30]}): {e}")
            if newly_supported:
                log.info(f"per-claim RAG 兜底：{len(newly_supported)}/{len(pending)} 个论断获证据支撑")
                pending = [c for c in pending if c not in newly_supported]

        # ---- 第四步：计算结果 ----
        unsupported = pending  # 剩下的就是真·未被支撑
        rate = len(supported) / len(claims)
        grounded = rate >= settings.GROUNDING_THRESHOLD
        log.info(
            f"幻觉核验：支撑率={rate:.0%} 阈值={settings.GROUNDING_THRESHOLD:.0%}"
            f"（{len(supported)}/{len(claims)}）→ {'通过' if grounded else '疑似幻觉'}"
        )
        return {
            "grounded": grounded,
            "support_rate": round(rate, 3),
            "unsupported": unsupported,
            "claims_total": len(claims),
        }


grounding_checker = GroundingChecker()
