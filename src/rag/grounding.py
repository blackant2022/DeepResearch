"""
src/rag/grounding.py — 幻觉治理（Grounding / 事实核验）

快路径原则：
  - 论断数封顶（默认 6）
  - 拆 claim + 批量核验合并为 1 次 LLM
  - per-claim 二次检索默认关闭（否则深研可到数分钟）
"""
from __future__ import annotations

import re

from config.settings import settings
from src.llm.provider import llm
from src.utils.logger import get_logger

log = get_logger("grounding")

_COMBINED_PROMPT = """你是严格的事实核查员。请完成两步并只输出一个 JSON：

1) 从【回答】提取最多 {max_claims} 条独立、可验证的事实论断（跳过「无法确定/未找到」等元认知句）
2) 根据【证据】判断每条论断是否被直接支撑；模糊或无关一律 false

【证据】
{evidence}

【回答】
{answer}

只输出 JSON：
{{"claims": [{{"text": "论断", "supported": true/false}}, ...]}}
"""

_FALLBACK_VERIFY = """证据：
{evidence}

论断：{claim}
只输出 JSON：{{"supported": true/false}}"""

# 拒答/证据缺口句：不算可核验事实论断；整篇若只剩这类句子，不得虚报 100% 通过。
_UNCERTAINTY = re.compile(
    r"(根据(现有资料|当前资料|已有资料)?(无法|难以|暂未|不能)(确定|判断|证实|找到|获取|提供|描述|说明))|"
    r"(未(找到|发现|提及|涉及|描述|说明|提供|有))|"
    r"(没有(找到|发现|提及|描述|说明|提供))|"
    r"(证据不(足|够|充分))|"
    r"(知识库(中)?(未|没有|无).{0,12}(文献|资料|内容|信息))|"
    r"(不(清楚|确定|知道|明确))"
)


def _is_refusal_answer(answer: str) -> bool:
    text = (answer or "").strip()
    if not text:
        return True
    return bool(_UNCERTAINTY.search(text))


class GroundingChecker:
    def check(self, answer: str, evidence_chunks: list[dict], question: str = "") -> dict:
        max_claims = int(getattr(settings, "GROUNDING_MAX_CLAIMS", 6) or 6)
        evidence_cap = int(getattr(settings, "GROUNDING_EVIDENCE_CHARS", 4000) or 4000)
        per_claim = bool(getattr(settings, "GROUNDING_PER_CLAIM", False))

        parts = []
        for c in evidence_chunks[:12]:
            text = str(c.get("content") or "")[:500]
            if text:
                parts.append(f"- {text}")
        evidence = "\n\n".join(parts) or "（无证据）"
        if len(evidence) > evidence_cap:
            evidence = evidence[:evidence_cap] + "…"

        data = llm.chat_json(
            [{"role": "user", "content": _COMBINED_PROMPT.format(
                max_claims=max_claims,
                evidence=evidence,
                answer=answer[:5000],
            )}]
        )
        raw_claims = data.get("claims") or []

        claims: list[str] = []
        flags: list[bool | None] = []
        for item in raw_claims:
            if isinstance(item, str):
                text, supported = item, None
            elif isinstance(item, dict):
                text = str(item.get("text") or item.get("claim") or "").strip()
                supported = item.get("supported")
                if supported is not None:
                    supported = bool(supported is True)
            else:
                continue
            if not text or _UNCERTAINTY.search(text):
                continue
            claims.append(text)
            flags.append(supported)
            if len(claims) >= max_claims:
                break

        if not claims:
            refusal = _is_refusal_answer(answer)
            reason = (
                "回答仅为证据缺口/拒答，无可核验事实论断"
                if refusal
                else "未能提取可核验事实论断"
            )
            log.info(f"幻觉核验：无论断 → 支撑率=0%（{reason}）→ 不通过")
            return {
                "grounded": False,
                "support_rate": 0.0,
                "supported": [],
                "unsupported": [reason],
                "claims_total": 0,
                "no_claims": True,
                "refusal": refusal,
            }

        # 漏标 supported 时最多补核 2 条
        missing_idx = [i for i, f in enumerate(flags) if f is None]
        for i in missing_idx[:2]:
            try:
                one = llm.chat_json(
                    [{"role": "user", "content": _FALLBACK_VERIFY.format(
                        evidence=evidence[:2000], claim=claims[i]
                    )}]
                )
                flags[i] = bool(one.get("supported") is True)
            except Exception:  # noqa: BLE001
                flags[i] = False
        for i in missing_idx[2:]:
            flags[i] = False

        supported = [c for c, ok in zip(claims, flags) if ok]
        pending = [c for c, ok in zip(claims, flags) if not ok]

        if per_claim and pending and question:
            pending = self._per_claim_rescue(pending, supported, question)

        rate = len(supported) / len(claims) if claims else 1.0
        grounded = rate >= settings.GROUNDING_THRESHOLD
        log.info(
            f"幻觉核验：支撑率={rate:.0%} 阈值={settings.GROUNDING_THRESHOLD:.0%}"
            f"（{len(supported)}/{len(claims)}，claims≤{max_claims}，"
            f"per_claim={'on' if per_claim else 'off'}，merged=1llm）→ "
            f"{'通过' if grounded else '疑似幻觉'}"
        )
        return {
            "grounded": grounded,
            "support_rate": round(rate, 3),
            "supported": supported,
            "unsupported": pending,
            "claims_total": len(claims),
        }

    def _per_claim_rescue(
        self, pending: list[str], supported: list[str], question: str
    ) -> list[str]:
        """仅处理前 GROUNDING_PER_CLAIM_MAX 条，防止灾难级延迟。"""
        from src.rag.retriever import retriever as _retriever

        cap = int(getattr(settings, "GROUNDING_PER_CLAIM_MAX", 3) or 3)
        still = []
        rescued = 0
        for c in pending:
            if rescued >= cap:
                still.append(c)
                continue
            try:
                hits = _retriever.search(c[:80], k=2)
                if not hits:
                    still.append(c)
                    continue
                extra = "\n".join(f"- {h['content'][:300]}" for h in hits)
                verdict = llm.chat_json(
                    [{"role": "user", "content": _FALLBACK_VERIFY.format(
                        evidence=extra, claim=c
                    )}]
                )
                if verdict.get("supported") is True:
                    supported.append(c)
                    rescued += 1
                else:
                    still.append(c)
            except Exception as e:  # noqa: BLE001
                log.debug(f"per-claim 失败: {e}")
                still.append(c)
        if rescued:
            log.info(f"per-claim 兜底：救回 {rescued} 条（上限 {cap}）")
        return still


grounding_checker = GroundingChecker()
