"""
src/memory/context_manager.py — 决策上下文管理

解决多轮工具调用导致 Prompt 膨胀、注意力稀释的问题：
  1. Token 预算滑动窗口：超预算时压缩最旧同类工具结果
  2. 同源工具结果语义压缩：多轮相同工具只保留最新摘要
  3. 错误栈精简：只保留 error / error_type，不把冗长堆栈喂给模型
"""
from __future__ import annotations

import json
from typing import Any


class ContextManager:
    def __init__(self, max_chars: int = 14000, tool_snippet: int = 1200) -> None:
        self.max_chars = max_chars
        self.tool_snippet = tool_snippet

    def prepare_for_policy(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """策略决策前压缩 messages，返回可安全送给 LLM 的副本。"""
        from src.llm.messages import repair_tool_message_chain

        compact = [self._compact_message(m) for m in messages]
        # 先修链再压预算，压预算时整组丢弃，避免留下孤儿 tool_calls
        repaired = repair_tool_message_chain(compact)
        return repair_tool_message_chain(self._enforce_budget(repaired))

    def _compact_message(self, msg: dict[str, Any]) -> dict[str, Any]:
        out = dict(msg)
        role = out.get("role")
        if role == "tool":
            out["content"] = self._compact_tool_content(out.get("content", ""))
        elif role == "assistant" and out.get("content") and len(str(out["content"])) > 4000:
            out["content"] = str(out["content"])[:4000] + "…（已截断）"
        return out

    def _compact_tool_content(self, content: Any) -> str:
        raw = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw[: self.tool_snippet] + ("…" if len(raw) > self.tool_snippet else "")

        if not isinstance(data, dict):
            text = json.dumps(data, ensure_ascii=False)
            return text[: self.tool_snippet] + ("…" if len(text) > self.tool_snippet else "")

        # 错误只保留码与短信息
        if data.get("ok") is False:
            return json.dumps(
                {
                    "ok": False,
                    "error": str(data.get("error", ""))[:240],
                    "error_type": data.get("error_type", "runtime"),
                },
                ensure_ascii=False,
            )

        # 检索结果：保留来源 + score + 正文摘要
        output = data.get("output")
        if isinstance(output, dict) and isinstance(output.get("hits"), list):
            slim = []
            for item in (output.get("hits") or [])[:5]:
                if not isinstance(item, dict):
                    continue
                slim.append({
                    "source": item.get("source") or item.get("filename"),
                    "score": item.get("score"),
                    "content": str(item.get("content", ""))[:400],
                })
            body = {
                "ok": True,
                "hits": slim,
                "confidence": output.get("confidence"),
                "low_confidence": output.get("low_confidence"),
            }
            if output.get("instruction"):
                body["instruction"] = str(output["instruction"])[:200]
            return json.dumps(body, ensure_ascii=False)

        if isinstance(output, list) and output and isinstance(output[0], dict):
            slim = []
            for item in output[:5]:
                slim.append({
                    "source": item.get("source") or item.get("filename"),
                    "score": item.get("score"),
                    "content": str(item.get("content", ""))[:400],
                })
            return json.dumps({"ok": True, "hits": slim}, ensure_ascii=False)

        if isinstance(output, dict) and "results" in output:
            results = output.get("results") or []
            slim = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": str(r.get("snippet", ""))[:280],
                }
                for r in results[:5]
                if isinstance(r, dict)
            ]
            return json.dumps(
                {"ok": True, "query": output.get("query"), "count": len(slim), "results": slim},
                ensure_ascii=False,
            )

        text = json.dumps(data, ensure_ascii=False)
        return text[: self.tool_snippet] + ("…" if len(text) > self.tool_snippet else "")

    def _enforce_budget(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        保留 system + 最新对话；超预算时整组删除最旧的
        assistant(+tool_calls)+其后 tool 回执，绝不单独删 tool 消息。
        """
        if not messages:
            return messages

        system = [m for m in messages if m.get("role") == "system"]
        rest = [m for m in messages if m.get("role") != "system"]

        def total(msgs: list[dict]) -> int:
            return sum(len(str(m.get("content") or "")) for m in msgs)

        def drop_oldest_tool_group(msgs: list[dict]) -> bool:
            """删除最早一段 assistant.tool_calls + 紧随 tool；成功返回 True。"""
            for i, m in enumerate(msgs):
                if m.get("role") != "assistant" or not m.get("tool_calls"):
                    continue
                # 保留末尾最近一轮（至少留 1 组工具交互 + 后续）
                j = i + 1
                while j < len(msgs) and msgs[j].get("role") == "tool":
                    j += 1
                # 若这是消息末尾唯一一组，不要删，避免空上下文
                if i == 0 and j >= len(msgs) - 1:
                    return False
                del msgs[i:j]
                return True
            return False

        while total(system + rest) > self.max_chars and len(rest) > 3:
            if drop_oldest_tool_group(rest):
                continue
            # 再丢最旧的非 user 前缀（不能是半截 tool）
            if rest and rest[0].get("role") == "tool":
                rest.pop(0)
                continue
            if rest and rest[0].get("role") != "user":
                rest.pop(0)
                continue
            break
        return system + rest


context_manager = ContextManager()
