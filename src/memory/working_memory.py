"""
src/memory/working_memory.py — 工作记忆（短期 / scratchpad）

【定位】内置能力，不在前端展示。它是单次任务运行期间的“草稿纸”：
  - 存放规划、子任务中间结果、工具调用观察、Agent 间传递的消息
  - 随任务生命周期存在，任务结束即可丢弃（或摘要后写入长期记忆）

设计要点（对应“如何设计工作记忆”）：
  1. 容量受限：用滑动窗口 + token 预算，避免上下文爆炸
  2. 分区存储：facts（已核实事实）/ scratch（临时）/ messages（Agent通信）
  3. 可摘要：超预算时对最旧内容做压缩，保留信息密度
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryItem:
    role: str            # 来源：planner / researcher / writer / critic / tool
    content: str
    kind: str = "scratch"  # fact | scratch | message | observation
    ts: float = field(default_factory=time.time)
    meta: dict[str, Any] = field(default_factory=dict)


class WorkingMemory:
    def __init__(self, max_chars: int = 12000) -> None:
        self._items: list[MemoryItem] = []
        self._max_chars = max_chars  # 简化的 token 预算（按字符近似）

    # ---- 写 ----
    def add(self, role: str, content: str, kind: str = "scratch", **meta: Any) -> None:
        self._items.append(MemoryItem(role=role, content=content, kind=kind, meta=meta))
        self._enforce_budget()

    def add_fact(self, role: str, content: str, **meta: Any) -> None:
        self.add(role, content, kind="fact", **meta)

    def add_message(self, sender: str, content: str, to: str = "all") -> None:
        """Agent 间通信：一条消息写入共享工作记忆。"""
        self.add(sender, content, kind="message", to=to)

    # ---- 读 ----
    def facts(self) -> list[MemoryItem]:
        return [i for i in self._items if i.kind == "fact"]

    def observations(self) -> list[MemoryItem]:
        return [i for i in self._items if i.kind == "observation"]

    def messages_for(self, agent: str) -> list[MemoryItem]:
        return [i for i in self._items if i.kind == "message" and i.meta.get("to") in (agent, "all")]

    def render(self, kinds: tuple[str, ...] = ("fact", "observation")) -> str:
        """把选定分区拼成给 LLM 的上下文文本。"""
        lines = [f"[{i.role}/{i.kind}] {i.content}" for i in self._items if i.kind in kinds]
        return "\n".join(lines) if lines else "（工作记忆为空）"

    # ---- 预算控制 ----
    def _enforce_budget(self) -> None:
        # fact 永不丢；优先淘汰最旧的 scratch/observation
        while self._total_chars() > self._max_chars:
            idx = next((k for k, it in enumerate(self._items) if it.kind != "fact"), None)
            if idx is None:
                break
            self._items.pop(idx)

    def _total_chars(self) -> int:
        return sum(len(i.content) for i in self._items)

    def snapshot(self) -> list[dict[str, Any]]:
        """导出（供长期记忆摘要用），不供前端展示。"""
        return self.export_snapshot()

    def export_snapshot(self) -> list[dict[str, Any]]:
        """可序列化快照，供 LangGraph 检查点持久化。"""
        return [
            {"role": i.role, "kind": i.kind, "content": i.content, "meta": dict(i.meta)}
            for i in self._items
        ]

    @classmethod
    def from_snapshot(cls, items: list[dict[str, Any]] | None, max_chars: int = 12000) -> WorkingMemory:
        wm = cls(max_chars=max_chars)
        for it in items or []:
            meta = dict(it.get("meta") or {})
            wm.add(it["role"], it["content"], kind=it.get("kind", "scratch"), **meta)
        return wm
