"""
src/llm/provider.py — LLM 统一入口（直连 DeepSeek）

【为什么不用 langchain 的 init_chat_model？】
langchain 封装层在 1.x + openai 2.x 组合下会把 `proxies` 透传给 httpx，
触发 `Client.__init__() got an unexpected keyword argument 'proxies'`。
这里我们直接用 openai SDK v2 自己构造 client，完全掌控参数，绕开该坑，
同时保留 DeepSeek 的 OpenAI 兼容协议，稳定可靠。

对外暴露两个能力：
  - chat():        普通对话（system + messages -> 文本）
  - chat_json():   强制返回 JSON（用于规划/评估等结构化输出）
单例复用连接，线程安全（openai client 内部已处理）。
"""
from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from config.settings import settings
from src.utils.logger import get_logger

log = get_logger("llm")


class LLMProvider:
    _instance: "LLMProvider | None" = None

    def __new__(cls) -> "LLMProvider":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        key = (settings.DEEPSEEK_API_KEY or "").strip()

        # 校验：key 不能为空，且必须是纯 ASCII
        if not key:
            raise RuntimeError(
                "❌ 未配置 DEEPSEEK_API_KEY。请在 .env 填入真实的 DeepSeek key"
                "（platform.deepseek.com 申请，形如 sk-xxxx，纯英文数字）。"
            )
        try:
            key.encode("ascii")
        except UnicodeEncodeError:
            raise RuntimeError(
                "❌ DEEPSEEK_API_KEY 含非 ASCII 字符（可能混进了中文/全角符号/占位符）。"
                "请重新从 DeepSeek 官网复制纯英文数字的 key 到 .env。"
            )

        # 关键：只传 api_key / base_url，绝不传 proxies
        self.client = OpenAI(api_key=key, base_url=settings.DEEPSEEK_BASE_URL)
        self.model = settings.LLM_MODEL
        log.info(f"LLMProvider 就绪 → model={self.model} base={settings.DEEPSEEK_BASE_URL}")

    def _multimodal_config(self) -> tuple[str, str, str]:
        key = (settings.MULTIMODAL_API_KEY or settings.DEEPSEEK_API_KEY or "").strip()
        base = (settings.MULTIMODAL_BASE_URL or settings.DEEPSEEK_BASE_URL or "").strip()
        model = (settings.MULTIMODAL_MODEL or "").strip()
        if not key:
            raise RuntimeError("未配置 MULTIMODAL_API_KEY 或 DEEPSEEK_API_KEY")
        if not model:
            raise RuntimeError("未配置 MULTIMODAL_MODEL（视觉模型名称，如 qwen-vl-max）")
        return key, base, model

    def chat_multimodal(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """多模态对话（图片 + 文本），messages 中 user content 可为 content parts 数组。"""
        if not settings.MULTIMODAL_ENABLED:
            raise RuntimeError("多模态功能已在配置中关闭（MULTIMODAL_ENABLED=false）")
        key, base, model = self._multimodal_config()
        client = OpenAI(api_key=key, base_url=base)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=settings.LLM_TEMPERATURE if temperature is None else temperature,
            max_tokens=max_tokens or settings.MULTIMODAL_MAX_TOKENS,
            stream=False,
        )
        return (resp.choices[0].message.content or "").strip()

    def describe_images(
        self,
        question: str,
        images: list[dict[str, Any]],
        *,
        system: str = "你是科研文献分析助手。请用中文客观描述用户上传的图片内容，"
        "包括图表、文字、实验结果、光谱曲线等可见信息。不要编造看不见的内容。",
    ) -> str:
        """对一批图片做视觉理解，返回文字描述。"""
        from src.multimodal.attachments import image_data_url

        parts: list[dict[str, Any]] = []
        text = question.strip() or "请描述这些图片中的关键信息。"
        parts.append({"type": "text", "text": text})
        for img in images:
            parts.append({
                "type": "image_url",
                "image_url": {"url": image_data_url(img)},
            })
        return self.chat_multimodal([
            {"role": "system", "content": system},
            {"role": "user", "content": parts},
        ])

    # ------------------------------------------------------------------ #
    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """标准对话，返回纯文本。"""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=settings.LLM_TEMPERATURE if temperature is None else temperature,
            max_tokens=max_tokens or settings.LLM_MAX_TOKENS,
            stream=False,
        )
        return (resp.choices[0].message.content or "").strip()

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """带 function calling 的对话，返回 assistant 消息结构（含 tool_calls）。"""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools or None,
            tool_choice=tool_choice if tools else None,
            temperature=settings.LLM_TEMPERATURE if temperature is None else temperature,
            max_tokens=max_tokens or settings.LLM_MAX_TOKENS,
            stream=False,
        )
        msg = resp.choices[0].message
        tool_calls: list[dict[str, Any]] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments or "{}",
                    },
                })
        return {
            "content": (msg.content or "").strip() or None,
            "tool_calls": tool_calls,
        }

    def chat_json(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """
        要求模型输出 JSON。DeepSeek 支持 response_format=json_object。
        失败时做一次“裸文本抽取 JSON”的兜底，保证规划器不会因格式崩溃。
        """
        raw = ""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=settings.LLM_MAX_TOKENS,
                response_format={"type": "json_object"},
                stream=False,
            )
            raw = (resp.choices[0].message.content or "{}").strip()
            return json.loads(raw)
        except Exception as e:  # noqa: BLE001
            log.warning(f"chat_json 解析失败，启用兜底抽取: {e}")
            return self._salvage_json(raw)

    @staticmethod
    def _salvage_json(text: str) -> dict[str, Any]:
        """从可能带 ```json 包裹的文本里抠出第一段合法 JSON。"""
        import re
        if not text:
            return {}
        text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE)
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start : end + 1])
            except Exception:  # noqa: BLE001
                return {}
        return {}


# 全局单例
llm = LLMProvider()
