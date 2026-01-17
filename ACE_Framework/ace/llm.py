"""LLM client abstractions used by ACE components."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
import json
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Union


@dataclass
class LLMResponse:
    """Container for LLM outputs."""

    text: str
    raw: Optional[Dict[str, Any]] = None


class LLMClient(ABC):
    """Abstract interface so ACE can plug into any chat/completions API."""

    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model

    @abstractmethod
    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Return the model text for a given prompt."""


class DummyLLMClient(LLMClient):
    """Deterministic LLM stub for testing and dry runs."""

    def __init__(self, responses: Optional[Deque[str]] = None) -> None:
        super().__init__(model="dummy")
        self._responses: Deque[str] = responses or deque()

    def queue(self, text: str) -> None:
        """Enqueue a response to be used on the next completion call."""
        self._responses.append(text)

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        if not self._responses:
            raise RuntimeError("DummyLLMClient ran out of queued responses.")
        text = self._responses.popleft()
        return LLMResponse(text=text, raw=None)


# class TransformersLLMClient(LLMClient):
#     """LLM client powered by `transformers` pipelines for chat-style models."""
#
#     def __init__(
#         self,
#         model_path: str,
#         *,
#         max_new_tokens: int = 512,
#         temperature: float = 0.0,
#         top_p: float = 0.9,
#         device_map: Union[str, Dict[str, int]] = "auto",
#         torch_dtype: Union[str, "torch.dtype"] = "auto",
#         trust_remote_code: bool = True,
#         system_prompt: Optional[str] = None,
#         generation_kwargs: Optional[Dict[str, Any]] = None,
#     ) -> None:
#         super().__init__(model=model_path)
#
#         # Import transformers lazily to avoid mandatory dependency for all users.
#         from transformers import AutoTokenizer, pipeline  # type: ignore[import-untyped]
#
#         self._tokenizer = AutoTokenizer.from_pretrained(
#             model_path, trust_remote_code=trust_remote_code
#         )
#         self._pipeline = pipeline(
#             "text-generation",
#             model=model_path,
#             tokenizer=self._tokenizer,
#             torch_dtype=torch_dtype,
#             device_map=device_map,
#             trust_remote_code=trust_remote_code,
#         )
#         self._system_prompt = system_prompt or (
#             "You are a JSON-only assistant that MUST reply with a single valid JSON object without extra text.\n"
#             "Reasoning: low\n"
#             "Do not expose analysis or chain-of-thought. Respond using the final JSON only."
#         )
#         self._defaults: Dict[str, Any] = {
#             "max_new_tokens": max_new_tokens,
#             "temperature": temperature,
#             "top_p": top_p,
#             "do_sample": temperature > 0.0,
#             "return_full_text": False,
#         }
#         if generation_kwargs:
#             self._defaults.update(generation_kwargs)
#
#     def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
#         call_kwargs = dict(self._defaults)
#         kwargs = dict(kwargs)
#         kwargs.pop("refinement_round", None)
#         call_kwargs.update(kwargs)
#
#         # Build chat-formatted messages to leverage harmony template.
#         messages = [
#             {"role": "system", "content": self._system_prompt},
#             {"role": "user", "content": prompt},
#         ]
#
#         outputs = self._pipeline(messages, **call_kwargs)
#         text = self._postprocess_text(self._extract_text(outputs))
#         return LLMResponse(text=text, raw={"outputs": outputs})
from typing import Any, Dict, Optional, Union, Literal

# 你现有项目里应该已经有这两个类型；如果没有，可用下面两个简化占位
class LLMResponse:
    def __init__(self, text: str, raw: Any) -> None:
        self.text = text
        self.raw = raw

class LLMClient:
    def __init__(self, model: str) -> None:
        self.model = model


class UniversalLLMClient(LLMClient):
    """
    通用 LLM 客户端：同时支持本地 transformers 与远端 OpenAI/兼容 API。

    - backend="transformers": 使用本地/本机显卡的 HF pipeline 进行推理
    - backend="openai": 使用 openai>=1.0 的官方 SDK 调 chat.completions
      * 也支持 OpenAI-兼容的 base_url（如自建 vLLM / OpenRouter 等）

    统一入口:
        complete(prompt: str, **kwargs) -> LLMResponse
    """

    def __init__(
        self,
        model: str,
        *,
        backend: Literal["transformers", "openai"] = "transformers",
        # ---- 通用生成参数（两端都会用，必要时做映射） ----
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.9,
        system_prompt: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        # ---- 本地 transformers 专用 ----
        device_map: Union[str, Dict[str, int]] = "auto",
        torch_dtype: Union[str, "torch.dtype"] = "auto",
        trust_remote_code: bool = True,
        # ---- 远端 OpenAI/兼容 专用 ----
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,  # 兼容端点时填写，如 http://localhost:8000/v1
        organization: Optional[str] = None,
    ) -> None:
        super().__init__(model=model)
        self._backend = backend
        self._system_prompt = system_prompt or (
            "You are a JSON-only assistant that MUST reply with a single valid JSON object without extra text.\n"
            "Reasoning: low\n"
            "Do not expose analysis or chain-of-thought. Respond using the final JSON only."
        )

        # 统一的默认采样参数；可被 complete(**kwargs) 覆盖
        self._defaults: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0.0,  # 仅 transformers 生效
            "return_full_text": False,       # 仅 transformers 生效
        }
        if generation_kwargs:
            self._defaults.update(generation_kwargs)

        # 按后端初始化
        if backend == "transformers":
            # 懒加载，避免对非本地用户强制安装依赖
            from transformers import AutoTokenizer, pipeline  # type: ignore
            self._tokenizer = AutoTokenizer.from_pretrained(
                model, trust_remote_code=trust_remote_code
            )
            self._pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=self._tokenizer,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
            )
            self._client = None
            self._openai_like = False

        elif backend == "openai":
            # 允许官方 OpenAI 或任意 OpenAI-兼容端点（通过 base_url）
            try:
                from openai import OpenAI  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "backend='openai' 需要安装 openai>=1.0：pip install openai"
                ) from e

            # 如果没显式传参，就从环境变量读取
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            base_url = base_url or os.getenv("OPENAI_BASE_URL")
            organization = organization or os.getenv("OPENAI_ORGANIZATION")


            self._client = OpenAI(
                api_key=api_key,  # 对兼容端点同样有效
                base_url=base_url,  # 为空则走官方 https://api.openai.com/v1
                organization=organization,
            )
            self._openai_like = True
            self._pipeline = None
            self._tokenizer = None

        else:
            raise ValueError("backend must be 'transformers' or 'openai'.")

    # -------------------------
    # 公共调用入口
    # -------------------------
    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        call_kwargs = dict(self._defaults)
        # 兼容你的原参数风格
        kwargs = dict(kwargs)
        kwargs.pop("refinement_round", None)
        call_kwargs.update(kwargs)

        # 标准 chat messages（兼容 HF chat 模型和 OpenAI）
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt},
        ]

        if self._backend == "transformers":
            return self._complete_with_transformers(messages, call_kwargs)
        else:
            return self._complete_with_openai(messages, call_kwargs)

    # -------------------------
    # 本地 transformers 流程
    # -------------------------
    def _complete_with_transformers(self, messages, call_kwargs) -> LLMResponse:
        # HF text-generation pipeline 支持 chat 格式（使用模型内置 chat template）
        outputs = self._pipeline(messages, **call_kwargs)
        text = self._postprocess_text(self._extract_text_hf(outputs))
        return LLMResponse(text=text, raw={"backend": "transformers", "outputs": outputs})

    @staticmethod
    def _extract_text_hf(outputs: Any) -> str:
        """
        兼容不同 transformers 版本的输出字段：
        - 有些返回 [{'generated_text': '...'}]
        - 有些返回 [{'generated_text': [{'role': 'assistant', 'content': '...'}]}]
        - 或者 [{'summary_text': '...'}] 等
        """
        if not outputs:
            return ""
        first = outputs[0]
        # pipeline(return_full_text=False) 常见：{'generated_text': '...'}
        if isinstance(first, dict):
            if "generated_text" in first and isinstance(first["generated_text"], str):
                return first["generated_text"]
            # Chat 模式可能返回一个消息列表
            if "generated_text" in first and isinstance(first["generated_text"], list):
                # 找 assistant 的 content
                for msg in first["generated_text"]:
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        return msg.get("content", "")
            # 兜底
            if "text" in first:
                return first["text"]
        # 兜底
        return str(first)

    # -------------------------
    # 远端 OpenAI/兼容 流程
    # -------------------------
    def _complete_with_openai(self, messages, call_kwargs) -> LLMResponse:
        """
        参数映射：
        - max_new_tokens -> max_tokens
        - temperature / top_p 原样透传
        说明：OpenAI 不支持 do_sample / return_full_text；忽略即可。
        """
        max_tokens = call_kwargs.get("max_new_tokens", None)
        temperature = call_kwargs.get("temperature", None)
        top_p = call_kwargs.get("top_p", None)

        # 允许用户透传 openai chat.completions 的其他参数（如 presence_penalty 等）
        extra = {
            k: v
            for k, v in call_kwargs.items()
            if k
            not in {
                "max_new_tokens",
                "do_sample",
                "return_full_text",
                "device_map",
                "torch_dtype",
                "trust_remote_code",
                "model",
                "messages",
                "max_tokens",
                "temperature",
                "top_p",
            }
        }

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **extra,
        )
        content = resp.choices[0].message.content if resp.choices else ""
        return LLMResponse(text=content, raw={"backend": "openai", "response": resp})

    # -------------------------
    # 通用后处理
    # -------------------------
    @staticmethod
    def _postprocess_text(text: str) -> str:
        # 按你的需要进行清洗；这里简单返回
        return text

    def _extract_text(self, outputs: Any) -> str:
        """Normalize pipeline outputs into a single string response."""
        if not outputs:
            return ""
        candidate = outputs[0]

        # Newer transformers versions return {"generated_text": [{"role": ..., "content": ...}, ...]}
        if isinstance(candidate, dict) and "generated_text" in candidate:
            generated = candidate["generated_text"]
            if isinstance(generated, list):
                # Grab the assistant role content if present.
                for message in generated:
                    if isinstance(message, dict) and message.get("role") == "assistant":
                        content = message.get("content")
                        if isinstance(content, str):
                            return content.strip()
                # Fallback to last item's content/text.
                last = generated[-1]
                if isinstance(last, dict):
                    return str(last.get("content") or last.get("text") or "")
                return str(last)
            if isinstance(generated, dict):
                return str(generated.get("content") or generated.get("text") or "")
            return str(generated)

        # Older versions might return {"generated_text": "..."}
        if isinstance(candidate, dict) and isinstance(candidate.get("generated_text"), str):
            return candidate["generated_text"].strip()

        # Ultimate fallback: string representation.
        return str(candidate).strip()

    def _postprocess_text(self, text: str) -> str:
        """Trim analyzer prefixes and isolate JSON payloads when present."""
        trimmed = text.strip()
        if not trimmed:
            return trimmed

        marker = "assistantfinal"
        if marker in trimmed:
            trimmed = trimmed.split(marker, 1)[1].strip()

        if trimmed.startswith(marker):
            trimmed = trimmed[len(marker) :].strip()

        # Attempt to extract the first JSON object substring.
        if trimmed and trimmed[0] != "{":
            start = trimmed.find("{")
            end = trimmed.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = trimmed[start : end + 1].strip()
                candidate_clean = candidate.replace("\r", " ").replace("\n", " ")
                try:
                    json.loads(candidate_clean)
                    return candidate_clean
                except json.JSONDecodeError:
                    pass

        return trimmed.replace("\r", " ").replace("\n", " ")
