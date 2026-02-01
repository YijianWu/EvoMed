"""LLM client abstractions used by Engine components."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
import json
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Union, Literal


@dataclass
class LLMResponse:
    """Container for LLM outputs."""

    text: str
    raw: Optional[Dict[str, Any]] = None


class LLMClient(ABC):
    """Abstract interface so Engine can plug into any chat/completions API."""

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


class UniversalLLMClient(LLMClient):
    """
    Universal LLM client: supports both local transformers and remote OpenAI/compatible APIs.

    - backend="transformers": Uses local HF pipeline for inference
    - backend="openai": Uses official openai SDK for chat.completions
      * Also supports OpenAI-compatible base_url (e.g., vLLM / OpenRouter)

    Unified Entry Point:
        complete(prompt: str, **kwargs) -> LLMResponse
    """

    def __init__(
        self,
        model: str,
        *,
        backend: Literal["transformers", "openai"] = "transformers",
        # ---- Common generation parameters ----
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.9,
        system_prompt: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        # ---- Local transformers specific ----
        device_map: Union[str, Dict[str, int]] = "auto",
        torch_dtype: Union[str, "torch.dtype"] = "auto",
        trust_remote_code: bool = True,
        # ---- Remote OpenAI/compatible specific ----
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,  # For compatible endpoints, e.g., http://localhost:8000/v1
        organization: Optional[str] = None,
    ) -> None:
        super().__init__(model=model)
        self._backend = backend
        self._system_prompt = system_prompt or (
            "You are a JSON-only assistant that MUST reply with a single valid JSON object without extra text.\n"
            "Reasoning: low\n"
            "Do not expose analysis or chain-of-thought. Respond using the final JSON only."
        )

        # Unified default sampling parameters; can be overridden by complete(**kwargs)
        self._defaults: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0.0,  # transformers only
            "return_full_text": False,       # transformers only
        }
        if generation_kwargs:
            self._defaults.update(generation_kwargs)

        # Initialize by backend
        if backend == "transformers":
            # Lazy load to avoid mandatory dependency for non-local users
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
            # Allow official OpenAI or any OpenAI-compatible endpoint (via base_url)
            try:
                from openai import OpenAI  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "backend='openai' requires installation of openai>=1.0: pip install openai"
                ) from e

            # Read from environment variables if not explicitly passed
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            base_url = base_url or os.getenv("OPENAI_BASE_URL")
            organization = organization or os.getenv("OPENAI_ORGANIZATION")

            self._client = OpenAI(
                api_key=api_key,  # Also valid for compatible endpoints
                base_url=base_url,  # If empty, goes to official https://api.openai.com/v1
                organization=organization,
            )
            self._openai_like = True
            self._pipeline = None
            self._tokenizer = None

        else:
            raise ValueError("backend must be 'transformers' or 'openai'.")

    # -------------------------
    # Public Call Entry
    # -------------------------
    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        call_kwargs = dict(self._defaults)
        kwargs = dict(kwargs)
        kwargs.pop("refinement_round", None)
        call_kwargs.update(kwargs)

        # Standard chat messages (compatible with HF chat models and OpenAI)
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt},
        ]

        if self._backend == "transformers":
            return self._complete_with_transformers(messages, call_kwargs)
        else:
            return self._complete_with_openai(messages, call_kwargs)

    # -------------------------
    # Local transformers process
    # -------------------------
    def _complete_with_transformers(self, messages, call_kwargs) -> LLMResponse:
        # HF text-generation pipeline supports chat format (using model's built-in chat template)
        outputs = self._pipeline(messages, **call_kwargs)
        text = self._postprocess_text(self._extract_text_hf(outputs))
        return LLMResponse(text=text, raw={"backend": "transformers", "outputs": outputs})

    @staticmethod
    def _extract_text_hf(outputs: Any) -> str:
        """
        Compatible with different transformers version output fields:
        - Some return [{'generated_text': '...'}]
        - Some return [{'generated_text': [{'role': 'assistant', 'content': '...'}]}]
        - Or [{'summary_text': '...'}] etc.
        """
        if not outputs:
            return ""
        first = outputs[0]
        # pipeline(return_full_text=False) common: {'generated_text': '...'}
        if isinstance(first, dict):
            if "generated_text" in first and isinstance(first["generated_text"], str):
                return first["generated_text"]
            # Chat mode might return a list of messages
            if "generated_text" in first and isinstance(first["generated_text"], list):
                # Find content of assistant role
                for msg in first["generated_text"]:
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        return msg.get("content", "")
            # Fallback
            if "text" in first:
                return first["text"]
        # Fallback
        return str(first)

    # -------------------------
    # Remote OpenAI/Compatible process
    # -------------------------
    def _complete_with_openai(self, messages, call_kwargs) -> LLMResponse:
        """
        Parameter mapping:
        - max_new_tokens -> max_tokens
        - temperature / top_p passed as is
        Note: OpenAI does not support do_sample / return_full_text; ignore them.
        """
        max_tokens = call_kwargs.get("max_new_tokens", None)
        temperature = call_kwargs.get("temperature", None)
        top_p = call_kwargs.get("top_p", None)

        # Allow user to pass other parameters for openai chat.completions (e.g., presence_penalty)
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
    # Common post-processing
    # -------------------------
    @staticmethod
    def _postprocess_text(text: str) -> str:
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
