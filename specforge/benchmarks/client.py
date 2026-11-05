# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Prompt rendering and the SGLang ``/generate`` client.

Prompts are rendered client-side with the target tokenizer's chat template so
that the raw ``/generate`` endpoint can be used; it is the one that returns
speculative-decoding telemetry (``spec_verify_ct``, ``spec_accept_length``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class PromptRenderer:
    """Turn chat messages into prompt text with the model's chat template."""

    def __init__(
        self,
        model: str,
        trust_remote_code: bool = False,
        enable_thinking: bool = False,
    ):
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            model, trust_remote_code=trust_remote_code
        )
        self._enable_thinking = enable_thinking

    def render(self, messages: List[Dict[str, Any]]) -> str:
        kwargs: Dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": self._enable_thinking,
        }
        try:
            return self._tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            # Templates without a thinking switch reject the extra kwarg.
            kwargs.pop("enable_thinking")
            return self._tokenizer.apply_chat_template(messages, **kwargs)


@dataclass(frozen=True)
class Generation:
    """One ``/generate`` response reduced to what the runner needs."""

    text: str
    completion_tokens: int
    #: Number of draft verification rounds; ``None`` without speculation.
    spec_verify_count: Optional[int] = None
    #: Server-reported mean accepted tokens per verify; ``None`` without speculation.
    spec_accept_length: Optional[float] = None

    @classmethod
    def from_response(cls, payload: Dict[str, Any]) -> "Generation":
        meta = payload.get("meta_info") or {}
        verify = meta.get("spec_verify_ct")
        accept = meta.get("spec_accept_length")
        return cls(
            text=payload.get("text", ""),
            completion_tokens=int(meta.get("completion_tokens", 0)),
            spec_verify_count=int(verify) if verify is not None else None,
            spec_accept_length=float(accept) if accept is not None else None,
        )


class SGLangClient:
    """Thin HTTP client for the endpoints the benchmark uses."""

    def __init__(self, base_url: str, timeout_seconds: float = 3600.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def is_ready(self) -> bool:
        import requests

        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
        except requests.RequestException:
            return False
        return response.ok

    def flush_cache(self) -> bool:
        """Drop the prefix cache; returns False (and keeps going) on failure."""
        import requests

        try:
            response = requests.get(
                f"{self.base_url}/flush_cache", timeout=min(self.timeout_seconds, 60)
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"Warning: /flush_cache failed ({exc}); continuing.")
            return False
        return True

    def generate(
        self,
        prompt: str,
        sampling_params: Dict[str, Any],
        image_path: Optional[str] = None,
    ) -> Generation:
        import requests

        body: Dict[str, Any] = {"text": prompt, "sampling_params": sampling_params}
        if image_path is not None:
            body["image_data"] = image_path
        response = requests.post(
            f"{self.base_url}/generate", json=body, timeout=self.timeout_seconds
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, list):
            payload = payload[0]
        return Generation.from_response(payload)


__all__ = ["Generation", "PromptRenderer", "SGLangClient"]
