# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Generic SGLang target engine, parameterized by a TargetCapturePolicy.

Composes :class:`SGLangCaptureBackend` (the sglang-version-pinned boundary) and
imports zero sglang itself; the policy's ``spec.sglang_build_kwargs`` carries
the per-algorithm build flags.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from .base import TargetEngine
from .target_capture_policy import TargetCapturePolicy, resolve_target_capture_policy


class SGLangTargetEngine(TargetEngine):
    """In-process SGLang frozen target for any algorithm with a registered policy."""

    backend = "sglang"

    def __init__(self, backend, policy: TargetCapturePolicy):
        self._backend = backend  # sglang_backend.SGLangCaptureBackend
        self.policy = policy
        self.capture_layers: Optional[List[int]] = None

    @property
    def model_runner(self):
        """The underlying sglang ModelRunner."""
        return self._backend.model_runner

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: Optional[torch.dtype] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        *,
        policy: "TargetCapturePolicy | str" = "eagle3",
        trust_remote_code: bool = False,
        **kwargs,
    ) -> "SGLangTargetEngine":
        # Lazy import: the entire sglang-version coupling lives in the backend.
        from .sglang_backend import SGLangCaptureBackend

        if isinstance(policy, str):
            policy = resolve_target_capture_policy(policy)
        backend = SGLangCaptureBackend.build(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            **policy.spec.sglang_build_kwargs,
            **kwargs,
        )
        return cls(backend, policy)

    def capture(self, input_ids, attention_mask, loss_mask, **kwargs):
        return self.policy.sglang_capture(
            self._backend, input_ids, attention_mask, loss_mask, **kwargs
        )

    def get_rope_index(self, **kwargs):
        """Return M-RoPE indices for a multimodal target batch."""
        return self._backend.get_rope_index(**kwargs)

    def set_capture_layers(self, layer_ids: Optional[List[int]] = None) -> None:
        self.capture_layers = layer_ids
        self._backend.set_eagle3_capture_layers(
            layer_ids, if_supported=not self.policy.spec.sglang_strict_capture_layers
        )


__all__ = ["SGLangTargetEngine"]
