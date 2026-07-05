# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Generic HuggingFace target engine, parameterized by a TargetCapturePolicy."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from .base import TargetEngine
from .target_capture_policy import TargetCapturePolicy, resolve_target_capture_policy


class HFTargetEngine(TargetEngine):
    """HF-backed frozen target for any algorithm with a registered policy."""

    backend = "hf"

    def __init__(self, model: nn.Module, policy: TargetCapturePolicy):
        self.model = model
        self.policy = policy
        self.capture_layers: Optional[List[int]] = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: Optional[torch.dtype] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        *,
        policy: "TargetCapturePolicy | str" = "eagle3",
        **kwargs,
    ) -> "HFTargetEngine":
        if isinstance(policy, str):
            policy = resolve_target_capture_policy(policy)
        model = policy.hf_load(
            pretrained_model_name_or_path, torch_dtype, device, cache_dir, **kwargs
        )
        return cls(model, policy)

    def capture(self, input_ids, attention_mask, loss_mask, **kwargs):
        return self.policy.hf_capture(
            self.model,
            self.capture_layers,
            input_ids,
            attention_mask,
            loss_mask,
            **kwargs,
        )

    def set_capture_layers(self, layer_ids: Optional[List[int]] = None) -> None:
        # config is only needed to derive defaults; explicit ids must not
        # require the wrapped module to expose one (test doubles don't).
        config = self.model.config if layer_ids is None else None
        self.capture_layers = self.policy.resolve_capture_layers(config, layer_ids)


__all__ = ["HFTargetEngine"]
