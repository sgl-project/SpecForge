# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Generic custom-backend target engine, parameterized by a CapturePolicy.

The custom backend loads architectures through ``AutoDistributedTargetModel``
(SpecForge's own model implementations with a bespoke inference plan). Only
policies implementing ``custom_capture`` support it (EAGLE3 today).
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from .base import TargetEngine
from .capture_policy import CapturePolicy, resolve_capture_policy


class CustomTargetEngine(TargetEngine):
    """Custom-architecture frozen target for policies with a custom capture."""

    backend = "custom"

    def __init__(self, model: nn.Module, policy: CapturePolicy):
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
        policy: "CapturePolicy | str" = "eagle3",
        **kwargs,
    ) -> "CustomTargetEngine":
        from specforge.modeling.auto import AutoDistributedTargetModel

        if isinstance(policy, str):
            policy = resolve_capture_policy(policy)
        model = AutoDistributedTargetModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            device=device,
            **kwargs,
        )
        return cls(model, policy)

    def capture(self, input_ids, attention_mask, loss_mask, **kwargs):
        return self.policy.custom_capture(
            self.model,
            self.capture_layers,
            input_ids,
            attention_mask,
            loss_mask,
            **kwargs,
        )

    def set_capture_layers(self, layer_ids: Optional[List[int]] = None) -> None:
        self.capture_layers = self.policy.resolve_capture_layers(
            self.model.config, layer_ids
        )


__all__ = ["CustomTargetEngine"]
