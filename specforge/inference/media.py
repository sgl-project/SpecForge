# coding=utf-8
"""Typed, inference-local multimodal inputs.

Media tensors are created inside rollout and consumed by a target engine. They
never enter PromptTask, the control plane, or FeatureStore.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class MediaInputs:
    """One target-capture batch of prepared image inputs."""

    pixel_values: torch.Tensor
    image_grid_thw: Tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class PreparedTargetInput:
    """One tokenized prompt plus optional ephemeral media tensors."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    loss_mask: torch.Tensor
    media: MediaInputs | None = None


__all__ = ["MediaInputs", "PreparedTargetInput"]
