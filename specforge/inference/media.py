# coding=utf-8
"""Legacy multimodal input types retained for import compatibility.

The current runtime does not support VLM training or media rollout inputs.
These types are not wired into the canonical training path.
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
