# coding=utf-8
"""Legacy multimodal input types retained for import compatibility.

Multimodal training now runs through the server-capture path
(``specforge/algorithms/common/vlm_input.py`` with
``model.input_modality="multimodal"``), which transports images inside the
capture request itself. These pixel-tensor types remain unused by the
canonical training path.
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
