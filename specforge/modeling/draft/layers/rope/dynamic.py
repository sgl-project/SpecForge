"""SGLang dynamic NTK uses fixed frequencies, independent of batch positions."""

import torch
from torch import Tensor

from specforge.config.draft.rope import DynamicNTKRoPESpec

from ..registry import register_component_builder
from .base import RotaryEmbedding


class DynamicNTKRotaryEmbedding(RotaryEmbedding):
    def __init__(self, config: DynamicNTKRoPESpec, *, head_dim: int):
        if type(config) is not DynamicNTKRoPESpec:
            raise TypeError("DynamicNTKRotaryEmbedding requires DynamicNTKRoPESpec")
        super().__init__(config, head_dim=head_dim)

    def _frequencies(self, device: torch.device) -> Tensor:
        theta = self.config.scaled_theta(self.head_dim)
        return 1.0 / (
            theta
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float32, device=device)
                / self.rotary_dim
            )
        )


register_component_builder("rope", DynamicNTKRoPESpec, DynamicNTKRotaryEmbedding)
