"""SGLang linear scaling: divide positions before computing phases."""

import torch
from torch import Tensor

from specforge.config.draft.rope import LinearRoPESpec

from ..registry import register_component_builder
from .base import RotaryEmbedding


class LinearRotaryEmbedding(RotaryEmbedding):
    def __init__(self, config: LinearRoPESpec, *, head_dim: int):
        if type(config) is not LinearRoPESpec:
            raise TypeError("LinearRotaryEmbedding requires LinearRoPESpec")
        super().__init__(config, head_dim=head_dim)

    def _positions(self, position_ids: Tensor, device: torch.device) -> Tensor:
        return super()._positions(position_ids, device) / self.config.factor


register_component_builder("rope", LinearRoPESpec, LinearRotaryEmbedding)
