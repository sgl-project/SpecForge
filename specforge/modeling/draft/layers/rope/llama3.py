"""Llama3 wavelength-dependent frequency scaling, including step boundaries."""

import math

import torch
from torch import Tensor

from specforge.config.draft.rope import Llama3RoPESpec

from .base import RotaryEmbedding


class Llama3RotaryEmbedding(RotaryEmbedding):
    def __init__(self, config: Llama3RoPESpec, *, head_dim: int):
        if type(config) is not Llama3RoPESpec:
            raise TypeError("Llama3RotaryEmbedding requires Llama3RoPESpec")
        super().__init__(config, head_dim=head_dim)

    def _frequencies(self, device: torch.device) -> Tensor:
        inv_freq = super()._frequencies(device)
        node = self.config
        wavelength = 2 * math.pi / inv_freq
        low = node.original_max_position_embeddings / node.low_freq_factor
        high = node.original_max_position_embeddings / node.high_freq_factor
        smooth = 0
        if node.high_freq_factor != node.low_freq_factor:
            smooth = (
                node.original_max_position_embeddings / wavelength
                - node.low_freq_factor
            ) / (node.high_freq_factor - node.low_freq_factor)
        return torch.where(
            wavelength < high,
            inv_freq,
            torch.where(
                wavelength > low,
                inv_freq / node.factor,
                (1 - smooth) * inv_freq / node.factor + smooth * inv_freq,
            ),
        )
