"""YaRN interpolation and rotary amplitude from fully resolved settings."""

import math

import torch
from torch import Tensor

from specforge.config.draft.rope import YaRNSpec

from ..registry import register_component_builder
from .base import RotaryEmbedding


class YaRNRotaryEmbedding(RotaryEmbedding):
    def __init__(self, config: YaRNSpec, *, head_dim: int):
        if type(config) is not YaRNSpec:
            raise TypeError("YaRNRotaryEmbedding requires YaRNSpec")
        super().__init__(config, head_dim=head_dim)

    def _frequencies(self, device: torch.device) -> Tensor:
        node, dim = self.config, self.rotary_dim
        pos_freqs = node.rope_theta ** (
            torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim
        )
        extrapolation = 1.0 / pos_freqs
        interpolation = 1.0 / (node.factor * pos_freqs)

        def correction(rotations):
            return (
                dim
                * math.log(
                    node.original_max_position_embeddings / (rotations * 2 * math.pi)
                )
            ) / (2 * math.log(node.rope_theta))

        low, high = correction(node.beta_fast), correction(node.beta_slow)
        if node.truncate:
            low, high = math.floor(low), math.ceil(high)
        low, high = max(low, 0), min(high, dim - 1)
        if low == high:
            high += 0.001
        # Match the reference FP32 ramp before transferring to the input device.
        ramp = (
            (torch.arange(dim // 2, dtype=torch.float32, device="cpu") - low)
            / (high - low)
        ).clamp(0, 1)
        mask = 1.0 - ramp.to(device)
        return interpolation * (1 - mask) + extrapolation * mask

    def _cos_sin(self, angles: Tensor) -> tuple[Tensor, Tensor]:
        cos, sin = super()._cos_sin(angles)
        # mscale_all_dim belongs to MLA's full QK score, not this rotary prefix.
        # Apply the resolved amplitude in FP32, before casting to the input dtype.
        return cos * self.config.attention_factor, sin * self.config.attention_factor


register_component_builder("rope", YaRNSpec, YaRNRotaryEmbedding)
