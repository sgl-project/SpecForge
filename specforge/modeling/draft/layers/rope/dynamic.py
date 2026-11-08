"""Dynamic NTK for complete position sets, with no cross-request cache state."""

import torch
from torch import Tensor

from specforge.config.draft.rope import DynamicNTKRoPESpec

from ..registry import register_component_builder
from .base import RotaryEmbedding


class DynamicNTKRotaryEmbedding(RotaryEmbedding):
    """Use max(position_ids) + 1 as the shared length of this batch.

    Positions must cover the complete attention sequence, including context.
    Prepare them together before selecting query/key embeddings. Incremental
    decoding with previously rotated KV caches needs a separate cache contract
    and is not supported by this stateless training component.
    """

    def __init__(self, config: DynamicNTKRoPESpec, *, head_dim: int):
        if type(config) is not DynamicNTKRoPESpec:
            raise TypeError("DynamicNTKRotaryEmbedding requires DynamicNTKRoPESpec")
        super().__init__(config, head_dim=head_dim)
        self.head_dim = head_dim

    @torch.no_grad()
    def forward(self, hidden: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        if position_ids.ndim != 2:
            raise ValueError("position_ids must have shape [batch, sequence]")
        # Resolve the scalar length once. No cached maximum may leak between
        # independent training batches; empty inputs use the original base.
        length = int(position_ids.max().item()) + 1 if position_ids.numel() else 0
        length = max(length, self.config.original_max_position_embeddings)
        theta = self.config.scaled_theta(self.head_dim, length)
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            inv_freq = 1.0 / (
                theta
                ** (
                    torch.arange(
                        0, self.rotary_dim, 2, dtype=torch.float32, device=hidden.device
                    )
                    / self.rotary_dim
                )
            )
            positions = position_ids.to(device=hidden.device, dtype=torch.float32)
            freqs = positions.unsqueeze(-1) * inv_freq
            cos, sin = self._cos_sin(torch.cat((freqs, freqs), dim=-1))
            return cos.to(hidden.dtype), sin.to(hidden.dtype)


register_component_builder("rope", DynamicNTKRoPESpec, DynamicNTKRotaryEmbedding)
