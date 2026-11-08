"""Position-selected rotary embeddings, without family configuration or caches."""

import torch
from torch import Tensor, nn

from specforge.config.draft.rope import DefaultRoPESpec


class DefaultRotaryEmbedding(nn.Module):
    """Return [batch, sequence, rotary_dim] cosine/sine for explicit positions.

    Frequencies are rebuilt in FP32 on the input device. They are derived state,
    not checkpoint weights, and cannot be rounded by module.half()/bfloat16() or
    left uninitialized by meta-device loading. No sequence-length cache is kept.
    """

    def __init__(self, config: DefaultRoPESpec, *, head_dim: int):
        super().__init__()
        if type(config) is not DefaultRoPESpec:
            raise TypeError("DefaultRotaryEmbedding requires DefaultRoPESpec")
        self.config = config
        self.rotary_dim = config.rotary_dim(head_dim)

    @torch.no_grad()
    def forward(self, hidden: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        if position_ids.ndim != 2:
            raise ValueError("position_ids must have shape [batch, sequence]")
        # Keep long-context phase computation out of mixed-precision autocast.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            inv_freq = 1.0 / (
                self.config.rope_theta
                ** (
                    torch.arange(
                        0, self.rotary_dim, 2, dtype=torch.float32, device=hidden.device
                    )
                    / self.rotary_dim
                )
            )
            positions = position_ids.to(device=hidden.device, dtype=torch.float32)
            freqs = positions.unsqueeze(-1) * inv_freq
            angles = torch.cat((freqs, freqs), dim=-1)
            return angles.cos().to(hidden.dtype), angles.sin().to(hidden.dtype)
