"""Select temporal/height/width rotary sections before attention sees them."""

import torch
from torch import Tensor

from specforge.config.draft.rope import MultiAxisRoPESpec

from .base import RotaryEmbedding


class MultiAxisRotaryEmbedding(RotaryEmbedding):
    def __init__(self, config: MultiAxisRoPESpec, *, head_dim: int):
        if type(config) is not MultiAxisRoPESpec:
            raise TypeError("MultiAxisRotaryEmbedding requires MultiAxisRoPESpec")
        super().__init__(config, head_dim=head_dim)

    @torch.no_grad()
    def forward(self, hidden: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Accept [3, batch, sequence], or [batch, sequence] for text-only input.

        Return the same [batch, sequence, rotary_dim] layout as other rotary
        components. The shared apply_rotary also handles an unrotated head tail.
        """
        if position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        if position_ids.ndim != 3 or position_ids.shape[0] != 3:
            raise ValueError(
                "multi-axis position_ids must have shape [3, batch, sequence] or [batch, sequence]"
            )
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            frequencies = self._frequencies(hidden.device)
            positions = position_ids.to(device=hidden.device, dtype=torch.float32)
            pairs = frequencies.numel()
            if self.config.mrope_interleaved:
                axes = torch.zeros(pairs, dtype=torch.long, device=hidden.device)
                for axis in (1, 2):
                    axes[axis : 3 * self.config.mrope_section[axis] : 3] = axis
            else:
                axes = torch.tensor(
                    [
                        axis
                        for axis, size in enumerate(self.config.mrope_section)
                        for _ in range(size)
                    ],
                    device=hidden.device,
                )
            freqs = positions[axes].permute(1, 2, 0) * frequencies
            cos, sin = self._cos_sin(torch.cat((freqs, freqs), dim=-1))
            return cos.to(hidden.dtype), sin.to(hidden.dtype)
