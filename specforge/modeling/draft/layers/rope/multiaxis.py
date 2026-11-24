"""Select temporal/height/width rotary sections before attention sees them."""

from contextlib import nullcontext

import torch
from torch import Tensor

from specforge.config.draft.rope import MultiAxisRoPESpec

from ..registry import register_component_builder
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
        with (
            nullcontext()
            if hidden.is_meta
            else torch.autocast(device_type=hidden.device.type, enabled=False)
        ):
            frequencies = self.frequencies(hidden.device)
            positions = position_ids.to(device=hidden.device, dtype=torch.float32)
            if self.config.mrope_interleaved:
                # SGLang's interleaved layout defaults to T, with H/W at 1::3/2::3.
                axes = torch.zeros(
                    frequencies.numel(), dtype=torch.long, device=hidden.device
                )
                for axis in (1, 2):
                    axes[axis : 3 * self.config.mrope_section[axis] : 3] = axis
                freqs = positions[axes].permute(1, 2, 0) * frequencies
            else:
                freqs = torch.cat(
                    [
                        positions[axis].unsqueeze(-1) * section
                        for axis, section in enumerate(
                            frequencies.split(self.config.mrope_section)
                        )
                    ],
                    dim=-1,
                )
            cos, sin = self._cos_sin(torch.cat((freqs, freqs), dim=-1))
            return cos, sin


register_component_builder("rope", MultiAxisRoPESpec, MultiAxisRotaryEmbedding)
