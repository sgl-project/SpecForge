"""Shared FP32 phase computation for static rotary frequency rules."""

from contextlib import nullcontext

import torch
from torch import Tensor, nn

from specforge.config.draft.rope.base import RotarySpec


class RotaryEmbedding(nn.Module):
    """FP32 frequencies are cached per device, never cast or checkpointed.

    This avoids rounding them during module casts or leaving uninitialized
    buffers after meta loading. Subclasses own frequency and amplitude rules.
    """

    def __init__(self, config: RotarySpec, *, head_dim: int):
        super().__init__()
        self.config = config
        self.head_dim = head_dim
        self.rotary_dim = config.rotary_dim(head_dim)
        self._frequency_cache = {}

    def _frequencies(self, device: torch.device) -> Tensor:
        return 1.0 / (
            self.config.rope_theta
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float32, device=device)
                / self.rotary_dim
            )
        )

    def _positions(self, position_ids: Tensor, device: torch.device) -> Tensor:
        return position_ids.to(device=device, dtype=torch.float32)

    def frequencies(self, device: torch.device) -> Tensor:
        if device not in self._frequency_cache:
            self._frequency_cache[device] = self._frequencies(device)
        return self._frequency_cache[device]

    def _apply(self, fn, recurse=True):
        # A module move/cast invalidates derived tensors, including meta loading.
        self._frequency_cache.clear()
        return super()._apply(fn, recurse)

    def _cos_sin(self, angles: Tensor) -> tuple[Tensor, Tensor]:
        return angles.cos(), angles.sin()

    @torch.no_grad()
    def forward(self, hidden: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Return [batch, sequence, rotary_dim] cosine/sine for explicit positions."""
        if position_ids.ndim != 2:
            raise ValueError("position_ids must have shape [batch, sequence]")
        with (
            nullcontext()
            if hidden.is_meta
            else torch.autocast(device_type=hidden.device.type, enabled=False)
        ):
            inv_freq = self.frequencies(hidden.device)
            positions = self._positions(position_ids, hidden.device)
            freqs = positions.unsqueeze(-1) * inv_freq
            angles = torch.cat((freqs, freqs), dim=-1)
            cos, sin = self._cos_sin(angles)
            return cos, sin
