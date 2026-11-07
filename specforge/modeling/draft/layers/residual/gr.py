"""Trainable Qwen3.8-Flash-Next gated residual, without inference-only kernels.

The branch-wise Gemma RMSNorm, low-rank read gate and injection gate follow
SGLang's hyperconnection.GatedResidual. No extra norm follows the gated read.
"""

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from .base import HyperConnectionState, Residual


class GRWriteState(NamedTuple):
    streams: torch.Tensor
    normalized: torch.Tensor


class GatedResidual(Residual[GRWriteState]):
    def __init__(
        self,
        hidden_size: int,
        count: int,
        rank: int,
        eps: float,
        *,
        readout: bool = False,
    ):
        super().__init__()
        self.count = count
        self.eps = eps
        # Separate weights per stream, zero-centred Gemma parameterization.
        self.norm_weight = nn.Parameter(torch.zeros(count, hidden_size))
        self.input_mix_weight_down = nn.Linear(count * hidden_size, rank, bias=False)
        self.input_mix_weight_up = nn.Linear(rank, count * hidden_size, bias=False)
        self.block_inject_weight = (
            None if readout else nn.Linear(count * hidden_size, count, bias=False)
        )

    def read(
        self, state: HyperConnectionState, norm: nn.Module
    ) -> tuple[torch.Tensor, GRWriteState]:
        return self.mix(state)

    def mix(self, state: HyperConnectionState) -> tuple[torch.Tensor, GRWriteState]:
        x = state.streams.float()
        normalized = (
            x
            * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
            * (1 + self.norm_weight.float())
        ).to(state.streams.dtype)
        flat = normalized.flatten(-2)
        gate = self.input_mix_weight_up(
            F.silu(self.input_mix_weight_down(flat) / self.count)
        ).sigmoid()
        mixed = (gate.view_as(normalized) * normalized).mean(-2)
        return mixed, GRWriteState(state.streams, flat)

    def write(self, update: torch.Tensor, state: GRWriteState) -> HyperConnectionState:
        if self.block_inject_weight is None:
            raise RuntimeError("A GR readout has no residual write")
        gate = 2 * (self.block_inject_weight(state.normalized) / self.count).sigmoid()
        return HyperConnectionState(
            state.streams + gate.unsqueeze(-1) * update.unsqueeze(-2)
        )

    def forward(self, state: HyperConnectionState) -> torch.Tensor:
        return self.mix(state)[0]
