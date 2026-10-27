"""DeepSeek-V4.1 single-pass mHC, expressed in differentiable PyTorch.

Matches inference/model.py Block and inference/kernel.py hc_split_sinkhorn:
a sublayer reads the incoming pre_mix and produces the *next* pre_mix.
Coefficient projection, Sinkhorn and stream accumulation use float32.
"""

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from .base import HyperConnectionState, Residual, collapse_streams


class MHCWriteState(NamedTuple):
    streams: torch.Tensor
    next_pre: torch.Tensor
    post: torch.Tensor
    combination: torch.Tensor


class ManifoldHyperConnection(Residual[MHCWriteState]):
    def __init__(
        self,
        hidden_size: int,
        count: int,
        norm_eps: float,
        sinkhorn_iters: int,
        eps: float,
    ):
        super().__init__()
        self.count = count
        self.norm_eps = norm_eps
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps
        size = (count + 2) * count
        self.mix_weight = nn.Parameter(
            torch.empty(size, count * hidden_size, dtype=torch.float32)
        )
        self.mix_scale = nn.Parameter(torch.full((3,), 0.01, dtype=torch.float32))
        self.mix_bias = nn.Parameter(torch.zeros(size, dtype=torch.float32))
        nn.init.normal_(self.mix_weight, std=(count * hidden_size) ** -0.5)
        # A finite identity-biased initial residual routing. These are training
        # defaults; the upstream inference code loads all three tensors.
        with torch.no_grad():
            self.mix_bias[2 * count :].view(count, count).copy_(torch.eye(count) * 4)

    def mixes(
        self, streams: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Disabling autocast is necessary: casting inputs alone does not prevent
        # an autocast linear from rounding the coefficient projection to bf16.
        with torch.autocast(device_type=streams.device.type, enabled=False):
            flat = streams.flatten(-2).float()
            scores = F.linear(flat, self.mix_weight.float()) * torch.rsqrt(
                flat.square().mean(-1, keepdim=True) + self.norm_eps
            )
            n = self.count
            pre = (
                scores[..., :n] * self.mix_scale[0].float() + self.mix_bias[:n].float()
            ).sigmoid() + self.eps
            post = (
                2
                * (
                    scores[..., n : 2 * n] * self.mix_scale[1].float()
                    + self.mix_bias[n : 2 * n].float()
                ).sigmoid()
            )
            matrix = (
                scores[..., 2 * n :] * self.mix_scale[2].float()
                + self.mix_bias[2 * n :].float()
            ).unflatten(-1, (n, n))
            matrix = matrix.softmax(-1) + self.eps
            matrix = matrix / (matrix.sum(-2, keepdim=True) + self.eps)
            for _ in range(self.sinkhorn_iters - 1):
                matrix = matrix / (matrix.sum(-1, keepdim=True) + self.eps)
                matrix = matrix / (matrix.sum(-2, keepdim=True) + self.eps)
        return pre, post, matrix

    def read(
        self, state: HyperConnectionState, norm: nn.Module
    ) -> tuple[torch.Tensor, MHCWriteState]:
        pre, post, combination = self.mixes(state.streams)
        return norm(collapse_streams(state)), MHCWriteState(
            state.streams, pre, post, combination
        )

    def write(self, update: torch.Tensor, state: MHCWriteState) -> HyperConnectionState:
        # Matrix indices are [source stream, destination stream].
        mixed = (
            state.combination.unsqueeze(-1) * state.streams.float().unsqueeze(-2)
        ).sum(-3)
        streams = mixed + state.post.unsqueeze(-1) * update.float().unsqueeze(-2)
        return HyperConnectionState(streams.to(update.dtype), state.next_pre)
