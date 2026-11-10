"""Adjacent-token injection, independent of the model loop and training policy."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers import initialization as init

from specforge.config.draft._validation import positive_int
from specforge.config.draft.architecture.dspine import AdjacentInjectionSpec


class AdjacentInjection(nn.Module):
    """Per-layer readout/write parameters with shared message and gating weights.

    The caller owns execution order, token codes, and replacement policy. Pass
    a replacement mask only for layers where replacement applies; schedules and
    layer-count thresholds are not properties of this computational head.
    """

    def __init__(
        self,
        config: AdjacentInjectionSpec,
        *,
        hidden_size: int,
        num_layers: int,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if type(config) is not AdjacentInjectionSpec:
            raise TypeError("AdjacentInjection requires AdjacentInjectionSpec")
        positive_int("hidden_size", hidden_size)
        positive_int("num_layers", num_layers)
        self.config = config
        options = dict(device=device, dtype=dtype)
        self.rank = config.transfer_rank
        self.readouts = nn.ModuleList(
            nn.Linear(hidden_size, self.rank, bias=False, **options)
            for _ in range(num_layers)
        )
        self.writes = nn.ModuleList(
            nn.Linear(config.message_size, hidden_size, bias=False, **options)
            for _ in range(num_layers)
        )
        self.message = nn.Linear(self.rank, config.message_size, bias=False, **options)
        self.receiver_gate = nn.Linear(
            self.rank, config.message_size, bias=False, **options
        )
        self.predecessor_gate = nn.Linear(
            self.rank, config.message_size, bias=False, **options
        )
        self.bias = nn.Parameter(torch.empty(config.message_size, **options))
        self._zero_identity_paths()

    def reset_draft_parameters(self):
        # Legacy builds this head after post_init, retaining nn.Linear defaults.
        for linear in (self.message, self.receiver_gate, *self.readouts):
            init.kaiming_uniform_(linear.weight, a=math.sqrt(5))
        self._zero_identity_paths()

    def _zero_identity_paths(self):
        # HF initialization helpers preserve tensors restored from a checkpoint.
        init.zeros_(self.bias)
        for write in self.writes:
            init.zeros_(write.weight)
        init.zeros_(self.predecessor_gate.weight)

    def code(self, token_ids: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        return math.sqrt(self.rank) * F.embedding(token_ids, codes)

    def feature(self, hidden: torch.Tensor, layer: int) -> torch.Tensor:
        projected = self.readouts[layer](hidden)
        return (math.sqrt(self.rank) * F.normalize(projected.float(), dim=-1)).to(
            hidden.dtype
        )

    def write(
        self,
        hidden: torch.Tensor,
        receiver: torch.Tensor,
        predecessor: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        # FSDP may keep the token-code buffer in FP32 while projections use BF16.
        predecessor = predecessor.to(receiver.dtype)
        gate = torch.sigmoid(
            self.receiver_gate(receiver)
            + self.predecessor_gate(predecessor)
            + self.bias
        )
        delta = self.writes[layer](gate * self.message(predecessor))
        scale = (
            torch.linalg.vector_norm(hidden.float(), dim=-1, keepdim=True)
            / math.sqrt(hidden.shape[-1])
        ).to(hidden.dtype)
        return hidden + scale * delta

    def forward(
        self,
        hidden: torch.Tensor,
        layer: int,
        token_ids: torch.Tensor,
        codes: torch.Tensor,
        replace: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = self.feature(hidden, layer)
        anchor = self.code(token_ids[..., :1], codes)
        predecessors = torch.cat((anchor, features[..., 1:-1, :]), dim=-2)
        if replace is not None:
            correct = self.code(token_ids[..., :-1], codes)
            predecessors = torch.where(replace[..., None, None], correct, predecessors)
        successors = self.write(
            hidden[..., 1:, :], features[..., 1:, :], predecessors, layer
        )
        return torch.cat((hidden[..., :1, :], successors), dim=-2)
