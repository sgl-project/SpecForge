# coding=utf-8
"""SwiGLU shared expert, optionally gated per token.

Module layout is the DeepSeek one (``shared_experts.w{1,2,3}``). With
``MoEConfig.shared_expert_gate == "sigmoid"`` (Qwen3.5/3.8 MoE) the output is
multiplied by ``sigmoid(gate(x))`` with ``gate`` a bias-free ``Linear(hidden,
1)``; the checkpoint file names it ``shared_expert_gate.weight`` (see
:mod:`.qwen_layout`).
"""

from __future__ import annotations

import torch
from torch import nn

from .config import MoEConfig
from .grouped_experts import swiglu_clamped
from .shared import SharedExpert, register_shared_expert

SHARED_EXPERT_GATES = ("none", "sigmoid")


@register_shared_expert("swiglu")
class SwiGLUSharedExpert(SharedExpert):
    def __init__(self, cfg: MoEConfig, hidden_size: int) -> None:
        super().__init__(cfg, hidden_size)
        if cfg.shared_expert_gate not in SHARED_EXPERT_GATES:
            raise ValueError(
                f"unknown shared_expert_gate {cfg.shared_expert_gate!r} for "
                f"shared_expert='swiglu'; choose from {SHARED_EXPERT_GATES}"
            )
        self.swiglu_limit = float(cfg.swiglu_limit)
        self.w1 = nn.Linear(hidden_size, self.intermediate_size, bias=False)
        self.w2 = nn.Linear(self.intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, self.intermediate_size, bias=False)
        # Per-token sigmoid gate (Qwen ``shared_expert_gate``): [1, hidden], no bias.
        self.gate = (
            nn.Linear(hidden_size, 1, bias=False)
            if cfg.shared_expert_gate == "sigmoid"
            else None
        )

    @property
    def gated(self) -> bool:
        return self.gate is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = swiglu_clamped(self.w1(x), self.w3(x), self.swiglu_limit)
        y = self.w2(h.to(x.dtype))
        if self.gate is not None:
            y = torch.sigmoid(self.gate(x).float()).to(y.dtype) * y
        return y
