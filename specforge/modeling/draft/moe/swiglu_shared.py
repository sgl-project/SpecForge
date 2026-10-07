# coding=utf-8
"""Ungated SwiGLU shared expert (DeepSeek layout ``shared_experts.w{1,2,3}``)."""

from __future__ import annotations

import torch
from torch import nn

from .config import MoEConfig
from .grouped_experts import swiglu_clamped
from .shared import SharedExpert, register_shared_expert


@register_shared_expert("swiglu")
class SwiGLUSharedExpert(SharedExpert):
    def __init__(self, cfg: MoEConfig, hidden_size: int) -> None:
        super().__init__(cfg, hidden_size)
        if cfg.shared_expert_gate != "none":
            raise ValueError(
                "shared_expert='swiglu' is ungated; shared_expert_gate="
                f"{cfg.shared_expert_gate!r} needs a gated shared-expert implementation"
            )
        self.swiglu_limit = float(cfg.swiglu_limit)
        self.w1 = nn.Linear(hidden_size, self.intermediate_size, bias=False)
        self.w2 = nn.Linear(self.intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, self.intermediate_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = swiglu_clamped(self.w1(x), self.w3(x), self.swiglu_limit)
        return self.w2(h.to(x.dtype))
