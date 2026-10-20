# coding=utf-8
"""Shared expert: an always-on SwiGLU FFN added to the routed output."""

from __future__ import annotations

import abc
from typing import Type

import torch
from torch import nn

from .base import Registry, TrainingState
from .config import MoEConfig
from .experts import swiglu


class SharedExpert(TrainingState, nn.Module, abc.ABC):
    """An always-on FFN, ``shared_expert_intermediate_size`` wide."""

    def __init__(self, cfg: MoEConfig, hidden_size: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.hidden_size = hidden_size
        self.intermediate_size = cfg.shared_expert_intermediate_size

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` is ``[T, hidden]``; return ``[T, hidden]`` in ``x.dtype``."""


SHARED_EXPERTS: Registry[Type[SharedExpert]] = Registry("MoE shared expert")


def register_shared_expert(name: str):
    return SHARED_EXPERTS.register(name)


def build_shared_expert(cfg: MoEConfig, hidden_size: int) -> SharedExpert:
    shared_cls = SHARED_EXPERTS.get(cfg.shared_expert)
    return shared_cls(cfg, hidden_size)


@register_shared_expert("swiglu")
class SwiGLUSharedExpert(SharedExpert):
    """``w2(swiglu(w1 x, w3 x))``, optionally scaled per token by ``sigmoid(gate(x))``."""

    def __init__(self, cfg: MoEConfig, hidden_size: int) -> None:
        super().__init__(cfg, hidden_size)
        width = cfg.shared_expert_intermediate_size
        # nn.Linear children are initialized by the HF _init_weights pass.
        self.w1 = nn.Linear(hidden_size, width, bias=False)
        self.w2 = nn.Linear(width, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, width, bias=False)
        self.gate = (
            nn.Linear(hidden_size, 1, bias=False)
            if cfg.shared_expert_gate == "sigmoid"
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = swiglu(self.w1(x), self.w3(x))
        y = self.w2(h.to(x.dtype))  # in the input dtype
        if self.gate is not None:
            y = torch.sigmoid(self.gate(x).float()).to(y.dtype) * y
        return y
