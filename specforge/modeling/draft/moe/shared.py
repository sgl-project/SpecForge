# coding=utf-8
"""Shared expert: an always-on FFN added to the routed output.

:class:`SharedExpert` is the contract; implementations register by name
(``MoEConfig.shared_expert``) and :class:`SwiGLUSharedExpert` (``"swiglu"``)
is the built-in one. The output is ``[T, hidden]`` in the input dtype;
:class:`~.layer.MoELayer` adds it into the fp32 routed sum.
"""

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
    return SHARED_EXPERTS.get(cfg.shared_expert)(cfg, hidden_size)


@register_shared_expert("swiglu")
class SwiGLUSharedExpert(SharedExpert):
    """SwiGLU FFN ``w2(swiglu(w1 x, w3 x))`` with the routed experts' clamp,
    ``shared_expert_intermediate_size`` wide. With ``shared_expert_gate ==
    "sigmoid"`` the output is scaled per token by ``sigmoid(gate(x))``,
    ``gate`` a bias-free ``Linear(hidden, 1)``. The output is in the input
    dtype; the ``nn.Linear`` children are initialized by the HF
    ``_init_weights`` pass.
    """

    def __init__(self, cfg: MoEConfig, hidden_size: int) -> None:
        super().__init__(cfg, hidden_size)
        width = cfg.shared_expert_intermediate_size
        self.swiglu_limit = float(cfg.swiglu_limit)
        self.w1 = nn.Linear(hidden_size, width, bias=False)
        self.w2 = nn.Linear(width, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, width, bias=False)
        self.gate = (
            nn.Linear(hidden_size, 1, bias=False)
            if cfg.shared_expert_gate == "sigmoid"
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = swiglu(self.w1(x), self.w3(x), self.swiglu_limit)
        y = self.w2(h.to(x.dtype))
        if self.gate is not None:
            y = torch.sigmoid(self.gate(x).float()).to(y.dtype) * y
        return y
