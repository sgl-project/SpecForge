# coding=utf-8
"""Router contract: hidden states -> per-token expert choices + combine weights.

A router owns the gate projection and composes a :class:`BalanceController`
(``self.balance``) that may shift scores for *selection only*. Concrete routers
register by name (``MoEConfig.router``); score functions register separately
(``MoEConfig.scoring_func``) so one top-k router serves softmax, sigmoid and
sqrtsoftplus families.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Callable, Optional, Type

import torch
from torch import nn

from ._registry import Registry
from .balance import BalanceController
from .config import MoEConfig


@dataclass
class RoutingResult:
    """Routing decision for a flat batch of ``T`` tokens.

    ``weights`` are the final combine weights (normalized and scaled as the
    recipe dictates) in fp32; ``indices`` the chosen experts; ``counts`` the
    per-expert token counts on device (no host sync), which dispatch and
    balancing both consume. ``scores`` are the full pre-selection affinities
    (differentiable, fp32) for balance policies that need a gradient signal,
    e.g. an auxiliary balance loss; routers may leave it ``None``.
    """

    weights: torch.Tensor  # [T, k] fp32
    indices: torch.Tensor  # [T, k] long
    counts: torch.Tensor  # [E] long
    scores: Optional[torch.Tensor] = None  # [T, E] fp32, differentiable

    @property
    def topk(self) -> int:
        return int(self.indices.shape[-1])


#: name -> f(logits [T, E] fp32) -> scores [T, E] fp32
SCORE_FUNCTIONS: Registry[Callable[[torch.Tensor], torch.Tensor]] = Registry(
    "MoE score function"
)


def register_score_function(name: str):
    return SCORE_FUNCTIONS.register(name)


def get_score_function(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    return SCORE_FUNCTIONS.get(name)


class Router(nn.Module, abc.ABC):
    """Base router. Subclasses implement :meth:`forward` and :meth:`reset_parameters`."""

    def __init__(
        self, cfg: MoEConfig, hidden_size: int, balance: BalanceController
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.hidden_size = hidden_size
        self.n_experts = cfg.n_routed_experts
        self.topk = cfg.num_experts_per_tok
        self.balance = balance

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> RoutingResult:
        """Route a flat ``[T, hidden]`` batch."""

    @abc.abstractmethod
    def reset_parameters(self, std: float) -> None:
        """Initialize the gate weights (called from the model's ``_init_weights``)."""


ROUTERS: Registry[Type[Router]] = Registry("MoE router")


def register_router(name: str):
    return ROUTERS.register(name)


def build_router(
    cfg: MoEConfig, hidden_size: int, balance: BalanceController
) -> Router:
    return ROUTERS.get(cfg.router)(cfg, hidden_size, balance)
