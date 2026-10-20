# coding=utf-8
"""Routers: hidden states -> per-token expert choices and combine weights."""

from __future__ import annotations

import abc
from typing import Callable, Optional, Type

import torch
import torch.nn.functional as F
from torch import nn

from .balance import BalanceController
from .base import Registry, RoutingResult, TrainingState
from .config import MoEConfig

#: name -> f(logits [T, E] fp32) -> scores [T, E] fp32
SCORE_FUNCTIONS: Registry[Callable[[torch.Tensor], torch.Tensor]] = Registry(
    "MoE score function"
)


def register_score_function(name: str):
    return SCORE_FUNCTIONS.register(name)


@register_score_function("softmax")
def _softmax(logits: torch.Tensor) -> torch.Tensor:
    return logits.softmax(dim=-1)


class Router(TrainingState, nn.Module, abc.ABC):
    """Base router: owns the gate and a balance controller (``self.balance``)."""

    def __init__(
        self, cfg: MoEConfig, hidden_size: int, balance: BalanceController
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.hidden_size = hidden_size
        self.n_experts = cfg.n_routed_experts
        self.topk = cfg.num_experts_per_tok
        self.balance = balance  # may shift the scores used for selection only

    @abc.abstractmethod
    def forward(
        self, x: torch.Tensor, tokens_per_sample: Optional[int] = None
    ) -> RoutingResult:
        """Route a flat ``[T, hidden]`` batch (``tokens_per_sample`` tokens per sample)."""

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


@register_router("topk")
class TopKRouter(Router):
    """Top-k router: ``f(x W^T)`` scores in fp32, selection on balance-adjusted scores."""

    def __init__(
        self, cfg: MoEConfig, hidden_size: int, balance: BalanceController
    ) -> None:
        super().__init__(cfg, hidden_size, balance)
        self.weight = nn.Parameter(torch.empty(self.n_experts, hidden_size))
        self.score_fn = SCORE_FUNCTIONS.get(cfg.scoring_func)

    def reset_parameters(self, std: float) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=std)

    def forward(
        self, x: torch.Tensor, tokens_per_sample: Optional[int] = None
    ) -> RoutingResult:
        logits = F.linear(x.float(), self.weight.float())
        scores = self.score_fn(logits)
        selection = self.balance.adjust_selection_scores(scores)
        indices = selection.topk(self.topk, dim=-1).indices
        weights = scores.gather(1, indices)
        if self.cfg.norm_topk_prob:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.cfg.routed_scaling_factor
        flat = indices.flatten()
        # scatter_add instead of bincount: CUDA bincount hides a device sync.
        counts = torch.zeros(
            self.n_experts, dtype=torch.long, device=x.device
        ).scatter_add_(0, flat, torch.ones_like(flat))
        return RoutingResult(
            weights=weights, indices=indices, counts=counts, scores=scores
        )
