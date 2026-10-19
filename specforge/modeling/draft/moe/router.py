# coding=utf-8
"""Routers: hidden states -> per-token expert choices + combine weights.

A router owns the gate projection and composes a :class:`~.balance.BalanceController`
(``self.balance``) that may shift scores for *selection only*. Routers register
by name (``MoEConfig.router``); score functions register separately
(``MoEConfig.scoring_func``) so one top-k router serves the softmax, sigmoid
and sqrtsoftplus families.

Every router receives ``tokens_per_sample``: the flat ``[T, hidden]`` batch is
``T / tokens_per_sample`` samples of that many tokens each, in order, or
``None`` when the layer input had no batch dimension.
"""

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


@register_score_function("sigmoid")
def _sigmoid(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits)


@register_score_function("sqrtsoftplus")
def _sqrtsoftplus(logits: torch.Tensor) -> torch.Tensor:
    """DeepSeek-V4 scoring: ``sqrt(softplus(logits))``.

    ``softplus`` underflows to exactly 0 once a logit drops below about -104
    in fp32, where ``sqrt``'s derivative is infinite: the backward would be
    NaN even where the incoming gradient is 0 (every unselected expert), while
    the forward looks fine. Clamping at the smallest normal value gives those
    entries a zero gradient and moves their score by less than 1e-19.
    """
    return F.softplus(logits).clamp_min(torch.finfo(logits.dtype).tiny).sqrt()


def group_limited_mask(
    selection: torch.Tensor, n_group: int, topk_group: int
) -> torch.Tensor:
    """Keep only the ``topk_group`` groups with the highest top-2 score sums
    (DeepSeek ``noaux_tc`` group scoring); other groups become ``-inf``."""
    tokens, n_experts = selection.shape
    grouped = selection.view(tokens, n_group, n_experts // n_group)
    group_scores = grouped.topk(min(2, grouped.shape[-1]), dim=-1).values.sum(-1)
    keep = group_scores.topk(topk_group, dim=-1).indices
    mask = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(1, keep, True)
    return grouped.masked_fill(~mask.unsqueeze(-1), float("-inf")).view(
        tokens, n_experts
    )


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
        self.balance = balance

    @abc.abstractmethod
    def forward(
        self, x: torch.Tensor, tokens_per_sample: Optional[int] = None
    ) -> RoutingResult:
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


@register_router("topk")
class TopKRouter(Router):
    """``scores = f(x W^T)``; pick top-k on balance-adjusted scores; combine
    with the raw scores (optionally renormalized, then scaled). Routing math
    runs in fp32 regardless of the model dtype.
    """

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
        if self.cfg.group_limited:
            selection = group_limited_mask(
                selection, self.cfg.n_group, self.cfg.topk_group
            )
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
