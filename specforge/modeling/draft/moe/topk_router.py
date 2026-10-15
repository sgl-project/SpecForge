# coding=utf-8
"""Top-k router with pluggable score functions and optional group-limited routing."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .balance import BalanceController
from .config import MoEConfig
from .router import (
    Router,
    RoutingResult,
    get_score_function,
    register_router,
    register_score_function,
)


@register_score_function("softmax")
def _softmax(logits: torch.Tensor) -> torch.Tensor:
    return logits.softmax(dim=-1)


@register_score_function("sigmoid")
def _sigmoid(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits)


@register_score_function("sqrtsoftplus")
def _sqrtsoftplus(logits: torch.Tensor) -> torch.Tensor:
    """DeepSeek-V4 scoring: ``sqrt(softplus(logits))``."""
    return F.softplus(logits).sqrt()


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


@register_router("topk")
class TopKRouter(Router):
    """``scores = f(x W^T)``; pick top-k on balance-adjusted scores; combine
    with the raw scores (optionally renormalized, then scaled)."""

    def __init__(
        self, cfg: MoEConfig, hidden_size: int, balance: BalanceController
    ) -> None:
        super().__init__(cfg, hidden_size, balance)
        self.weight = nn.Parameter(torch.empty(self.n_experts, hidden_size))
        self.score_fn = get_score_function(cfg.scoring_func)

    def reset_parameters(self, std: float) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> RoutingResult:
        # Routing math in fp32 regardless of the model dtype.
        scores = self.score_fn(F.linear(x.float(), self.weight.float()))
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
