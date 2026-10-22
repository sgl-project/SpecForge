# coding=utf-8
"""Load balancing: the aux loss the router adds, and selection-time policies."""

from __future__ import annotations

from typing import Callable, Optional, Type

import torch
from torch import nn

from .base import Registry, RoutingResult, TrainingState
from .config import MoEConfig


class BalanceController(TrainingState, nn.Module):
    """Shifts the scores that pick experts; ``"none"`` keeps them (see DESIGN.md)."""

    def __init__(self, cfg: MoEConfig, n_experts: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_experts = n_experts

    def adjust_selection_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """Scores used to pick experts; combine weights still use the raw ones."""
        return scores


BALANCE_CONTROLLERS: Registry[Type[BalanceController]] = Registry(
    "MoE balance controller"
)
BALANCE_CONTROLLERS.register("none", BalanceController)


def register_balance_controller(name: str):
    return BALANCE_CONTROLLERS.register(name)


def build_balance_controller(cfg: MoEConfig, n_experts: int) -> BalanceController:
    controller_cls = BALANCE_CONTROLLERS.get(cfg.balance)
    return controller_cls(cfg, n_experts)


#: ``MoEConfig.aux_loss_type`` -> f(routing, tokens_per_sample) -> unscaled loss
AUX_LOSSES: Registry[Callable[..., torch.Tensor]] = Registry("MoE aux loss")


def register_aux_loss(name: str):
    return AUX_LOSSES.register(name)


@register_aux_loss("switch")
def load_balancing_loss(
    routing: RoutingResult, tokens_per_sample: Optional[int] = None
) -> torch.Tensor:
    """``E * sum_e (counts_e / T) * mean_t scores[t, e]`` (transformers' aux loss)."""
    # Equals load_balancing_loss_func(logits, E, top_k) for one layer; uniform gives k.
    n_tokens, n_experts = routing.scores.shape
    f = routing.counts.to(torch.float32) / float(n_tokens)
    p = routing.scores.float().mean(dim=0)
    return n_experts * (f * p).sum()
