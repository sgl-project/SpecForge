# coding=utf-8
"""Load balancing as a swappable policy; registered: ``"none"``, ``"aux_loss"``."""

from __future__ import annotations

from typing import Dict, Optional, Type

import torch
from torch import nn

from .base import MetricValue, Registry, RoutingResult, TrainingState
from .config import MoEConfig


class BalanceController(TrainingState, nn.Module):
    """No balancing (``"none"``) and the base of real policies; see DESIGN.md."""

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


def load_balancing_loss(
    counts: torch.Tensor, scores: torch.Tensor, n_tokens: int
) -> torch.Tensor:
    """``E * sum_e (counts_e / T) * mean_t scores[t, e]`` (transformers' aux loss)."""
    # Equals load_balancing_loss_func(logits, E, top_k) for one layer; uniform gives k.
    n_experts = scores.shape[-1]
    f = counts.to(torch.float32) / float(n_tokens)
    p = scores.float().mean(dim=0)
    return n_experts * (f * p).sum()


@register_balance_controller("aux_loss")
class AuxLossController(BalanceController):
    """No selection bias; emits ``aux_loss_coeff * load_balancing_loss`` in training."""

    def __init__(self, cfg: MoEConfig, n_experts: int) -> None:
        super().__init__(cfg, n_experts)
        self.aux_loss_coeff = float(cfg.aux_loss_coeff)
        self._aux_loss: Optional[torch.Tensor] = None
        self.last_load_frac: Optional[torch.Tensor] = None

    def observe(self, routing: RoutingResult) -> None:
        self._aux_loss = None  # overwrite, never accumulate
        tokens = routing.indices.shape[0]
        self.last_load_frac = routing.counts.detach().float() / max(
            1, tokens * routing.topk
        )
        scores = routing.scores
        if self.aux_loss_coeff <= 0 or scores is None or not scores.requires_grad:
            return
        self._aux_loss = self.aux_loss_coeff * load_balancing_loss(
            routing.counts, scores, tokens
        )

    def aux_loss(self) -> Optional[torch.Tensor]:
        return self._aux_loss

    def metrics(self) -> Dict[str, MetricValue]:
        out: Dict[str, MetricValue] = {}
        if self._aux_loss is not None:
            out["aux_loss"] = self._aux_loss.detach()
        if self.last_load_frac is not None:
            # Fraction of (token, slot) assignments per expert: uniform is 1/E.
            out["load_frac_max"] = self.last_load_frac.max()
            out["load_frac_mean"] = self.last_load_frac.mean()
        return out
