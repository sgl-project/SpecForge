# coding=utf-8
"""Load balancing as a swappable policy.

The controller sees routing outcomes and may (a) shift scores used for expert
*selection* (never the combine weights), (b) emit an auxiliary loss, and (c)
report load metrics. It is an ``nn.Module`` so implementations can own
buffers that travel with the checkpoint (e.g. a selection bias).

Two timing rules every implementation must respect:

- :meth:`observe` is called from the layer forward and must only *stash*
  (overwrite, never accumulate): an activation-checkpoint recompute re-runs the
  forward and must leave identical state behind.
- :meth:`apply_pending_update` is called by the *model* before the next
  forward, outside any checkpoint region, and may mutate selection state and
  run collectives. Mutating selection state inside the forward would make the
  recompute route differently and break checkpointing.
"""

from __future__ import annotations

from typing import Dict, Optional, Type, Union

import torch
from torch import nn

from ._registry import Registry
from .config import MoEConfig

MetricValue = Union[torch.Tensor, float]


class BalanceController(nn.Module):
    """No balancing (registered as ``"none"``); the base for real policies."""

    def __init__(self, cfg: MoEConfig, n_experts: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_experts = n_experts

    def adjust_selection_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """Scores used to pick experts; combine weights still use the raw ones."""
        return scores

    def observe(self, counts: torch.Tensor) -> None:
        """Stash this forward's per-expert token counts (training only)."""

    def apply_pending_update(self) -> None:
        """Consume the stash; called by the model outside checkpoint regions."""

    def aux_loss(self) -> Optional[torch.Tensor]:
        """Scaled auxiliary loss for the last forward, or ``None``."""
        return None

    def metrics(self) -> Dict[str, MetricValue]:
        """Scalar diagnostics (rank-local; the trainer DP-averages them)."""
        return {}


BALANCE_CONTROLLERS: Registry[Type[BalanceController]] = Registry(
    "MoE balance controller"
)
BALANCE_CONTROLLERS.register("none", BalanceController)


def register_balance_controller(name: str):
    return BALANCE_CONTROLLERS.register(name)


def build_balance_controller(cfg: MoEConfig, n_experts: int) -> BalanceController:
    return BALANCE_CONTROLLERS.get(cfg.balance)(cfg, n_experts)
