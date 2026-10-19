# coding=utf-8
"""Load balancing as a swappable policy.

A controller sees routing outcomes and may (a) shift the scores used for
expert *selection* (never the combine weights), (b) emit an auxiliary loss,
and (c) report load metrics. It is an ``nn.Module`` so an implementation can
own buffers that travel with the checkpoint (e.g. a selection bias). The
timing rules are those of :class:`~.base.TrainingState`.

Registered policies:

- ``"none"``: no balancing.
"""

from __future__ import annotations

from typing import Type

import torch
from torch import nn

from .base import Registry, TrainingState
from .config import MoEConfig


class BalanceController(TrainingState, nn.Module):
    """No balancing (registered as ``"none"``); the base for real policies."""

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
    return BALANCE_CONTROLLERS.get(cfg.balance)(cfg, n_experts)
