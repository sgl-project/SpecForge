# coding=utf-8
"""Model-level plumbing for MoE layers.

A draft model with MoE FFNs needs three things from its trainer loop, all
expressed here as functions over any ``nn.Module`` tree so DFlash, DFlash2 and
DSpark share them:

- :func:`apply_pending_balance_updates` at the top of the model forward (in
  training), outside activation-checkpoint regions;
- :func:`collect_moe_aux_loss` to add to the objective when a balance policy
  emits one;
- :func:`collect_moe_metrics` for per-step diagnostics (``moe/...``).
"""

from __future__ import annotations

from typing import Dict, Iterator, Optional

import torch
from torch import nn

from .balance import MetricValue
from .layer import MoELayer


def iter_moe_layers(module: nn.Module) -> Iterator[MoELayer]:
    for sub in module.modules():
        if isinstance(sub, MoELayer):
            yield sub


def apply_pending_balance_updates(module: nn.Module) -> None:
    for layer in iter_moe_layers(module):
        layer.apply_pending_balance_update()


def collect_moe_aux_loss(module: nn.Module) -> Optional[torch.Tensor]:
    """Sum of the layers' (already scaled) auxiliary losses, or ``None``."""
    total: Optional[torch.Tensor] = None
    for layer in iter_moe_layers(module):
        loss = layer.aux_loss()
        if loss is None:
            continue
        total = loss if total is None else total + loss
    return total


def collect_moe_metrics(
    module: nn.Module, prefix: str = "moe/"
) -> Dict[str, MetricValue]:
    """Layer-averaged scalar diagnostics; ``{}`` for dense models."""
    sums: Dict[str, MetricValue] = {}
    n = 0
    for layer in iter_moe_layers(module):
        n += 1
        for key, value in layer.metrics().items():
            sums[key] = value if key not in sums else sums[key] + value
    if n == 0:
        return {}
    return {
        f"{prefix}{key}": (value / n if isinstance(value, torch.Tensor) else value / n)
        for key, value in sums.items()
    }
