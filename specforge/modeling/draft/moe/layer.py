# coding=utf-8
"""``MoELayer``: the FFN that composes router, experts and shared expert.

Attribute names follow the official DeepSeek-style checkpoint layout
(``gate``, ``experts``, ``shared_experts``) so that per-implementation
converters only need to handle their own internals.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import torch
from torch import nn

from .balance import MetricValue, build_balance_controller
from .config import MoEConfig, resolve_moe_config
from .experts import build_routed_experts
from .router import RoutingResult, build_router
from .shared import build_shared_expert


class MoELayer(nn.Module):
    """Routed FFN: ``y = experts(x, gate(x)) + shared_experts(x)``."""

    def __init__(self, cfg: MoEConfig, hidden_size: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.hidden_size = hidden_size
        balance = build_balance_controller(cfg, cfg.n_routed_experts)
        self.gate = build_router(cfg, hidden_size, balance)
        self.experts = build_routed_experts(cfg, hidden_size)
        if cfg.freeze_experts:
            self.experts.requires_grad_(False)
        self.shared_experts: Optional[nn.Module] = (
            build_shared_expert(cfg, hidden_size) if cfg.n_shared_experts else None
        )
        # Detached per-expert counts of the last training forward, for metrics.
        self.last_counts: Optional[torch.Tensor] = None

    @property
    def balance(self):
        return self.gate.balance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, self.hidden_size)
        routing: RoutingResult = self.gate(x)
        if self.training:
            self.last_counts = routing.counts.detach()
            self.balance.observe(routing)
        y = self.experts(x, routing)
        if self.shared_experts is not None:
            y = y + self.shared_experts(x)
        return y.view(shape)

    # -- model-level hooks (see hooks.py) ---------------------------------
    def apply_pending_balance_update(self) -> None:
        self.balance.apply_pending_update()

    def aux_loss(self) -> Optional[torch.Tensor]:
        return self.balance.aux_loss()

    def metrics(self) -> Dict[str, MetricValue]:
        out: Dict[str, MetricValue] = {}
        counts = self.last_counts
        if counts is not None and counts.numel():
            load = counts.float()
            mean = load.mean().clamp_min(1e-9)
            out["load_max_ratio"] = load.max() / mean
            out["load_min_ratio"] = load.min() / mean
            out["experts_unused_frac"] = (load == 0).float().mean()
        out.update(self.balance.metrics())
        return out

    def reset_parameters(self, std: float) -> None:
        """Initialize bare Parameters the HF ``_init_weights`` pass cannot see."""
        self.gate.reset_parameters(std)
        self.experts.reset_parameters(std)
        if self.shared_experts is not None:
            self.shared_experts.reset_parameters(std)


def build_ffn(config, dense: Callable[[object], nn.Module]) -> nn.Module:
    """The dense/MoE switch for a decoder layer's FFN.

    ``config`` is the draft's HF config; ``dense`` builds the dense MLP (the
    kernel provider's factory) and is used verbatim when the config is dense,
    so dense drafts are byte-for-byte unaffected by this package.
    """
    moe_cfg = resolve_moe_config(config)
    if moe_cfg is None:
        return dense(config)
    return MoELayer(moe_cfg, int(config.hidden_size))
