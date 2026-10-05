# coding=utf-8
"""``MoELayer``: the FFN that composes router, experts and shared expert.

Attribute names follow the official DeepSeek-style checkpoint layout
(``gate``, ``experts``, ``shared_experts``) so that per-implementation
converters only need to handle their own internals.

Under expert parallelism (:meth:`MoELayer.apply_expert_parallel`) the layer
all-gathers its tokens over the EP group, routes and computes the locally owned
experts on all of them, and reduce-scatters the partial outputs back. The
shared expert and everything outside the layer stay data-parallel.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from .balance import MetricValue, build_balance_controller
from .config import MoEConfig, resolve_moe_config
from .expert_parallel import ExpertParallelLayout, gather_tokens, scatter_outputs
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
        #: Expert-parallel layout once :meth:`apply_expert_parallel` ran.
        self.ep: Optional[ExpertParallelLayout] = None
        # Detached per-expert counts of the last training forward, for metrics.
        self.last_counts: Optional[torch.Tensor] = None

    @property
    def balance(self):
        return self.gate.balance

    def apply_expert_parallel(self, ep_mesh: DeviceMesh) -> ExpertParallelLayout:
        """Slice the routed experts over ``ep_mesh`` (a 1-D mesh of the EP group).

        The router, balance controller and shared expert stay replicated; the
        experts backend decides how its weights are sliced.
        """
        apply = getattr(self.experts, "apply_expert_parallel", None)
        if apply is None:
            raise NotImplementedError(
                f"{type(self.experts).__name__} does not support expert parallelism"
            )
        self.ep = apply(ep_mesh)
        return self.ep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, self.hidden_size)
        ep = self.ep
        # Under EP every rank routes and computes its experts for the whole
        # group's tokens; the replicated router sees identical inputs on every
        # rank of the group, so no routing has to be exchanged.
        routed_in = gather_tokens(x, ep) if ep is not None else x
        routing: RoutingResult = self.gate(routed_in)
        if self.training:
            self.last_counts = routing.counts.detach()
            self.balance.observe(routing)
        y = self.experts(routed_in, routing)
        if ep is not None:
            # fp32 partial sums from every expert owner -> this rank's tokens.
            y = scatter_outputs(y, ep).to(x.dtype)
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
