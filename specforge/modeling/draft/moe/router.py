# coding=utf-8
"""Routers: hidden states -> per-token expert choices and combine weights."""

from __future__ import annotations

import abc
from typing import Any, Callable, Dict, Optional, Type

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from .balance import BalanceController
from .base import Registry, RoutingResult, TrainingState, apply_keeping_fp32
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
    router_cls = ROUTERS.get(cfg.router)
    return router_cls(cfg, hidden_size, balance)


@register_router("topk")
class TopKRouter(Router):
    """Top-k router: ``f(x W^T + b)`` scores in fp32, selection on balance-adjusted scores."""

    #: Routing state that stays fp32 and exact through module-wide dtype casts.
    FP32_BUFFERS = ("bias", "input_mean")

    def __init__(
        self, cfg: MoEConfig, hidden_size: int, balance: BalanceController
    ) -> None:
        super().__init__(cfg, hidden_size, balance)
        self.weight = nn.Parameter(torch.empty(self.n_experts, hidden_size))
        # Buffers, not parameters: they stay fp32 under FSDP and no optimizer owns them.
        self.register_buffer(
            "bias",
            (
                torch.zeros(self.n_experts, dtype=torch.float32)
                if cfg.router_bias
                else None
            ),
        )
        centered = cfg.router_center == "sample"
        self.register_buffer(
            "input_mean",
            torch.zeros(hidden_size, dtype=torch.float32) if centered else None,
        )
        self.register_buffer(
            "input_mean_steps",
            torch.zeros((), dtype=torch.long) if centered else None,
        )
        self.center_momentum = float(cfg.router_center_momentum)
        self.score_fn = SCORE_FUNCTIONS.get(cfg.scoring_func)
        self._pending_mean: Optional[torch.Tensor] = None

    def _apply(self, fn, recurse=True):
        return apply_keeping_fp32(self, super()._apply, fn, recurse, self.FP32_BUFFERS)

    def reset_parameters(self, std: float) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _center(
        self, x32: torch.Tensor, tokens_per_sample: Optional[int]
    ) -> torch.Tensor:
        self._pending_mean = None  # overwrite, never accumulate
        if self.input_mean is None:
            return x32
        if not self.training:
            return x32 - self.input_mean
        # Stash only; the running mean moves in apply_pending_update.
        self._pending_mean = x32.mean(dim=0).detach()
        n = tokens_per_sample or x32.shape[0]
        if x32.shape[0] % n:
            raise ValueError(
                f"{x32.shape[0]} router tokens are not a multiple of "
                f"tokens_per_sample={n}"
            )
        grouped = x32.view(-1, n, x32.shape[-1])
        return (grouped - grouped.mean(dim=1, keepdim=True)).view_as(x32)

    def forward(
        self, x: torch.Tensor, tokens_per_sample: Optional[int] = None
    ) -> RoutingResult:
        x32 = self._center(x.float(), tokens_per_sample)
        logits = F.linear(x32, self.weight.float(), self.bias)
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

    def apply_pending_update(self) -> None:
        """Move the running centering mean, averaged across ranks."""
        mean = self._pending_mean
        self._pending_mean = None
        if mean is None:
            return
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(mean)
            mean = mean / dist.get_world_size()
        with torch.no_grad():
            if int(self.input_mean_steps) == 0:
                self.input_mean.copy_(mean)
            else:
                self.input_mean.mul_(self.center_momentum).add_(
                    mean, alpha=1.0 - self.center_momentum
                )
            self.input_mean_steps += 1


def fold_router_centering(layer: Dict[str, Any]) -> Dict[str, Any]:
    """Export fold: ``W (x - mu) = W x - W mu``, so ``gate.bias -= W @ mu`` (fp32)."""
    if "gate.input_mean" not in layer:
        return layer
    out = dict(layer)
    mean = out.pop("gate.input_mean").float()
    out.pop("gate.input_mean_steps", None)
    # fp32 on purpose: bf16 rounding of this common-mode logit could flip top-k.
    bias = -(out["gate.weight"].float() @ mean)
    if "gate.bias" in out:
        bias = bias + out["gate.bias"].float()
    out["gate.bias"] = bias
    return out


def fold_router_centering_config(cfg: MoEConfig) -> Dict[str, Any]:
    """Draft-JSON changes of a centering export: no centering, a router bias instead."""
    if cfg.router_center == "none":
        return {}
    return {"moe_router_bias": True, "dflash_config": {"moe_router_center": "none"}}
