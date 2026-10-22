# coding=utf-8
"""Load balancing as a swappable policy.

A controller sees routing outcomes and may (a) shift the scores used for
expert *selection* (never the combine weights), (b) emit an auxiliary loss,
and (c) report load metrics. It is an ``nn.Module`` so an implementation can
own buffers that travel with the checkpoint (e.g. a selection bias). The
timing rules are those of :class:`~.base.TrainingState`.

Registered policies:

- ``"none"``: no balancing.
- ``"noaux_tc"``: DeepSeek-V3/V4 aux-loss-free balancing. A per-expert fp32
  bias shifts selection scores; a sign controller moves it against the
  all-reduced expert load. Optionally (``aux_loss_coeff`` > 0) it also emits
  DeepSeek-V3's complementary sequence-wise balance loss.
- ``"aux_loss"``: the Switch-Transformer auxiliary load-balancing loss that
  Qwen MoE models train with (transformers' ``load_balancing_loss_func``).
"""

from __future__ import annotations

from typing import Dict, Optional, Type

import torch
from torch import nn

from .base import (
    MetricValue,
    Registry,
    RoutingResult,
    TrainingState,
    apply_keeping_fp32,
)
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


@register_balance_controller("noaux_tc")
class NoAuxTCController(BalanceController):
    """Aux-loss-free balancing with a sign-controlled selection bias.

    The bias is updated by the trainer loop (:meth:`apply_pending_update`),
    not by gradients, and lives as an fp32 buffer (``bias``) that survives
    module-wide dtype casts: the controller's ~1e-3 steps vanish under bf16
    rounding once the bias grows.
    """

    def __init__(self, cfg: MoEConfig, n_experts: int) -> None:
        super().__init__(cfg, n_experts)
        self.update_rate = float(cfg.bias_update_rate)
        self.aux_loss_coeff = float(cfg.aux_loss_coeff)
        self.register_buffer("bias", torch.zeros(n_experts, dtype=torch.float32))
        self._pending_counts: Optional[torch.Tensor] = None
        self._aux_loss: Optional[torch.Tensor] = None
        self.last_load: Optional[torch.Tensor] = None

    def _apply(self, fn, recurse=True):
        return apply_keeping_fp32(self, super()._apply, fn, recurse, ("bias",))

    def adjust_selection_scores(self, scores: torch.Tensor) -> torch.Tensor:
        return scores + self.bias

    def observe(self, routing: RoutingResult) -> None:
        # Overwrite, never accumulate: a checkpoint recompute re-runs the
        # forward and must leave identical state behind.
        self._pending_counts = routing.counts
        self._aux_loss = None
        scores = routing.scores
        if self.aux_loss_coeff <= 0 or scores is None or not scores.requires_grad:
            return
        tokens, n_experts = scores.shape
        # f_e: routed fraction scaled so that a uniform load gives 1.
        f = routing.counts.float() * (n_experts / (routing.topk * tokens))
        # P_e: mean normalized affinity (the differentiable side).
        p = (scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-20)).mean(0)
        self._aux_loss = self.aux_loss_coeff * (f * p).sum()

    def aux_loss(self) -> Optional[torch.Tensor]:
        return self._aux_loss

    def apply_pending_update(self) -> None:
        import torch.distributed as dist

        counts = self._pending_counts
        self._pending_counts = None
        if counts is None or self.update_rate <= 0:
            return
        load = counts.float()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(load)
        self.last_load = load
        error = load.mean() - load
        with torch.no_grad():
            self.bias += self.update_rate * torch.sign(error)

    def metrics(self) -> Dict[str, MetricValue]:
        out: Dict[str, MetricValue] = {
            "bias_abs_max": self.bias.abs().max(),
            # A warm-started bias carries a large common offset; only the
            # spread changes which experts are selected.
            "bias_spread": self.bias.max() - self.bias.min(),
        }
        if self._aux_loss is not None:
            out["aux_loss"] = self._aux_loss.detach()
        if self.last_load is not None:
            mean = self.last_load.mean().clamp_min(1e-9)
            out["global_load_max_ratio"] = self.last_load.max() / mean
            out["global_load_min_ratio"] = self.last_load.min() / mean
        return out


def load_balancing_loss(
    counts: torch.Tensor, scores: torch.Tensor, n_tokens: int
) -> torch.Tensor:
    """``E * sum_e (counts_e / T) * mean_t scores[t, e]`` in fp32.

    Matches ``transformers...load_balancing_loss_func(logits, E, top_k)`` for
    one layer when ``scores = softmax(logits)`` and ``counts`` are its top-k
    assignment counts; a perfectly uniform routing gives ``k``.
    """
    n_experts = scores.shape[-1]
    f = counts.to(torch.float32) / float(n_tokens)
    p = scores.float().mean(dim=0)
    return n_experts * (f * p).sum()


@register_balance_controller("aux_loss")
class AuxLossController(BalanceController):
    """No selection bias; routing follows the raw scores. In training the
    controller emits ``aux_loss_coeff * load_balancing_loss`` over the tokens
    of the current micro-batch."""

    def __init__(self, cfg: MoEConfig, n_experts: int) -> None:
        super().__init__(cfg, n_experts)
        self.aux_loss_coeff = float(cfg.aux_loss_coeff)
        self._aux_loss: Optional[torch.Tensor] = None
        self.last_load_frac: Optional[torch.Tensor] = None

    def observe(self, routing: RoutingResult) -> None:
        # Overwrite, never accumulate (activation-checkpoint recompute safety).
        self._aux_loss = None
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
