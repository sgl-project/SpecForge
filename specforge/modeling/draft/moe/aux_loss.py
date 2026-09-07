# coding=utf-8
"""Auxiliary load-balancing loss (Switch Transformer / Qwen MoE ``aux_loss``).

No selection bias: routing follows the raw scores. In training the controller
emits the differentiable loss transformers' ``load_balancing_loss_func``
computes for Qwen3.5/3.8 MoE (``router_aux_loss_coef``), over the tokens of
the current micro-batch::

    loss = coeff * E * sum_e f_e * P_e
    f_e  = (# of (token, top-k slot) assignments to expert e) / T
    P_e  = mean over tokens of the router probability of expert e

``f_e`` is transformers' ``tokens_per_expert`` summed over the ``k`` slots, so
a perfectly uniform routing gives ``coeff * k`` (not ``coeff``: the
``noaux_tc`` controller's complementary loss divides by ``k`` as DeepSeek-V3
does). ``coeff`` is ``dflash_config.moe_aux_loss_coeff``; the trainer adds the
result to the objective through :func:`.hooks.collect_moe_aux_loss`.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from .balance import BalanceController, MetricValue, register_balance_controller
from .config import MoEConfig
from .router import RoutingResult


def load_balancing_loss(
    counts: torch.Tensor, scores: torch.Tensor, n_tokens: int
) -> torch.Tensor:
    """``E * sum_e (counts_e / T) * mean_t scores[t, e]`` in fp32.

    Matches ``transformers...load_balancing_loss_func(logits, E, top_k)`` when
    ``scores = softmax(logits)`` and ``counts`` are its top-k assignment counts.
    """
    n_experts = scores.shape[-1]
    f = counts.to(torch.float32) / float(n_tokens)
    p = scores.float().mean(dim=0)
    return n_experts * (f * p).sum()


@register_balance_controller("aux_loss")
class AuxLossController(BalanceController):
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
