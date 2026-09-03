# coding=utf-8
"""Aux-loss-free balancing (DeepSeek-V3/V4 ``noaux_tc``).

A per-expert fp32 bias shifts the scores used for *selection*; combine weights
still use the raw scores. A sign controller moves the bias against the
all-reduced expert load, so it is updated by the trainer loop, not gradients.

Optionally (``dflash_config.moe_aux_loss_coeff`` > 0) the controller also
emits DeepSeek-V3's complementary sequence-wise balance loss,
``coeff * sum_e f_e * P_e`` with ``f_e`` the (E/(k*T))-scaled routed-token
fraction and ``P_e`` the mean normalized affinity, computed over the
micro-batch. The bias alone only *reorders* selection; from-scratch drafters
whose early router inputs are near-identical (mask-token embeddings) collapse
onto a few experts while the gate logits grow faster than the bias can
follow. The differentiable term bounds the logit gaps so input-driven routing
can emerge.

Checkpoint naming: the bias is stored as ``<layer>.gate.bias`` (the DeepSeek
native key SGLang maps onto ``e_score_correction_bias``); the module keeps it
at ``gate.balance.bias``, converted at the state-dict boundary.
"""

from __future__ import annotations

import re
from typing import Dict, Optional

import torch

from .balance import BalanceController, MetricValue, register_balance_controller
from .config import MoEConfig
from .router import RoutingResult
from .state_dict import register_state_dict_converter


@register_balance_controller("noaux_tc")
class NoAuxTCController(BalanceController):
    def __init__(self, cfg: MoEConfig, n_experts: int) -> None:
        super().__init__(cfg, n_experts)
        self.update_rate = float(cfg.bias_update_rate)
        self.aux_loss_coeff = float(cfg.aux_loss_coeff)
        self.register_buffer("bias", torch.zeros(n_experts, dtype=torch.float32))
        self._pending_counts: Optional[torch.Tensor] = None
        self._aux_loss: Optional[torch.Tensor] = None
        self.last_load: Optional[torch.Tensor] = None

    def _apply(self, fn, recurse=True):
        module = super()._apply(fn, recurse)
        # Sign-controller steps (~1e-3) vanish under bf16 rounding once the
        # bias grows; keep the buffer fp32 through module-wide dtype casts.
        if module.bias.dtype != torch.float32:
            module.bias.data = module.bias.data.float()
        return module

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
        out: Dict[str, MetricValue] = {"bias_abs_max": self.bias.abs().max()}
        if self._aux_loss is not None:
            out["aux_loss"] = self._aux_loss.detach()
        if self.last_load is not None:
            mean = self.last_load.mean().clamp_min(1e-9)
            out["global_load_max_ratio"] = self.last_load.max() / mean
            out["global_load_min_ratio"] = self.last_load.min() / mean
        return out


_NATIVE_BIAS = re.compile(r"^(?P<base>(?:.*\.)?)gate\.balance\.bias$")
_OFFICIAL_BIAS = re.compile(r"^(?P<base>(?:.*\.)?)gate\.bias$")


def _to_checkpoint(state: dict) -> dict:
    return {
        (f"{m['base']}gate.bias" if (m := _NATIVE_BIAS.match(k)) else k): v
        for k, v in state.items()
    }


def _is_moe_layer(state: dict, base: str) -> bool:
    return f"{base}experts.w1" in state or f"{base}experts.0.w1.weight" in state


def _from_checkpoint(state: dict) -> dict:
    out = {}
    for key, value in state.items():
        m = _OFFICIAL_BIAS.match(key)
        # Only an MoE layer's gate: a dense module named ``gate`` keeps its bias.
        if m is not None and _is_moe_layer(state, m["base"]):
            key = f"{m['base']}gate.balance.bias"
        out[key] = value
    return out


register_state_dict_converter(
    "noaux_tc_bias", to_checkpoint=_to_checkpoint, from_checkpoint=_from_checkpoint
)
