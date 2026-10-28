# coding=utf-8
"""MoELayer and build_ffn, the FFN switch between a dense MLP and an MoE layer."""

from typing import Callable, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from specforge.modeling.draft.moe.experts import RoutedExperts, SharedExpert
from specforge.modeling.draft.moe.router import TopKRouter


def count_tokens_per_expert(
    indices: torch.Tensor, mask: Optional[torch.Tensor], num_experts: int
) -> torch.Tensor:
    """Kept slots per expert, int64 [E]; ``mask`` [T] marks the kept tokens."""
    slots = indices.reshape(-1)
    kept = torch.ones_like(slots)
    if mask is not None:
        kept = mask.to(slots.dtype).repeat_interleave(indices.shape[-1])
    # scatter_add_, not bincount: bincount reads the indices' range on the host.
    return slots.new_zeros(num_experts).scatter_add_(0, slots, kept)


def aux_loss_tokens_per_expert(
    aux_loss_type: str, counts: torch.Tensor, step_counts: torch.Tensor
) -> torch.Tensor:
    """The load the aux loss balances: this micro-batch's, or the step's pooled over ranks."""
    if aux_loss_type != "batch_wise":
        return counts
    load = step_counts.clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(load)
    return load


def load_balance_loss(
    scores: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``E * sum_e share_e * P_e`` over the kept tokens; ``1`` when balanced."""
    # The MoE "aux loss" in torchtitan's, Megatron's and DeepSeek-V3's units:
    # share_e is expert e's part of the kept slots (K per token), so f = E * share.
    # Per-token L1 normalization: a no-op for softmax, needed for sigmoid scores.
    probs = F.normalize(scores, p=1, dim=-1)
    kept = probs.new_ones(probs.shape[0]) if mask is None else mask.to(probs.dtype)
    mean_prob = (probs * kept.unsqueeze(-1)).sum(dim=0) / kept.sum().clamp(min=1)
    share = tokens_per_expert.float() / tokens_per_expert.sum().clamp(min=1)
    return probs.shape[-1] * (share * mean_prob).sum()


class MoELayer(nn.Module):
    """Routed experts plus an optional shared expert with an optional sigmoid gate."""

    def __init__(
        self,
        gate: TopKRouter,
        experts: RoutedExperts,
        shared_expert: Optional[SharedExpert] = None,
        shared_expert_gate: Optional[nn.Linear] = None,
        aux_loss_coef: float = 0.0,
        aux_loss_type: str = "batch_wise",
    ) -> None:
        super().__init__()
        if aux_loss_type not in ("microbatch_wise", "batch_wise"):
            raise ValueError(f"unknown aux_loss_type {aux_loss_type!r}")
        self.gate = gate
        self.experts = experts
        self.shared_expert = shared_expert
        self.shared_expert_gate = shared_expert_gate
        # The step's kept-slot counts per expert; the balancing hook pools and resets them.
        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(gate.num_experts, dtype=torch.int64),
            persistent=False,
        )
        # One coefficient per layer, as in torchtitan and Megatron; 0 skips the loss.
        self.aux_loss_coef = aux_loss_coef
        self.aux_loss_type = aux_loss_type
        # This forward's load-balance loss, for the balancing hook to fold before backward.
        self.aux_loss: Optional[torch.Tensor] = None

    def reset_parameters(self, std: float) -> None:
        # Only the router and experts hold bare Parameters that HF's init skips.
        self.gate.reset_parameters(std)
        self.experts.reset_parameters(std)

    def forward(
        self, hidden_states: torch.Tensor, moe_token_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """``moe_token_mask`` [B, S] marks the tokens the balancing counts; None counts all."""
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        scores, indices, weights = self.gate(x)
        self.aux_loss = None
        if self.training:
            mask = None if moe_token_mask is None else moe_token_mask.reshape(-1)
            counts = count_tokens_per_expert(indices, mask, self.gate.num_experts)
            self.tokens_per_expert += counts
            if self.aux_loss_coef != 0:
                load = aux_loss_tokens_per_expert(
                    self.aux_loss_type, counts, self.tokens_per_expert
                )
                self.aux_loss = load_balance_loss(scores, load, mask)
        out = self.experts(x, indices, weights)
        if self.shared_expert is not None:
            shared = self.shared_expert(x)
            if self.shared_expert_gate is not None:
                shared = torch.sigmoid(self.shared_expert_gate(x)) * shared
            out = out + shared
        # The routed sum is fp32; cast back once, after adding the shared expert.
        return out.to(hidden_states.dtype).view_as(hidden_states)


def build_ffn(
    config: PretrainedConfig,
    *,
    dense: Callable[[PretrainedConfig], nn.Module],
) -> nn.Module:
    """``dense(config)``, or the MoE layer of the family the ``moe_preset`` names."""

    moe_preset = getattr(config, "moe_preset", None)
    if moe_preset is None:
        return dense(config)
    # Families import MoELayer from this module, so load them lazily.
    from specforge.modeling.draft.moe.families import FAMILIES

    if moe_preset not in FAMILIES:
        raise ValueError(
            f"unknown moe_preset={moe_preset!r}; expected one of {sorted(FAMILIES)}"
        )
    return FAMILIES[moe_preset](config)
