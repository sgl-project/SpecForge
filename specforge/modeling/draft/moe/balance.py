# coding=utf-8
"""MoE load balancing: the loss math, and the hooks the DFlash-family model calls."""

from typing import Dict, List, Optional, Protocol, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


def count_tokens_per_expert(
    indices: torch.Tensor, mask: Optional[torch.Tensor], num_experts: int
) -> torch.Tensor:
    """Kept slots per expert, fp32 [E]; ``mask`` [T] marks the kept tokens."""
    slots = indices.reshape(-1)
    weights = torch.ones_like(slots, dtype=torch.float32)
    if mask is not None:
        weights = mask.to(torch.float32).repeat_interleave(indices.shape[-1])
    # scatter_add_, not bincount: bincount reads the indices' range on the host.
    return weights.new_zeros(num_experts).scatter_add_(0, slots, weights)


def aux_loss_tokens_per_expert(
    aux_loss_type: str, counts: torch.Tensor, step_counts: torch.Tensor
) -> torch.Tensor:
    """Counts the aux loss balances: this micro-batch's, or the step's pooled over ranks."""
    if aux_loss_type != "batch_wise":
        return counts
    # Batch-wise: the step's counts so far, pooled over ranks; the loss uses shares.
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
    share = tokens_per_expert / tokens_per_expert.sum().clamp(min=1)
    return probs.shape[-1] * (share * mean_prob).sum()


def dispatch_balance(tokens_per_expert: torch.Tensor) -> torch.Tensor:
    """A dispatch's balance against itself: 1 when uniform, E / K when collapsed."""
    # load_balance_loss with the load standing in for the probabilities.
    share = tokens_per_expert / tokens_per_expert.sum().clamp(min=1)
    return share.shape[-1] * share.square().sum()


# Hooks the DFlash-family model calls; routers and layers are found by their state.


class _RouterLike(Protocol):
    tokens_per_expert: torch.Tensor
    selection_bias: Optional[torch.Tensor]
    aux_loss: Optional[torch.Tensor]
    aux_loss_coef: float

    def update_expert_bias(self, tokens_per_expert: torch.Tensor) -> None: ...


class _MoELayerLike(Protocol):
    gate: _RouterLike
    token_mask: Optional[torch.Tensor]


def set_moe_token_mask(model: nn.Module, mask: Optional[torch.Tensor]) -> None:
    """Hand every MoE layer this batch's ``[B, S]`` kept-slot mask."""
    for layer in _moe_layers(model):
        layer.token_mask = mask


def take_moe_aux_loss(
    model: nn.Module,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Pop this forward's load-balance loss: (weighted sum over layers, raw mean)."""
    # Weighted like its objective: by denominator under loss_terms, equally otherwise.
    layers = _moe_layers(model)
    terms = [
        (layer.gate.aux_loss_coef, layer.gate.aux_loss)
        for layer in layers
        if layer.gate.aux_loss is not None
    ]
    for layer in layers:
        layer.gate.aux_loss = None
        layer.token_mask = None
    if not terms:
        return None
    # One coefficient per layer, as in torchtitan and Megatron; the metric stays a mean.
    aux = torch.stack([coef * loss for coef, loss in terms]).sum()
    raw = torch.stack([loss for _, loss in terms]).mean()
    return aux, raw


def step_moe_balancing(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Optimizer step boundary: pool counts once, move the biases, score the step."""
    routers = _routers(model)
    if not routers:
        return {}
    # One collective for every layer; all ranks ran the same forwards, so it lines up.
    counts = torch.stack([router.tokens_per_expert for router in routers])
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(counts)
    for router, pooled in zip(routers, counts):
        if router.selection_bias is not None:
            router.update_expert_bias(pooled)
        router.tokens_per_expert.zero_()  # the next step's count window
    # The step's pooled load: the collapse signal, unlike the per-micro-batch loss.
    balance = torch.stack([dispatch_balance(c) for c in counts]).mean()
    return {"moe/step_balance": balance}


def _routers(model: nn.Module) -> List[_RouterLike]:
    return [m for m in model.modules() if hasattr(m, "tokens_per_expert")]


def _moe_layers(model: nn.Module) -> List[_MoELayerLike]:
    return [m for m in model.modules() if hasattr(m, "token_mask")]
