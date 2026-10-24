# coding=utf-8
"""MoELayer, the FFN switch between dense MLP and MoE, and the model-facing glue."""

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn
from transformers import PretrainedConfig

from specforge.modeling.draft.moe.experts import RoutedExperts, SharedExpert
from specforge.modeling.draft.moe.router import TopKRouter, dispatch_balance


class MoELayer(nn.Module):
    """Routed experts plus an optional shared expert with an optional sigmoid gate."""

    def __init__(
        self,
        gate: TopKRouter,
        experts: RoutedExperts,
        shared_expert: Optional[SharedExpert] = None,
        shared_expert_gate: Optional[nn.Linear] = None,
    ) -> None:
        super().__init__()
        self.gate = gate
        self.experts = experts
        self.shared_expert = shared_expert
        self.shared_expert_gate = shared_expert_gate
        # This batch's [B, S] kept-slot mask, from set_moe_token_mask.
        self.token_mask: Optional[torch.Tensor] = None

    def reset_parameters(self, std: float) -> None:
        # Only the router and experts hold bare Parameters that HF's init skips.
        self.gate.reset_parameters(std)
        self.experts.reset_parameters(std)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        mask = self.token_mask if self.training else None
        _, indices, weights = self.gate(x, None if mask is None else mask.reshape(-1))
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


def _routers(model: nn.Module) -> List[TopKRouter]:
    return [module for module in model.modules() if isinstance(module, TopKRouter)]


def _moe_layers(model: nn.Module) -> List[MoELayer]:
    return [module for module in model.modules() if isinstance(module, MoELayer)]
