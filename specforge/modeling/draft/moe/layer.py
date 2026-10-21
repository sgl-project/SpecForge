# coding=utf-8
"""MoELayer, and the FFN switch between a draft layer's dense MLP and MoE."""

from typing import Callable, Optional

import torch
from torch import nn
from transformers import PretrainedConfig

from specforge.modeling.draft.moe.experts import RoutedExperts, SharedExpert
from specforge.modeling.draft.moe.router import TopKRouter
from specforge.modeling.draft.moe.util import RouterCentering, switch_aux_loss


class MoELayer(nn.Module):
    """Routed experts plus an optional shared expert with an optional sigmoid gate."""

    def __init__(
        self,
        gate: TopKRouter,
        experts: RoutedExperts,
        shared_expert: Optional[SharedExpert] = None,
        shared_expert_gate: Optional[nn.Linear] = None,
        centering: Optional[RouterCentering] = None,
        aux_loss_coef: float = 0.0,
    ) -> None:
        super().__init__()
        self.gate = gate
        self.experts = experts
        self.shared_expert = shared_expert
        self.shared_expert_gate = shared_expert_gate
        self.centering = centering
        self.aux_loss_coef = aux_loss_coef
        # Per-batch state for set_moe_token_mask and include_moe_aux_loss (util.py).
        self.token_mask: Optional[torch.Tensor] = None
        self.aux_loss: Optional[torch.Tensor] = None

    def reset_parameters(self, std: float) -> None:
        # HF's init skips bare Parameters and buffers; reset them here.
        self.gate.reset_parameters(std)
        self.experts.reset_parameters(std)
        if self.centering is not None:
            nn.init.zeros_(self.centering.running_mean)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_size = hidden_states.shape[-1]
        mask = self.token_mask if self.training else None
        router_input = hidden_states
        if self.centering is not None:
            router_input = self.centering(hidden_states, mask)
        x = hidden_states.reshape(-1, hidden_size)
        scores, indices, weights = self.gate(router_input.reshape(-1, hidden_size))
        if self.training and self.aux_loss_coef > 0:
            flat_mask = None if mask is None else mask.reshape(-1)
            self.aux_loss = switch_aux_loss(scores, indices, flat_mask)
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
