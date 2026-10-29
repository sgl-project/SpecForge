# coding=utf-8
"""MoELayer and build_ffn, the FFN switch between a dense MLP and an MoE layer."""

from typing import Callable, Optional

import torch
from torch import nn
from transformers import PretrainedConfig

from specforge.modeling.draft.moe.experts import RoutedExperts, SharedExpert
from specforge.modeling.draft.moe.router import TopKRouter


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

    def reset_parameters(self, std: float) -> None:
        """Every part of the layer, bare Parameters and Linears alike."""
        self.gate.reset_parameters(std)
        self.experts.reset_parameters(std)
        if self.shared_expert is not None:
            self.shared_expert.reset_parameters(std)
        if self.shared_expert_gate is not None:
            nn.init.normal_(self.shared_expert_gate.weight, std=std)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        _, indices, weights = self.gate(x)
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
