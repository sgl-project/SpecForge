# coding=utf-8
"""MoE load balancing at the optimizer step: pooled counts, the bias update, the metric."""

from typing import Dict, List

import torch
import torch.distributed as dist
from torch import nn

from specforge.modeling.draft.moe.layer import MoELayer


def dispatch_balance(tokens_per_expert: torch.Tensor) -> torch.Tensor:
    """A dispatch's balance against itself: 1 when uniform, E / K when collapsed."""
    share = tokens_per_expert.float() / tokens_per_expert.sum().clamp(min=1)
    return share.shape[-1] * share.square().sum()


def moe_layers(model: nn.Module) -> List[MoELayer]:
    """The model's MoE layers, in module order."""
    return [m for m in model.modules() if isinstance(m, MoELayer)]


class MoEBalancing:
    """The trainer's per-step hook over a model's MoE layers; registered once."""

    def __init__(self, model: nn.Module) -> None:
        self.layers = moe_layers(model)

    def on_step(self) -> Dict[str, torch.Tensor]:
        """Optimizer step boundary: pool the counts once, move the biases, score the step."""
        if not self.layers:
            return {}
        # One collective for every layer; all ranks ran the same forwards, so it lines up.
        counts = torch.stack([layer.tokens_per_expert for layer in self.layers])
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts)
        pooled = counts.float()
        for layer, load in zip(self.layers, pooled):
            layer.gate.update_expert_bias(load)
            layer.tokens_per_expert.zero_()  # the next step's count window
        # The step's pooled load: the collapse signal.
        balance = torch.stack([dispatch_balance(load) for load in pooled]).mean()
        return {"moe/step_balance": balance}
