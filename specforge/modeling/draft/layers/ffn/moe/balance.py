# coding=utf-8
"""MoE load balancing around the step: the aux-loss fold, pooled counts, the bias update."""

import dataclasses
from typing import Any, Dict, List

import torch
import torch.distributed as dist
from torch import nn

from specforge.modeling.draft.layers.ffn.moe.layer import MoELayer


def dispatch_balance(tokens_per_expert: torch.Tensor) -> torch.Tensor:
    """A dispatch's balance against itself: 1 when uniform, E / K when collapsed."""
    share = tokens_per_expert.float() / tokens_per_expert.sum().clamp(min=1)
    return share.shape[-1] * share.square().sum()


def moe_layers(model: nn.Module) -> List[MoELayer]:
    """The model's MoE layers, in module order."""
    return [m for m in model.modules() if isinstance(m, MoELayer)]


class MoEBalancing:
    """The trainer's hook over a model's MoE layers; registered once."""

    def __init__(self, model: nn.Module) -> None:
        self.layers = moe_layers(model)

    def before_backward(self, out: Any) -> Any:
        """Fold this micro-batch's load-balance losses into the objective; log their mean."""
        terms = [
            (layer.aux_loss_coef, layer.aux_loss)
            for layer in self.layers
            if layer.aux_loss is not None
        ]
        for layer in self.layers:
            layer.aux_loss = None  # consumed: never folded twice
        if not terms:
            return out
        # One coefficient per layer, as in torchtitan and Megatron; the metric stays a mean.
        weighted = torch.stack([coef * loss for coef, loss in terms]).sum()
        raw = torch.stack([loss for _, loss in terms]).mean().detach()
        ratio_metrics = {
            **out.ratio_metrics,
            "moe/aux_loss": (raw, torch.ones_like(raw)),
        }
        loss_terms = out.loss_terms
        if loss_terms is not None:
            # Into the numerator the trainer backpropagates and normalizes globally.
            numerator, denominator = loss_terms
            loss_terms = (numerator + weighted * denominator, denominator)
        return dataclasses.replace(
            out,
            loss=out.loss + weighted,
            loss_terms=loss_terms,
            ratio_metrics=ratio_metrics,
        )

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
