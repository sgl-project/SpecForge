# coding=utf-8
"""MoE training helpers: the Switch aux loss and router centering."""

from dataclasses import replace
from typing import TYPE_CHECKING, List, Optional

import torch
import torch.distributed as dist
from torch import nn

if TYPE_CHECKING:
    from specforge.modeling.draft.moe.layer import MoELayer
    from specforge.training.strategies.base import StepOutput


def switch_aux_loss(
    scores: torch.Tensor,
    indices: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``E * sum_e load_e * mean_score_e`` over unmasked tokens; ``K`` when balanced."""
    # Expects softmax probs per token; other score functions need their own loss.
    if mask is not None:
        scores, indices = scores[mask], indices[mask]
    if indices.numel() == 0:
        return scores.sum()  # no kept slots: a zero that stays on the graph
    num_experts = scores.shape[-1]
    load = torch.bincount(indices.reshape(-1), minlength=num_experts) / len(indices)
    return num_experts * (load * scores.mean(dim=0)).sum()


class RouterCentering(nn.Module):
    """Router input minus its per-sample mean in training, minus a running mean in eval."""

    def __init__(self, hidden_size: int, momentum: float) -> None:
        super().__init__()
        self.momentum = momentum
        self.register_buffer(
            "running_mean", torch.zeros(hidden_size, dtype=torch.float32)
        )

    def _apply(self, fn, recurse=True):
        # Keep the running mean fp32, values included, through casts like .to(bfloat16).
        kept = self.running_mean
        super()._apply(fn, recurse)
        device = self.running_mean.device
        self.running_mean = (
            self.running_mean.float() if kept.is_meta else kept.to(device)
        )
        return self

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """``x`` [B, S, H]; ``mask`` [B, S] marks the kept slots (DFlash: kept blocks)."""
        if not self.training:
            return x - self.running_mean.to(x.dtype)
        weight = x.new_ones(x.shape[:-1]) if mask is None else mask.to(x.dtype)
        total = (x * weight.unsqueeze(-1)).sum(dim=1, dtype=torch.float32)
        count = weight.sum(dim=1, dtype=torch.float32).unsqueeze(-1)
        with torch.no_grad():
            stats = torch.cat([total.sum(dim=0), count.sum().reshape(1)])
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(stats)  # one running mean on every rank
            batch_mean = stats[:-1] / stats[-1].clamp(min=1)
            self.running_mean.lerp_(batch_mean, 1 - self.momentum)
        return x - (total / count.clamp(min=1)).unsqueeze(1).to(x.dtype)


def set_moe_token_mask(model: nn.Module, mask: Optional[torch.Tensor]) -> None:
    """Hand every MoE layer this batch's ``[B, S]`` kept-slot mask."""
    for layer in _moe_layers(model):
        layer.token_mask = mask


def include_moe_aux_loss(model: nn.Module, out: "StepOutput") -> "StepOutput":
    """Add the MoE layers' aux loss to a training step; ``out`` as is without one."""
    layers = _moe_layers(model)
    terms = [
        (layer.aux_loss_coef, layer.aux_loss)
        for layer in layers
        if layer.aux_loss is not None
    ]
    for layer in layers:
        layer.aux_loss = layer.token_mask = None
    if not terms:
        return out
    raw = torch.stack([loss for _, loss in terms]).mean()
    aux = torch.stack([coef * loss for coef, loss in terms]).mean()
    metric = (raw.detach(), torch.ones_like(raw))
    ratio_metrics = {**out.ratio_metrics, "moe/aux_loss": metric}
    if out.loss_terms is None:
        return replace(out, loss=out.loss + aux, ratio_metrics=ratio_metrics)
    numerator, denominator = out.loss_terms
    # The trainer divides the numerator by the global denominator.
    return replace(
        out,
        loss=out.loss + aux,
        loss_terms=(numerator + aux * denominator.detach(), denominator),
        ratio_metrics=ratio_metrics,
    )


def _moe_layers(model: nn.Module) -> List["MoELayer"]:
    # layer.py imports this module, so import MoELayer lazily.
    from specforge.modeling.draft.moe.layer import MoELayer

    return [module for module in model.modules() if isinstance(module, MoELayer)]
