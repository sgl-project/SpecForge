# coding=utf-8
"""Top-k routers for draft MoE layers, with the load-balance loss they own."""

from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn

ScoreFn = Callable[[torch.Tensor], torch.Tensor]


def softmax_scores(logits: torch.Tensor) -> torch.Tensor:
    """``softmax(logits)`` over experts: the default router scores."""
    return logits.softmax(dim=-1)


def count_tokens_per_expert(
    indices: torch.Tensor, mask: Optional[torch.Tensor], num_experts: int
) -> torch.Tensor:
    """Kept slots per expert, fp32 [E]; ``mask`` [T] marks the kept tokens."""
    weights = None
    if mask is not None:
        weights = mask.to(torch.float32).repeat_interleave(indices.shape[-1])
    counts = torch.bincount(indices.reshape(-1), weights=weights, minlength=num_experts)
    return counts.float()


def load_balance_loss(
    scores: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``E * sum_e load_e * mean_prob_e`` over the kept tokens; ``K`` when balanced."""
    # The MoE "aux loss": Switch-style load balancing, stored per router as aux_loss.
    # Per-token L1 normalization: a no-op for softmax, needed for sigmoid scores.
    probs = F.normalize(scores, p=1, dim=-1)
    kept = probs.new_ones(probs.shape[0]) if mask is None else mask.to(probs.dtype)
    tokens = kept.sum().clamp(min=1)
    mean_prob = (probs * kept.unsqueeze(-1)).sum(dim=0) / tokens
    return probs.shape[-1] * (tokens_per_expert / tokens * mean_prob).sum()


class TopKRouter(nn.Module):
    """Top-k router; ``weight`` [E, H] matches Qwen3.5-MoE's ``gate``."""

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        top_k: int,
        norm_topk_prob: bool,
        initializer_range: float,
        score_fn: ScoreFn = softmax_scores,
        selection_bias: bool = False,
        route_scale: float = 1.0,
        aux_loss_coef: float = 0.0,
    ) -> None:
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError(
                f"top_k={top_k} must be between 1 and num_experts={num_experts}"
            )
        self.num_experts = num_experts
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.score_fn = score_fn
        # Slot for DeepSeek-V4's selection bias: shifts top-k picks, not weights.
        self.register_buffer(
            "selection_bias", torch.zeros(num_experts) if selection_bias else None
        )
        self.route_scale = route_scale
        self.aux_loss_coef = aux_loss_coef
        # This micro-batch's load-balance (aux) loss, kept in training for the trainer.
        self.aux_loss: Optional[torch.Tensor] = None
        self.reset_parameters(initializer_range)

    def reset_parameters(self, std: float) -> None:
        # HF's default _init_weights skips bare Parameters.
        nn.init.normal_(self.weight, std=std)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``x`` [T, H] -> fp32 scores [T, E], indices [T, K], fp32 weights [T, K]."""
        self.aux_loss = None  # per-forward state: never carry a stale loss into eval
        # Logits in x's dtype as in Qwen; DeepSeek-V4 projects in fp32.
        scores = self.score_fn(F.linear(x, self.weight).float())
        choice = scores if self.selection_bias is None else scores + self.selection_bias
        indices = choice.topk(self.top_k, dim=-1).indices
        weights = scores.gather(-1, indices)
        if self.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        if self.training:
            counts = count_tokens_per_expert(indices, mask, self.num_experts)
            # Always kept in training: with coefficient 0 it is the balance metric.
            self.aux_loss = load_balance_loss(scores, counts, mask)
        return scores, indices, weights * self.route_scale
