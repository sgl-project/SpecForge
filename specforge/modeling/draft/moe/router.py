# coding=utf-8
"""Top-k routers for draft MoE layers."""

from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

ScoreFn = Callable[[torch.Tensor], torch.Tensor]


def softmax_scores(logits: torch.Tensor) -> torch.Tensor:
    """``softmax(logits)`` over experts: the default router scores."""
    return logits.softmax(dim=-1)


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
    ) -> None:
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError(
                f"top_k={top_k} must be between 1 and num_experts={num_experts}"
            )
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.score_fn = score_fn
        # Slot for DeepSeek-V4's selection bias: shifts top-k picks, not weights.
        self.register_buffer(
            "selection_bias", torch.zeros(num_experts) if selection_bias else None
        )
        self.route_scale = route_scale
        self.reset_parameters(initializer_range)

    def reset_parameters(self, std: float) -> None:
        # HF's default _init_weights skips bare Parameters.
        nn.init.normal_(self.weight, std=std)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``x`` [T, H] -> fp32 scores [T, E], indices [T, K], fp32 weights [T, K]."""
        # Logits in x's dtype as in Qwen; DeepSeek-V4 projects in fp32.
        scores = self.score_fn(F.linear(x, self.weight).float())
        choice = scores if self.selection_bias is None else scores + self.selection_bias
        indices = choice.topk(self.top_k, dim=-1).indices
        weights = scores.gather(-1, indices)
        if self.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return scores, indices, weights * self.route_scale
