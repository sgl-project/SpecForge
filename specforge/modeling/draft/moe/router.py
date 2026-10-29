# coding=utf-8
"""The router contract and the top-k router for draft MoE layers."""

from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

ScoreFn = Callable[[torch.Tensor], torch.Tensor]


def softmax_scores(logits: torch.Tensor) -> torch.Tensor:
    """``softmax(logits)`` over experts: the default router scores."""
    return logits.softmax(dim=-1)


class Router(nn.Module):
    """Routing in three steps, score -> select -> combine; families override the steps."""

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        *,
        norm_topk_prob: bool = False,
        route_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError(
                f"top_k={top_k} must be between 1 and num_experts={num_experts}"
            )
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.route_scale = route_scale

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` [T, H] -> fp32 affinities [T, E]; load balancing reads these."""
        raise NotImplementedError

    def select(self, scores: torch.Tensor) -> torch.Tensor:
        """``scores`` [T, E] -> the chosen experts [T, K]."""
        return scores.topk(self.top_k, dim=-1).indices

    def combine(self, scores: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """The combine weights [T, K], renormalized and scaled as the family asks."""
        weights = scores.gather(-1, indices)
        if self.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights * self.route_scale

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = self.score(x)
        indices = self.select(scores)
        weights = self.combine(scores, indices)
        return scores, indices, weights


class TopKRouter(Router):
    """A linear gate scored by ``score_fn``; ``weight`` [E, H] matches Qwen3.5-MoE's ``gate``."""

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        top_k: int,
        norm_topk_prob: bool,
        initializer_range: float,
        score_fn: ScoreFn = softmax_scores,
        route_scale: float = 1.0,
    ) -> None:
        super().__init__(
            num_experts, top_k, norm_topk_prob=norm_topk_prob, route_scale=route_scale
        )
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        self.score_fn = score_fn
        self.reset_parameters(initializer_range)

    def reset_parameters(self, std: float) -> None:
        # HF's default _init_weights skips bare Parameters.
        nn.init.normal_(self.weight, std=std)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        # Logits in x's dtype as in Qwen; DeepSeek-V4 projects in fp32.
        return self.score_fn(F.linear(x, self.weight).float())
