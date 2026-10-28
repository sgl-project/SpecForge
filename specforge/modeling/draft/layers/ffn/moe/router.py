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
    """Routing as score -> select -> routing_weights; families override the steps."""

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        *,
        norm_topk_prob: bool = False,
        route_scale: float = 1.0,
        expert_bias: bool = False,
        bias_update_rate: float = 0.001,
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
        # DeepSeek's expert bias: shifts the top-k picks, not the weights; fp32.
        self.register_buffer(
            "expert_bias",
            torch.zeros(num_experts, dtype=torch.float32) if expert_bias else None,
        )
        # 0 keeps a loaded bias fixed: a routing input the step never moves.
        self.bias_update_rate = bias_update_rate

    def _apply(self, fn, recurse=True):
        # Keep the bias fp32, values included, through casts like .to(bfloat16).
        old = self.expert_bias
        super()._apply(fn, recurse)
        if old is not None:
            new = self.expert_bias
            self.expert_bias = new.float() if old.is_meta else old.to(new.device)
        return self

    def reset_parameters(self, std: float) -> None:
        # HF's default _init_weights skips buffers; subclasses add their Parameters.
        if self.expert_bias is not None:
            nn.init.zeros_(self.expert_bias)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Identity here; families map ``x`` [T, H] to fp32 affinities [T, E]."""
        return x

    def select(self, scores: torch.Tensor) -> torch.Tensor:
        """``scores`` [T, E] -> the chosen experts [T, K]; the bias shifts the picks only."""
        choice = scores if self.expert_bias is None else scores + self.expert_bias
        return choice.topk(self.top_k, dim=-1).indices

    def routing_weights(
        self, scores: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """The weights [T, K] that combine the chosen experts' outputs."""
        weights = scores.gather(-1, indices)
        if self.norm_topk_prob:
            # DeepSeek's 1e-20: a sigmoid top-k sum can be 0; a no-op for softmax.
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return weights * self.route_scale

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = self.score(x)
        indices = self.select(scores)
        weights = self.routing_weights(scores, indices)
        return scores, indices, weights

    @torch.no_grad()
    def update_expert_bias(self, tokens_per_expert: torch.Tensor) -> None:
        """One DeepSeek-V3 aux-loss-free step from the step's pooled kept-slot counts."""
        if self.expert_bias is None or self.bias_update_rate == 0:
            return
        gap = tokens_per_expert.mean() - tokens_per_expert
        delta = self.bias_update_rate * torch.sign(gap)
        # Zero-mean delta: the same top-k picks, and the bias never drifts.
        self.expert_bias += delta - delta.mean()


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
        expert_bias: bool = False,
        bias_update_rate: float = 0.001,
    ) -> None:
        super().__init__(
            num_experts,
            top_k,
            norm_topk_prob=norm_topk_prob,
            route_scale=route_scale,
            expert_bias=expert_bias,
            bias_update_rate=bias_update_rate,
        )
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        self.score_fn = score_fn
        self.reset_parameters(initializer_range)

    def reset_parameters(self, std: float) -> None:
        super().reset_parameters(std)
        # HF's default _init_weights skips bare Parameters.
        nn.init.normal_(self.weight, std=std)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        # Logits in x's dtype as in Qwen; DeepSeek-V4 projects in fp32.
        return self.score_fn(F.linear(x, self.weight).float())
