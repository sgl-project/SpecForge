# coding=utf-8
"""Top-k routers for draft MoE layers, and the contract their load balancing relies on."""

from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

ScoreFn = Callable[[torch.Tensor], torch.Tensor]


def softmax_scores(logits: torch.Tensor) -> torch.Tensor:
    """``softmax(logits)`` over experts: the default router scores."""
    return logits.softmax(dim=-1)


class Router(nn.Module):
    """What the balancing relies on: the forward signature, the expert bias, its update."""

    def __init__(
        self, num_experts: int, *, expert_bias: bool, bias_update_rate: float
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
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

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``x`` [T, H] -> scores [T, E], indices [T, K], weights [T, K]."""
        raise NotImplementedError

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
    """Top-k router; ``weight`` [E, H] matches Qwen3.5-MoE's ``gate``."""

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
            expert_bias=expert_bias,
            bias_update_rate=bias_update_rate,
        )
        if not 1 <= top_k <= num_experts:
            raise ValueError(
                f"top_k={top_k} must be between 1 and num_experts={num_experts}"
            )
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.score_fn = score_fn
        self.route_scale = route_scale
        self.reset_parameters(initializer_range)

    def reset_parameters(self, std: float) -> None:
        super().reset_parameters(std)
        nn.init.normal_(self.weight, std=std)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``x`` [T, H] -> fp32 scores [T, E], indices [T, K], fp32 weights [T, K]."""
        # Logits in x's dtype as in Qwen; DeepSeek-V4 projects in fp32.
        scores = self.score_fn(F.linear(x, self.weight).float())
        choice = scores if self.expert_bias is None else scores + self.expert_bias
        indices = choice.topk(self.top_k, dim=-1).indices
        weights = scores.gather(-1, indices)
        if self.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return scores, indices, weights * self.route_scale
