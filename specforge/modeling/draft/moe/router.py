# coding=utf-8
"""Top-k routers for draft MoE layers: selection, the selection bias, the counts."""

from typing import Callable, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from specforge.modeling.draft.moe.balance import (
    count_tokens_per_expert,
    load_balance_loss,
)

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
        aux_loss_coef: float = 0.0,
        aux_loss_type: str = "batch_wise",
        bias_update_rate: float = 0.001,
    ) -> None:
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError(
                f"top_k={top_k} must be between 1 and num_experts={num_experts}"
            )
        if aux_loss_type not in ("microbatch_wise", "batch_wise"):
            raise ValueError(f"unknown aux_loss_type {aux_loss_type!r}")
        self.num_experts = num_experts
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.score_fn = score_fn
        # DeepSeek's selection bias: shifts the top-k picks, not the weights; fp32.
        self.register_buffer(
            "selection_bias",
            torch.zeros(num_experts, dtype=torch.float32) if selection_bias else None,
        )
        # The step's kept-slot counts: bias update, batch-wise loss, step balance.
        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(num_experts, dtype=torch.float32),
            persistent=False,
        )
        self.route_scale = route_scale
        self.aux_loss_coef = aux_loss_coef
        self.aux_loss_type = aux_loss_type
        self.bias_update_rate = bias_update_rate
        # This micro-batch's load-balance (aux) loss, kept in training for the trainer.
        self.aux_loss: Optional[torch.Tensor] = None
        self.reset_parameters(initializer_range)

    def _apply(self, fn, recurse=True):
        # Keep both buffers fp32, values included, through casts like .to(bfloat16).
        names = ("selection_bias", "tokens_per_expert")
        kept = {name: getattr(self, name) for name in names}
        super()._apply(fn, recurse)
        for name, old in kept.items():
            if old is not None:
                new = getattr(self, name)
                setattr(self, name, new.float() if old.is_meta else old.to(new.device))
        return self

    def reset_parameters(self, std: float) -> None:
        # HF's default _init_weights skips bare Parameters and buffers.
        nn.init.normal_(self.weight, std=std)
        for buffer in (self.selection_bias, self.tokens_per_expert):
            if buffer is not None:
                nn.init.zeros_(buffer)

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
            self.tokens_per_expert += counts
            load = self._aux_loss_load(counts)
            # Always kept in training; at coefficient 0 it is a graph-free metric.
            graph = self.aux_loss_coef != 0 and torch.is_grad_enabled()
            with torch.set_grad_enabled(graph):
                self.aux_loss = load_balance_loss(scores, load, mask)
        return scores, indices, weights * self.route_scale

    def _aux_loss_load(self, counts: torch.Tensor) -> torch.Tensor:
        """The load the aux loss balances: this micro-batch's counts, or the step's."""
        if self.aux_loss_type != "batch_wise":
            return counts
        # Batch-wise: the step's counts so far, pooled over ranks; the loss uses shares.
        load = self.tokens_per_expert.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(load)
        return load

    @torch.no_grad()
    def update_expert_bias(self, tokens_per_expert: torch.Tensor) -> None:
        """One DeepSeek-V3 aux-loss-free step from the pooled kept-slot counts."""
        gap = tokens_per_expert.mean() - tokens_per_expert
        delta = self.bias_update_rate * torch.sign(gap)
        # Zero-mean delta: the same top-k picks, and the bias never drifts.
        self.selection_bias += delta - delta.mean()
