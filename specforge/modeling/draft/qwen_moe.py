"""Qwen-style sparse MoE FFN for DFlash-family drafters.

The routing recipe is Qwen3-Next's, which Qwen3.8-Flash-Next (``qwen4_exp``)
keeps: softmax over all experts, top-k, the top-k weights renormalized, and one
shared expert scaled by a sigmoid gate. Expert width and naming follow the
target's MTP layer, so a drafter can be warm-started from it expert for expert.

Load balancing is the Switch Transformer auxiliary loss the target was trained
with (``E * sum_e f_e * P_e``, scaled by ``router_aux_loss_coef``), taken per
layer and averaged over layers -- Transformers pools the counts over layers
first, which only matters once layers disagree. An auxiliary loss is chosen
over a selection bias because a bias would have to be reproduced at serving
time; the loss leaves nothing behind in the checkpoint.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN


def moe_enabled(config) -> bool:
    return int(getattr(config, "num_experts", 0) or 0) > 0


class QwenExpert(nn.Module):
    """One SwiGLU expert, named like the Qwen MoE checkpoints' per-expert keys."""

    def __init__(self, hidden_size: int, intermediate_size: int, act: str) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class QwenSparseMoE(nn.Module):
    """Softmax top-k routed experts plus a sigmoid-gated shared expert."""

    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = int(config.num_experts)
        self.top_k = int(config.num_experts_per_tok)
        if not 0 < self.top_k <= self.num_experts:
            raise ValueError(
                f"num_experts_per_tok={self.top_k} must be in "
                f"[1, num_experts={self.num_experts}]"
            )
        self.norm_topk_prob = bool(getattr(config, "norm_topk_prob", True))
        self.aux_loss_coef = float(getattr(config, "router_aux_loss_coef", 0.0))
        act = getattr(config, "hidden_act", "silu")

        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                QwenExpert(self.hidden_size, config.moe_intermediate_size, act)
                for _ in range(self.num_experts)
            ]
        )
        self.shared_expert = QwenExpert(
            self.hidden_size, config.shared_expert_intermediate_size, act
        )
        self.shared_expert_gate = nn.Linear(self.hidden_size, 1, bias=False)
        self._aux_losses: List[torch.Tensor] = []
        self._last_load: Optional[torch.Tensor] = None

    def route(self, x: torch.Tensor):
        """Return ``(weights, indices, probabilities)`` for ``x`` of shape [T, H]."""

        logits = self.gate(x)
        probabilities = F.softmax(logits, dim=-1, dtype=torch.float32)
        weights, indices = probabilities.topk(self.top_k, dim=-1)
        if self.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights.to(x.dtype), indices, probabilities

    def _record_balance(
        self,
        indices: torch.Tensor,
        probabilities: torch.Tensor,
        balance_mask: Optional[torch.Tensor],
    ):
        tokens = probabilities.shape[0]
        if balance_mask is None:
            weight = probabilities.new_ones(tokens)
        else:
            weight = balance_mask.reshape(-1).to(
                device=probabilities.device, dtype=probabilities.dtype
            )
            if weight.numel() != tokens:
                raise ValueError(
                    f"MoE balance mask covers {weight.numel()} tokens but the "
                    f"forward routed {tokens}"
                )
        # Weights, not boolean indexing: indexing would add a host sync per
        # layer to learn the kept count.
        counts = probabilities.new_zeros(self.num_experts).scatter_add_(
            0, indices.reshape(-1), weight.repeat_interleave(self.top_k)
        )
        self._last_load = counts.detach()
        if not self.training or self.aux_loss_coef <= 0:
            return
        # Switch Transformer's balance term for this layer: the fraction of
        # top-k assignments each expert receives times its mean router
        # probability, summed and scaled by the expert count. Perfect balance
        # gives top_k. Both means run over the counted tokens only.
        counted = weight.sum().clamp_min(1.0)
        fraction = counts / counted
        mean_probability = (probabilities * weight[:, None]).sum(dim=0) / counted
        self._aux_losses.append(self.num_experts * (fraction * mean_probability).sum())

    def pop_aux_losses(self) -> List[torch.Tensor]:
        losses, self._aux_losses = self._aux_losses, []
        return losses

    def load_balance(self) -> Optional[tuple[float, float]]:
        """``(max/mean expert load, fraction of experts unused)`` of the last forward."""

        if self._last_load is None:
            return None
        load = self._last_load.float()
        return (
            float(load.max() / load.mean().clamp_min(1e-9)),
            float((load == 0).float().mean()),
        )

    def _group_by_expert(self, indices: torch.Tensor):
        """Sort assignments by expert with one host readback, not one per expert."""

        flat = indices.reshape(-1)
        # Ascend has no AiCore ArgSort for integer keys; expert ids are exact
        # in float32, so the float view sorts to the same permutation.
        order = torch.argsort(flat.float(), stable=True)
        counts = torch.bincount(flat, minlength=self.num_experts)
        return order, [0] + torch.cumsum(counts, dim=0).tolist()

    def forward(
        self, x: torch.Tensor, balance_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Route ``x``; ``balance_mask`` marks the tokens that count toward balance.

        The training wrapper passes it so padded draft blocks, which the main
        loss already excludes, neither pull on the router nor skew the load
        telemetry; ``None`` counts every token. It is an argument rather than
        module state so a checkpointed recompute sees the same mask.
        """

        shape = x.shape
        flat = x.reshape(-1, self.hidden_size)
        weights, indices, probabilities = self.route(flat)
        self._record_balance(indices, probabilities, balance_mask)

        # Keep the router and every expert in the graph even when an expert
        # receives no token: DDP needs every parameter to join each reduction.
        # The zero terms change no value.
        routed = flat.float() * 0.0 + weights.float().sum() * 0.0
        order, bounds = self._group_by_expert(indices)
        for expert_id, expert in enumerate(self.experts):
            start, end = bounds[expert_id], bounds[expert_id + 1]
            if start == end:
                routed = routed + sum(
                    parameter.reshape(-1)[0].float() * 0.0
                    for parameter in expert.parameters()
                )
                continue
            slots = order[start:end]
            token = torch.div(slots, self.top_k, rounding_mode="floor")
            slot = slots % self.top_k
            contribution = expert(flat[token]) * weights[token, slot, None]
            routed = routed.index_add(0, token, contribution.float())

        shared = self.shared_expert(flat) * torch.sigmoid(self.shared_expert_gate(flat))
        return (routed.to(flat.dtype) + shared).view(shape)


__all__ = ["QwenExpert", "QwenSparseMoE", "moe_enabled"]
