"""Expert load balancing for MoE drafts: DeepSeek-V3's aux-loss-free bias update."""

import torch
import torch.distributed as dist
from torch import Tensor, nn

from specforge.modeling.draft.layers.mlp import MoEFFN


def count_tokens_per_expert(
    expert_ids: Tensor, keep: Tensor, num_experts: int
) -> Tensor:
    """[T, K] choices and a [T] keep mask -> int64 [E] counts over the kept tokens."""
    return torch.bincount(expert_ids[keep].reshape(-1), minlength=num_experts)


def dispatch_balance(counts: Tensor) -> Tensor:
    """A dispatch's balance against itself: 1 when uniform, E / K when collapsed."""
    share = counts.float() / counts.sum().clamp(min=1)
    return share.numel() * share.square().sum()


class MoEBalancer:
    """Counts the kept tokens each MoE layer routes over an optimizer step, then moves
    every router's expert bias by `rate`; the trainer calls observe and step."""

    def __init__(self, model: nn.Module, *, rate: float):
        self.rate = rate
        self.layers = [m for m in model.modules() if isinstance(m, MoEFFN)]
        if not self.layers:
            raise ValueError("bias_update_rate needs a draft with MoE layers")
        if any(layer.gate.expert_bias is None for layer in self.layers):
            raise ValueError(
                "bias_update_rate needs router nodes with expert_bias: true"
            )
        self.counts: list[Tensor | None] = [None] * len(self.layers)

    @classmethod
    def for_objective(cls, model: nn.Module, objective) -> "MoEBalancer | None":
        rate = getattr(objective, "bias_update_rate", 0.0)
        return cls(model, rate=rate) if rate else None

    def observe(self, output, *, keep: Tensor) -> None:
        """Count one training forward; forwards without gradients (eval, prepass) count nothing."""
        if not torch.is_grad_enabled():
            return
        choices = output.expert_choices or ()
        if len(choices) != len(self.layers):
            raise ValueError(
                f"expected {len(self.layers)} expert choices, got {len(choices)}"
            )
        for index, (layer, choice) in enumerate(zip(self.layers, choices)):
            counts = count_tokens_per_expert(
                choice.expert_ids,
                keep.to(choice.expert_ids.device),
                layer.config.num_experts,
            )
            previous = self.counts[index]
            self.counts[index] = counts if previous is None else previous + counts

    def step(self) -> dict[str, Tensor]:
        """At the optimizer step: pool the window's counts over ranks, move the biases, reset."""
        if any(counts is None for counts in self.counts):
            return {}
        pooled = torch.stack(self.counts)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(pooled)
        self.counts = [None] * len(self.layers)
        pooled = pooled.float()
        for layer, load in zip(self.layers, pooled):
            layer.gate.update_expert_bias(load, rate=self.rate)
        balance = torch.stack([dispatch_balance(load) for load in pooled]).mean()
        return {"moe/step_balance": balance}
