"""MoE load balancing: DeepSeek-V3's aux-loss-free expert-bias update from routed-token counts."""

import torch
import torch.distributed as dist
from torch import Tensor, nn

from specforge.modeling.draft.layers.mlp import MoEFFN

from .base import check_options

# The options that join a resume contract only while balancing is on.
BALANCING_OPTIONS = ("bias_update_rate",)


def balancing_settings(options: dict) -> dict:
    """The balancing options with defaults filled in and checked; off when the rate is 0."""
    settings = {"bias_update_rate": 0.0, **options}
    check_options(nonnegative={"bias_update_rate": settings["bias_update_rate"]})
    return settings


def count_tokens_per_expert(
    expert_ids: Tensor, keep: Tensor, num_experts: int
) -> Tensor:
    """[T, K] expert ids and a [T] keep mask -> int64 [E] counts over the kept tokens."""
    kept = keep.to(expert_ids.dtype)[:, None].expand_as(expert_ids).reshape(-1)
    # scatter_add_, not bincount: bincount reads the index range on the host.
    return expert_ids.new_zeros(num_experts).scatter_add_(
        0, expert_ids.reshape(-1), kept
    )


def dispatch_balance(counts: Tensor) -> Tensor:
    """A dispatch's balance against itself: 1 when uniform, E / K when collapsed."""
    share = counts.float() / counts.sum().clamp(min=1)
    return share.numel() * share.square().sum()


class MoEBalancer:
    """The objectives' StepStatistics for MoE drafts: per-step counts and the bias update."""

    def __init__(self, model: nn.Module, *, rate: float):
        self.rate = rate
        self.layers = [m for m in model.modules() if isinstance(m, MoEFFN)]
        if not self.layers:
            raise ValueError("MoE load balancing needs a draft with MoE layers")
        if rate and any(layer.gate.expert_bias is None for layer in self.layers):
            raise ValueError(
                "bias_update_rate needs router nodes with expert_bias: true"
            )
        # Statistics stay per layer: the layers' expert counts may differ.
        self.sizes = [layer.config.num_experts for layer in self.layers]
        self.counts: list[Tensor | None] = [None] * len(self.layers)

    @classmethod
    def from_settings(cls, model: nn.Module, settings: dict) -> "MoEBalancer | None":
        """A balancer from balancing_settings; None while balancing is off."""
        if not settings.get("bias_update_rate"):
            return None
        return cls(model, rate=settings["bias_update_rate"])

    @property
    def settings(self) -> dict:
        """The StepStatistics contract: the options that built this balancer."""
        return {"bias_update_rate": self.rate}

    def _routings(self, output):
        routings = output.expert_routing or ()
        if len(routings) != len(self.layers):
            raise ValueError(
                f"expected routing from {len(self.layers)} MoE layers, got {len(routings)}"
            )
        return routings

    def _sum_over_ranks(self, counts, process_group) -> tuple[Tensor, ...]:
        """Sum per-layer counts over ranks with one collective: cat, all-reduce, split."""
        flat = torch.cat(tuple(counts))
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(flat, group=process_group)
        return flat.split(self.sizes)

    def observe(self, output, *, keep: Tensor) -> None:
        """Count one forward's kept tokens per expert, per layer; training forwards accumulate."""
        if not (torch.is_grad_enabled() and self.layers[0].training):
            return
        routings = self._routings(output)
        kept = keep.to(routings[0].expert_ids.device)
        for index, (layer, routing) in enumerate(zip(self.layers, routings)):
            tokens = count_tokens_per_expert(
                routing.expert_ids, kept, layer.config.num_experts
            )
            previous = self.counts[index]
            self.counts[index] = tokens if previous is None else previous + tokens

    def step(self, *, process_group=None) -> dict[str, Tensor]:
        """At the optimizer step: pool the window's counts over ranks, move the biases, reset."""
        if any(counts is None for counts in self.counts):
            return {}
        loads = [
            load.float() for load in self._sum_over_ranks(self.counts, process_group)
        ]
        self.counts = [None] * len(self.layers)
        if self.rate:
            for layer, load in zip(self.layers, loads):
                layer.gate.update_expert_bias(load, rate=self.rate)
        balance = torch.stack([dispatch_balance(load) for load in loads]).mean()
        return {"moe/step_balance": balance}
