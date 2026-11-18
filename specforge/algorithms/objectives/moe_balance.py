"""MoE load balancing: DeepSeek-V3's aux-loss-free bias update and its auxiliary loss."""

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from specforge.modeling.draft.layers.mlp import MoEFFN

from .base import check_options

# The load the auxiliary loss balances: this forward's, or the step's so far over ranks.
AUX_LOSS_TYPES = ("microbatch_wise", "batch_wise")
# The options that join a resume contract only while balancing is on; the balance
# loss's coefficient is the named loss term's (training.loss.terms.moe_balance).
BALANCING_OPTIONS = ("bias_update_rate", "aux_loss_type")


def check_aux_loss_type(value: str) -> None:
    if value not in AUX_LOSS_TYPES:
        raise ValueError(
            f"invalid aux_loss_type: {value!r}; expected one of {AUX_LOSS_TYPES}"
        )


def balancing_settings(options: dict) -> dict:
    """The balancing options with defaults filled in and checked."""
    settings = {"bias_update_rate": 0.0, "aux_loss_type": "batch_wise", **options}
    check_options(nonnegative={"bias_update_rate": settings["bias_update_rate"]})
    check_aux_loss_type(settings["aux_loss_type"])
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


@dataclass(frozen=True)
class PreparedLoad:
    """The load the auxiliary loss balances against: per layer [num_experts] kept tokens per expert."""

    counts: tuple[Tensor, ...]


def moe_balance_loss(output, keep: Tensor, load: PreparedLoad) -> Tensor:
    """E * sum_e share_e * P_e per layer with its own E, averaged; 1 when balanced; gradient through P_e."""
    kept = keep.to(output.expert_routing[0].expert_ids.device)
    values = []
    for routing, counts in zip(output.expert_routing, load.counts):
        # P_e over the kept tokens; the L1 normalization matters for sigmoid scores only.
        probs = (F.normalize(routing.scores, p=1, dim=-1) * kept[:, None]).sum(dim=0)
        probs = probs / kept.sum().clamp(min=1)
        share = counts.float() / counts.sum().clamp(min=1)
        values.append(probs.numel() * (share * probs).sum())
    return torch.stack(values).mean()


class MoEBalancer:
    """The objectives' StepStatistics for MoE drafts: counts, the bias step, the aux loss."""

    def __init__(self, model: nn.Module, *, rate: float, aux_loss_type="batch_wise"):
        self.rate = rate
        self.aux_loss_type = aux_loss_type
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
        # Per layer: this step's counts so far, pooled over ranks (batch_wise).
        self.pooled: tuple[Tensor, ...] | None = None

    @classmethod
    def from_settings(
        cls, model: nn.Module, settings: dict, *, balance_term: bool = False
    ) -> "MoEBalancer | None":
        """A balancer from balancing_settings; None while the bias stays and no balance term is on."""
        if not settings.get("bias_update_rate") and not balance_term:
            return None
        return cls(
            model,
            rate=settings["bias_update_rate"],
            aux_loss_type=settings["aux_loss_type"],
        )

    @property
    def settings(self) -> dict:
        """The StepStatistics contract: the options that built this balancer."""
        return {"bias_update_rate": self.rate, "aux_loss_type": self.aux_loss_type}

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

    def observe(self, output, *, keep: Tensor) -> tuple[Tensor, ...]:
        """Count one forward per layer (training forwards accumulate for the step); this forward's counts."""
        counting = torch.is_grad_enabled() and self.layers[0].training
        routings = self._routings(output)
        kept = keep.to(routings[0].expert_ids.device)
        counts = []
        for index, (layer, routing) in enumerate(zip(self.layers, routings)):
            tokens = count_tokens_per_expert(
                routing.expert_ids, kept, layer.config.num_experts
            )
            if counting:
                previous = self.counts[index]
                self.counts[index] = tokens if previous is None else previous + tokens
            counts.append(tokens)
        return tuple(counts)

    def balance(self, output, *, keep: Tensor, pooled: bool, process_group=None):
        """Count this forward; then its balance value against its own load, or (batch_wise while `pooled`) the step's so far over micro-batches and ranks."""
        counts = self.observe(output, keep=keep)
        if pooled and self.aux_loss_type == "batch_wise":
            counts = self._sum_over_ranks(counts, process_group)
            if self.pooled is not None:
                counts = tuple(a + b for a, b in zip(self.pooled, counts))
            self.pooled = counts
        return moe_balance_loss(output, keep, PreparedLoad(counts))

    def step(self, *, process_group=None) -> dict[str, Tensor]:
        """At the optimizer step: pool the window's counts over ranks, move the biases, reset."""
        if any(counts is None for counts in self.counts):
            return {}
        loads = [
            load.float() for load in self._sum_over_ranks(self.counts, process_group)
        ]
        self.counts, self.pooled = [None] * len(self.layers), None
        if self.rate:
            for layer, load in zip(self.layers, loads):
                layer.gate.update_expert_bias(load, rate=self.rate)
        balance = torch.stack([dispatch_balance(load) for load in loads]).mean()
        return {"moe/step_balance": balance}
