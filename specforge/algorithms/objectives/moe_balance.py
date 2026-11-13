"""Expert load balancing for MoE drafts: DeepSeek-V3's aux-loss-free bias update and
its load-balance auxiliary loss on the router scores."""

from dataclasses import dataclass, replace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from specforge.modeling.draft.layers.mlp import MoEFFN

# The load the auxiliary loss balances: this forward's own, or this optimizer step's
# so far, pooled over micro-batches and ranks.
AUX_LOSS_TYPES = ("microbatch_wise", "batch_wise")


def check_aux_loss_type(value: str) -> None:
    if value not in AUX_LOSS_TYPES:
        raise ValueError(
            f"invalid aux_loss_type: {value!r}; expected one of {AUX_LOSS_TYPES}"
        )


def count_tokens_per_expert(
    expert_ids: Tensor, keep: Tensor, num_experts: int
) -> Tensor:
    """[T, K] expert ids and a [T] keep mask -> int64 [E] counts over the kept tokens."""
    return torch.bincount(expert_ids[keep].reshape(-1), minlength=num_experts)


def dispatch_balance(counts: Tensor) -> Tensor:
    """A dispatch's balance against itself: 1 when uniform, E / K when collapsed."""
    share = counts.float() / counts.sum().clamp(min=1)
    return share.numel() * share.square().sum()


def balance_from_terms(mean_probs: Tensor, load: Tensor) -> Tensor:
    """E * sum_e share_e * P_e per layer, averaged over the layers: share_e is expert e's
    part of `load` [L, E] and P_e its mean routing probability [L, E]; 1 when balanced.
    The gradient reaches the routers through P_e only."""
    share = load.float() / load.sum(dim=-1, keepdim=True).clamp(min=1)
    return (mean_probs.shape[-1] * (share * mean_probs).sum(dim=-1)).mean()


@dataclass(frozen=True)
class MoEBalanceTerm:
    """One forward's halves of the load-balance loss, per MoE layer: the trainer supplies
    the load and finishes it, so the objective performs no collective."""

    mean_probs: Tensor  # [layers, num_experts] P_e, differentiable
    counts: Tensor  # [layers, num_experts] int64 kept tokens routed to each expert


class MoEBalancer:
    """Counts the kept tokens each MoE layer routes over an optimizer step and, at the
    step, pools them over ranks and moves every router's expert bias by `rate`. For the
    auxiliary loss it turns a forward's routing into a MoEBalanceTerm and later finishes
    it with the load: the forward's own, or the step's so far pooled over ranks."""

    def __init__(
        self,
        model: nn.Module,
        *,
        rate: float,
        aux_loss_coef: float = 0.0,
        aux_loss_type: str = "batch_wise",
    ):
        check_aux_loss_type(aux_loss_type)
        self.rate = rate
        self.aux_loss_coef = aux_loss_coef
        self.aux_loss_type = aux_loss_type
        self.layers = [m for m in model.modules() if isinstance(m, MoEFFN)]
        if not self.layers:
            raise ValueError("MoE load balancing needs a draft with MoE layers")
        if rate and any(layer.gate.expert_bias is None for layer in self.layers):
            raise ValueError(
                "bias_update_rate needs router nodes with expert_bias: true"
            )
        self.counts: list[Tensor | None] = [None] * len(self.layers)
        # [layers, num_experts]: this step's counts so far, pooled over ranks (batch_wise).
        self.pooled: Tensor | None = None

    @classmethod
    def for_objective(
        cls, model: nn.Module, objective, registration
    ) -> "MoEBalancer | None":
        """A balancer when the objective asks for one and declares expert routing."""
        rate = getattr(objective, "bias_update_rate", 0.0)
        coef = getattr(objective, "aux_loss_coef", 0.0)
        if not rate and not coef:
            return None
        if "expert_routing" not in registration.provides:
            raise ValueError(
                f"objective {type(objective).__name__} does not provide expert "
                "routing, so it cannot balance MoE layers"
            )
        return cls(
            model,
            rate=rate,
            aux_loss_coef=coef,
            aux_loss_type=getattr(objective, "aux_loss_type", "batch_wise"),
        )

    def _routings(self, output):
        routings = output.expert_routing or ()
        if len(routings) != len(self.layers):
            raise ValueError(
                f"expected routing from {len(self.layers)} MoE layers, got {len(routings)}"
            )
        return routings

    def observe(self, output, *, keep: Tensor) -> None:
        """Count one training forward; forwards without gradients (eval, prepass) count nothing."""
        if not torch.is_grad_enabled():
            return
        for index, (layer, routing) in enumerate(
            zip(self.layers, self._routings(output))
        ):
            counts = count_tokens_per_expert(
                routing.expert_ids,
                keep.to(routing.expert_ids.device),
                layer.config.num_experts,
            )
            previous = self.counts[index]
            self.counts[index] = counts if previous is None else previous + counts

    def term(self, output, *, keep: Tensor) -> MoEBalanceTerm | None:
        """This forward's halves of the auxiliary loss; None when the loss is off."""
        if not self.aux_loss_coef:
            return None
        mean_probs, counts = [], []
        for layer, routing in zip(self.layers, self._routings(output)):
            kept = keep.to(routing.expert_ids.device)
            # P_e over the kept tokens; the per-token L1 normalization is a no-op for
            # softmax scores and needed for sigmoid ones.
            probs = F.normalize(routing.scores, p=1, dim=-1) * kept[:, None]
            mean_probs.append(probs.sum(dim=0) / kept.sum().clamp(min=1))
            counts.append(
                count_tokens_per_expert(
                    routing.expert_ids, kept, layer.config.num_experts
                )
            )
        return MoEBalanceTerm(torch.stack(mean_probs), torch.stack(counts))

    def finish(self, result, *, pooled: bool, process_group=None):
        """Fold the result's balance term into its loss and report it as moe/aux_loss.
        `pooled` (training forwards) lets batch_wise add the forward's counts, summed
        over ranks, to this step's load; otherwise the forward balances its own."""
        term = result.balance
        if term is None:
            return result
        load = term.counts
        if pooled and self.aux_loss_type == "batch_wise":
            load = load.clone()
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(load, group=process_group)
            self.pooled = load if self.pooled is None else self.pooled + load
            load = self.pooled
        value = balance_from_terms(term.mean_probs, load)
        numerator = result.numerator + self.aux_loss_coef * value * result.denominator
        ratios = {
            **result.ratios,
            "moe/aux_loss": (value.detach(), result.denominator.new_ones(())),
        }
        return replace(result, numerator=numerator, ratios=ratios, balance=None)

    def step(self, *, process_group=None) -> dict[str, Tensor]:
        """At the optimizer step: pool the window's counts over ranks, move the biases, reset."""
        if any(counts is None for counts in self.counts):
            return {}
        pooled = torch.stack(self.counts)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(pooled, group=process_group)
        self.counts, self.pooled = [None] * len(self.layers), None
        pooled = pooled.float()
        if self.rate:
            for layer, load in zip(self.layers, pooled):
                layer.gate.update_expert_bias(load, rate=self.rate)
        balance = torch.stack([dispatch_balance(load) for load in pooled]).mean()
        return {"moe/step_balance": balance}
