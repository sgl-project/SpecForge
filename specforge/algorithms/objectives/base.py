"""Objective contracts and the composite module executed by distributed training."""

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn


@dataclass
class ObjectiveResult:
    numerator: Tensor
    denominator: Tensor
    ratios: dict[str, tuple[Tensor, Tensor]] = field(default_factory=dict)

    @property
    def loss(self):
        return self.numerator / self.denominator.clamp_min(1)


def reduce_loss(per_token, weights, *, logits=None, labels=None, ratios=None):
    denominator = weights.sum().detach()
    metrics = dict(ratios or {})
    if logits is not None:
        metrics["accuracy"] = (
            ((logits.argmax(-1) == labels) * weights).sum().detach(),
            denominator,
        )
    return ObjectiveResult((per_token * weights).sum(), denominator, metrics)


class ObjectiveModel(nn.Module):
    """One wrapped forward includes the draft and every differentiable loss path.

    Teacher features are immutable batch data. The frozen head only projects
    them; this module never runs a target model or owns optimizer/checkpoint I/O.
    """

    def __init__(
        self, draft_model, objective, target_head=None, *, registration=None, name=None
    ):
        super().__init__()
        self.draft_model = draft_model
        self.objective = objective
        self.registration = registration
        self.objective_name = name
        self.target_head = target_head
        if target_head is not None:
            target_head.requires_grad_(False)
            target_head.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.target_head is not None:
            self.target_head.eval()
        return self

    def teacher_logits(self, batch, positions):
        from .inputs import gather

        with torch.no_grad():
            if "target_logits" in batch:
                return gather(batch["target_logits"], positions).detach()
            if self.target_head is None:
                raise ValueError(
                    "this objective requires frozen target logits or a target head"
                )
            hidden = batch["target_last_hidden_states"]
            return self.target_head(gather(hidden, positions).detach())

    def forward(self, batch, *, step=0, total_steps=None, alignment_ce=None):
        # Detach capture tensors centrally, including offline MTP features.
        batch = {
            key: value.detach() if isinstance(value, Tensor) else value
            for key, value in batch.items()
        }
        dtype = self.draft_model.embed_tokens.weight.dtype
        for key in ("hidden_states", "target_last_hidden_states", "mtp_hidden_states"):
            if key in batch:
                batch[key] = batch[key].to(dtype=dtype)
        return self.objective(
            self, batch, step=step, total_steps=total_steps, alignment_ce=alignment_ce
        )


def check_options(**groups):
    """Small scalar validators opted into by each objective, not registry policy."""
    for group, values in groups.items():
        for name, value in values.items():
            if value is None:
                continue
            valid = type(value) in (int, float) and math.isfinite(value)
            if group in ("counts", "zero_counts"):
                valid = valid and type(value) is int
            if group in ("positive", "counts"):
                valid = valid and value > 0
            else:
                valid = valid and value >= 0
            if group == "probabilities":
                valid = valid and value <= 1
            if not valid:
                raise ValueError(f"invalid {name}: {value!r} ({group})")
