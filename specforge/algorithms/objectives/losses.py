"""Token losses and position weights, independent of algorithm execution."""

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from specforge.core import fused_loss

from .base import check_options


@dataclass(frozen=True)
class LossRegistration:
    compute: Callable
    requires: frozenset[str]
    inputs: tuple[str, ...] = ()


_LOSSES = {}


def register_loss(name, *, requires=("logits", "labels"), inputs=()):
    """Register a per-token loss; algorithm-specific auxiliary terms stay separate."""

    def register(compute):
        if name == "default" or name in _LOSSES:
            raise ValueError(f"loss already registered or reserved: {name!r}")
        required = frozenset(requires)
        capture_inputs = tuple(inputs)
        if (
            "teacher_logits" in required
            and "target_last_hidden_states" not in capture_inputs
        ):
            capture_inputs += ("target_last_hidden_states",)
        _LOSSES[name] = LossRegistration(compute, required, capture_inputs)
        return compute

    return register


@register_loss("ce")
def cross_entropy(logits, labels, teacher=None):
    return F.cross_entropy(
        logits.to(torch.promote_types(logits.dtype, torch.float32)).reshape(
            -1, logits.shape[-1]
        ),
        labels.reshape(-1),
        reduction="none",
    ).reshape(labels.shape)


@register_loss("soft_ce", requires=("logits", "labels", "teacher_logits"))
def soft_cross_entropy(logits, labels, teacher):
    return -(teacher.float().softmax(-1) * logits.float().log_softmax(-1)).sum(-1)


@register_loss("ce_l1", requires=("logits", "labels", "teacher_logits"))
def ce_l1(logits, labels, teacher, *, ce_weight=0.1, l1_weight=0.9):
    terms = fused_loss.ce_l1_loss(logits, labels, teacher_logits=teacher)
    return ce_weight * terms.cross_entropy + l1_weight * terms.teacher_l1


def _distillation(compute):
    def loss(logits, labels, teacher):
        return compute(logits, teacher)

    return loss


for _name, _compute in {
    "kl": fused_loss.kl_div_loss,
    "reverse_kl": fused_loss.reverse_kl_div_loss,
    "js": fused_loss.js_div_loss,
    "tv": fused_loss.tv_loss,
    "nla": fused_loss.neg_log_acceptance_loss,
    "lk": fused_loss.lk_hybrid_loss,
}.items():
    register_loss(_name, requires=("logits", "labels", "teacher_logits"))(
        _distillation(_compute)
    )


@dataclass(frozen=True)
class TokenLoss:
    """Select a common loss without replacing algorithm execution or auxiliary loss."""

    kind: str = "default"
    options: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind != "default" and self.kind not in _LOSSES:
            raise ValueError(f"unknown token loss: {self.kind!r}")

    @property
    def registration(self):
        return _LOSSES["ce" if self.kind == "default" else self.kind]

    @property
    def requires(self):
        return self.registration.requires

    @property
    def inputs(self):
        return self.registration.inputs

    @property
    def requires_teacher(self):
        return "teacher_logits" in self.requires

    def __call__(self, logits, labels, *, teacher=None, default=None):
        if self.kind == "default" and default is not None:
            return default
        if self.requires_teacher and (teacher is None or teacher.shape != logits.shape):
            raise ValueError("distillation requires aligned teacher and draft logits")
        return self.registration.compute(logits, labels, teacher, **self.options)


@dataclass(frozen=True)
class PositionWeighting:
    requires = frozenset({"chain_positions"})

    kind: str = "uniform"
    gamma: float | None = None
    factor: float | None = None

    def __post_init__(self):
        if self.kind not in ("uniform", "exponential", "geometric"):
            raise ValueError(f"unknown position weighting: {self.kind!r}")
        check_options(
            positive={"gamma": self.gamma}, probabilities={"factor": self.factor}
        )
        if (self.kind == "exponential") != (self.gamma is not None):
            raise ValueError("gamma is required only for exponential weighting")

        if (self.kind == "geometric") != (self.factor is not None):
            raise ValueError("factor is required only for geometric weighting")

    def __call__(self, weights, positions):
        if self.kind == "uniform":
            return weights
        if self.kind == "geometric":
            return weights * self.factor**positions
        return weights * torch.exp(-positions.float().clamp_min(0) / self.gamma)
