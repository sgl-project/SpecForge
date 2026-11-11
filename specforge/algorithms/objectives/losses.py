"""Token losses and position weights, independent of algorithm execution."""

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from specforge.core import fused_loss
from specforge.core.position_weights import DPACE_LOSS_TYPES

from .base import check_options


@dataclass(frozen=True)
class LossRegistration:
    compute: Callable
    requires: frozenset[str]
    inputs: tuple[str, ...] = ()
    contextual: bool = False


_LOSSES = {}


def register_loss(name, *, requires=("logits", "labels"), inputs=(), contextual=False):
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
        _LOSSES[name] = LossRegistration(compute, required, capture_inputs, contextual)
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
    def loss(logits, labels, teacher, **options):
        return compute(logits, teacher, **options)

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
class LossContext:
    mask: torch.Tensor
    teacher_mass: torch.Tensor | None = None
    denominator: torch.Tensor | None = None


def _lk(mode):
    def compute(
        logits, labels, teacher, *, context, target="labels", kl_scale=1.0, kl_decay=1.0
    ):
        log_q = logits.float().log_softmax(-1)
        if target == "labels":
            ce = -log_q.gather(-1, labels[..., None]).squeeze(-1)
            acceptance = (-ce).exp()
        else:
            p = teacher.float().softmax(-1)
            ce = -(p * log_q).sum(-1)
            if context.teacher_mass is not None:
                p = p * context.teacher_mass[..., None]
            acceptance = torch.minimum(p, log_q.exp()).sum(-1)
        # Legacy EAGLE normalizes CE by all positions, acceptance by supervised ones.
        scale = 1
        mask = context.mask.float()
        count = mask.sum()
        if target == "teacher" and context.denominator is not None:
            scale = context.denominator / count.where(count > 0, 1)
        if mode == "alpha":
            return (
                ce
                if target == "labels"
                else -torch.where(acceptance > 0, acceptance, 1).log() * scale
            )
        tv = (1 - acceptance) * scale
        if mode == "tv":
            return tv
        mean_acceptance = (acceptance.detach() * mask).sum() / count.where(count > 0, 1)
        weight = kl_scale * torch.exp(-kl_decay * mean_acceptance)
        return weight * ce + (1 - weight) * tv

    return compute


for _mode in ("alpha", "tv", "lambda"):
    register_loss(f"lk_{_mode}", contextual=True)(_lk(_mode))


@dataclass(frozen=True)
class TokenLoss:
    """Select a common loss without replacing algorithm execution or auxiliary loss."""

    kind: str = "default"
    options: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind != "default" and self.kind not in _LOSSES:
            raise ValueError(f"unknown token loss: {self.kind!r}")

        if self.kind in ("lk_alpha", "lk_tv", "lk_lambda"):
            unknown = self.options.keys() - {"target", "kl_scale", "kl_decay"}
            if unknown:
                raise ValueError(f"unknown LK options: {sorted(unknown)}")
            if self.options.get("target", "labels") not in ("labels", "teacher"):
                raise ValueError("LK target must be labels or teacher")
            check_options(
                nonnegative={
                    key: self.options.get(key, 1.0) for key in ("kl_scale", "kl_decay")
                }
            )

    @property
    def registration(self):
        return _LOSSES["ce" if self.kind == "default" else self.kind]

    @property
    def requires(self):
        required = self.registration.requires
        if self.registration.contextual and self.options.get("target") == "teacher":
            required |= {"teacher_logits"}
        return required

    @property
    def inputs(self):
        inputs = self.registration.inputs
        if self.requires_teacher and "target_last_hidden_states" not in inputs:
            inputs += ("target_last_hidden_states",)
        return inputs

    @property
    def requires_teacher(self):
        return "teacher_logits" in self.requires

    def __call__(self, logits, labels, *, teacher=None, default=None, context=None):
        if self.kind == "default" and default is not None:
            return default
        if self.requires_teacher and (teacher is None or teacher.shape != logits.shape):
            raise ValueError("distillation requires aligned teacher and draft logits")
        kwargs = {"context": context} if self.registration.contextual else {}
        return self.registration.compute(
            logits, labels, teacher, **kwargs, **self.options
        )


@dataclass(frozen=True)
class PositionWeighting:
    @property
    def requires(self):
        if self.kind == "uniform":
            return frozenset()
        flags = {"chain_positions"}
        if self.kind in DPACE_LOSS_TYPES:
            flags |= {"logits", "labels", "acceptance_chain"}
        return frozenset(flags)

    kind: str = "uniform"
    gamma: float | None = None
    factor: float | None = None
    alpha: float | None = None

    def __post_init__(self):
        if self.kind not in {"uniform", "exponential", "geometric"} | DPACE_LOSS_TYPES:
            raise ValueError(f"unknown position weighting: {self.kind!r}")
        check_options(
            positive={"gamma": self.gamma},
            probabilities={"factor": self.factor, "alpha": self.alpha},
        )
        if (self.kind == "exponential") != (self.gamma is not None):
            raise ValueError("gamma is required only for exponential weighting")

        if (self.kind == "geometric") != (self.factor is not None):
            raise ValueError("factor is required only for geometric weighting")

        if self.kind in DPACE_LOSS_TYPES:
            if self.alpha is None:
                object.__setattr__(self, "alpha", 0.5)
        elif self.alpha is not None:
            raise ValueError("alpha is only valid for D-PACE")

    def __call__(self, weights, positions):
        if self.kind in DPACE_LOSS_TYPES:
            raise ValueError("D-PACE requires complete chains in the batch executor")
        if self.kind == "uniform":
            return weights
        if self.kind == "geometric":
            return weights * self.factor**positions
        return weights * torch.exp(-positions.float().clamp_min(0) / self.gamma)
