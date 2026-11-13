"""Token losses and position weights, independent of algorithm execution."""

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from specforge.core.position_weights import DPACE_LOSS_TYPES

from .base import check_options
from .implementations import fused_loss


@dataclass
class LossOutput:
    """Per-position loss values and reusable named statistics.

    Values retain gradients. Statistics are detached by the executor before
    auxiliary callbacks consume them; they are local to this evaluation/chunk.
    """

    values: torch.Tensor
    statistics: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class LossRegistration:
    """Mathematical loss definition and its data requirements.

    Attributes:
        compute: Reference per-position loss callable.
        requires: Capabilities the prediction set must provide.
        inputs: Additional batch/capture fields needed by the loss.
        contextual: Whether evaluation needs whole-set statistics and context.
    """

    compute: Callable
    requires: frozenset[str]
    inputs: tuple[str, ...] = ()
    contextual: bool = False


_LOSSES = {}


def register_loss(
    name,
    *,
    requires=("logits", "labels"),
    inputs=(),
    contextual=False,
):
    """Register a mathematical per-position loss and its reference formula.

    Args:
        name: Unique loss identifier; "default" is reserved.
        requires: Prediction capabilities required by this formula.
        inputs: Additional batch fields required for supervision/capture.
        contextual: Whether compute also accepts a LossContext and needs
            whole-set evaluation rather than independent chunks.

    Returns:
        A decorator preserving the callable. It must accept
        (logits, labels, teacher, **options), plus context when contextual, and
        return a tensor aligned with labels or a LossOutput with reusable
        statistics. Optimized kernels register separately.

    Raises:
        ValueError: The identifier is reserved or already registered.
    """

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


@register_loss(
    "soft_ce",
    requires=("logits", "labels", "teacher_logits"),
)
def soft_cross_entropy(logits, labels, teacher):
    return -(teacher.float().softmax(-1) * logits.float().log_softmax(-1)).sum(-1)


@register_loss("ce_l1", requires=("logits", "labels", "teacher_logits"))
def ce_l1(logits, labels, teacher, *, ce_weight=0.1, l1_weight=0.9):
    terms = fused_loss.ce_l1_loss(logits, labels, teacher_logits=teacher)
    return LossOutput(
        ce_weight * terms.cross_entropy + l1_weight * terms.teacher_l1,
        {"teacher_l1": terms.teacher_l1},
    )


def _distillation(compute):
    def loss(logits, labels, teacher, **options):
        return compute(logits, teacher, **options)

    loss._options_from = compute
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


def _lk(mode, *, ce_compute=None):
    def compute(
        logits, labels, teacher, *, context, target="labels", kl_scale=1.0, kl_decay=1.0
    ):
        if target == "labels":
            ce = (
                ce_compute(logits, labels)
                if ce_compute is not None
                else -logits.float()
                .log_softmax(-1)
                .gather(-1, labels[..., None])
                .squeeze(-1)
            )
            acceptance = (-ce).exp()
        else:
            log_q = logits.float().log_softmax(-1)
            p = teacher.float().softmax(-1)
            ce = -(p * log_q).sum(-1)
            if context.teacher_mass is not None:
                p = p * context.teacher_mass[..., None]
            acceptance = torch.minimum(p, log_q.exp()).sum(-1)
        # Legacy EAGLE normalizes CE by all positions, acceptance by supervised ones.
        scale = 1
        mask = context.mask.float()
        count = mask.sum()
        if context.denominator is not None:
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


def _accepted_options(compute):
    """Keyword options a loss accepts, or None when its signature cannot tell."""
    try:
        parameters = inspect.signature(
            getattr(compute, "_options_from", compute), follow_wrapped=False
        ).parameters.values()
    except (TypeError, ValueError):
        return None
    if any(p.kind is p.VAR_KEYWORD for p in parameters):
        return None
    return {
        p.name
        for p in parameters
        if p.kind is p.KEYWORD_ONLY
        or (p.default is not p.empty and p.name != "teacher")
    }


def decay_weighting(gamma):
    """A block protocol's default weighting: exponential decay when gamma is set."""
    if gamma is None:
        return {"kind": "uniform"}
    return {"kind": "exponential", "gamma": gamma}


@dataclass(frozen=True)
class TokenLoss:
    """Select a common loss without replacing algorithm execution or auxiliary loss."""

    kind: str = "default"
    options: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind != "default" and self.kind not in _LOSSES:
            raise ValueError(f"unknown token loss: {self.kind!r}")
        if self.kind == "default" and self.options:
            raise ValueError("loss options require an explicit loss kind")
        if self.options and not self.registration.contextual:
            accepted = _accepted_options(self.registration.compute)
            unknown = set() if accepted is None else self.options.keys() - accepted
            if unknown:
                raise ValueError(f"unknown {self.kind} options: {sorted(unknown)}")

        if self.kind in ("lk_alpha", "lk_tv", "lk_lambda"):
            accepted = {"target"} | (
                {"kl_scale", "kl_decay"} if self.kind == "lk_lambda" else set()
            )
            unknown = self.options.keys() - accepted
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
        """Return per-position reference values, preserving the tensor-only API."""
        if self.kind == "default" and default is not None:
            return default
        return self.evaluate(logits, labels, teacher=teacher, context=context).values

    def evaluate(self, logits, labels, *, teacher=None, context=None):
        """Evaluate the mathematical reference loss without weighting or reduction.

        Args:
            logits: Draft scores with vocabulary in the final dimension.
            labels: Token IDs aligned with the scores' leading dimensions.
            teacher: Optional aligned teacher logits required by distillation.
            context: Optional supervision/normalization context for contextual losses.

        Returns:
            LossOutput containing per-position values and optional statistics.

        Raises:
            ValueError: Distillation teacher logits are absent or misaligned.
        """
        if self.requires_teacher and (teacher is None or teacher.shape != logits.shape):
            raise ValueError("distillation requires aligned teacher and draft logits")
        kwargs = {"context": context} if self.registration.contextual else {}
        result = self.registration.compute(
            logits, labels, teacher, **kwargs, **self.options
        )
        return result if isinstance(result, LossOutput) else LossOutput(result)


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
            return weights * self.factor ** positions.clamp_min(0)
        return weights * torch.exp(-positions.float().clamp_min(0) / self.gamma)
