"""Learning-rate schedulers and the shared LR scheduler registry."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, Dict, Tuple
from warnings import warn

from torch.optim.lr_scheduler import CosineAnnealingLR as _CosineAnnealingLR
from torch.optim.lr_scheduler import LRScheduler as _LRScheduler

if TYPE_CHECKING:
    from specforge.training.contracts import SchedulerSpec


def warmup_steps_from_ratio(ratio: float, total_steps: int) -> int:
    """Return the number of complete optimizer steps in a warmup period."""
    return int(ratio * total_steps)


class _TwoStageScheduler(_LRScheduler):
    def __init__(self, optimizer, after_scheduler: _LRScheduler, last_epoch=-1):
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch)

    def state_dict(self):
        state_dict = {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }
        if isinstance(state_dict["after_scheduler"], _LRScheduler):
            state_dict["after_scheduler_type"] = type(
                state_dict["after_scheduler"]
            ).__name__
            state_dict["after_scheduler_dict"] = state_dict[
                "after_scheduler"
            ].state_dict()
            del state_dict["after_scheduler"]
        else:
            raise NotImplementedError()
        return state_dict

    def load_state_dict(self, state_dict):
        # Save _last_lr before it gets filtered out
        last_lr = state_dict.get("_last_lr", None)

        if "after_scheduler_dict" not in state_dict:
            warn(
                "after_scheduler_dict is missing; the nested scheduler state "
                "cannot be restored"
            )
        else:
            self.after_scheduler.load_state_dict(state_dict["after_scheduler_dict"])
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if key not in ("after_scheduler_type", "after_scheduler_dict")
        }
        super().load_state_dict(state_dict)

        # Restore optimizer's lr from _last_lr to ensure consistency
        # This is critical because PyTorch's CosineAnnealingLR.get_lr() uses
        # group["lr"] to compute the next lr, but load_state_dict doesn't
        # update the optimizer's lr automatically.
        if last_lr is not None:
            for param_group, lr in zip(self.optimizer.param_groups, last_lr):
                param_group["lr"] = lr


class _WarmupScheduler(_TwoStageScheduler):
    """Starts with a linear warmup lr schedule until it reaches N epochs then applies
    the specific scheduler (For example: ReduceLROnPlateau).

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        warmup_epochs (int): Number of epochs to linearly warmup lr until starting applying the scheduler.
        after_scheduler (:class:`torch.optim.lr_scheduler`): After target_epoch, use this scheduler.
        last_epoch (int, optional): The index of last epoch, defaults to -1. When last_epoch=-1,
            the schedule is started from the beginning or When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(self, optimizer, warmup_epochs, after_scheduler, last_epoch=-1):
        self.warmup_epochs = int(warmup_epochs)
        super().__init__(optimizer, after_scheduler, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epochs:
            if not self.finished:
                self.after_scheduler.base_lrs = self.base_lrs
                self.finished = True
            return self.after_scheduler.get_lr()

        return [(self.last_epoch + 1) / self.warmup_epochs * lr for lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished:
            if epoch is None:
                self.after_scheduler.step(None)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                self.after_scheduler.step(epoch - self.warmup_epochs)
                self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super().step(epoch)


class CosineAnnealingWarmupLR(_WarmupScheduler):
    """Cosine annealing learning rate scheduler with learning rate warmup. A linear warmup schedule will be applied.

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        total_steps (int): Number of total training steps.
        warmup_steps (int, optional): Number of warmup steps, defaults to 0.
        eta_min (int, optional): Minimum learning rate, defaults to 0.
        last_epoch (int, optional): The index of last epoch, defaults to -1. When last_epoch=-1,
            the schedule is started from the beginning or When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        base_scheduler = _CosineAnnealingLR(
            optimizer,
            total_steps - warmup_steps,
            eta_min=eta_min,
            last_epoch=last_epoch,
        )
        super().__init__(optimizer, warmup_steps, base_scheduler, last_epoch=last_epoch)


class _FlatLR(_LRScheduler):
    """Keep every parameter group at its configured base learning rate."""

    def get_lr(self):
        return self.base_lrs


class ConstantWarmupLR(_WarmupScheduler):
    """Linear warmup followed by a constant learning rate."""

    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        last_epoch: int = -1,
    ):
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        if not 0 <= warmup_steps < total_steps:
            raise ValueError(
                "warmup_steps must be in [0, total_steps), got "
                f"{warmup_steps} for total_steps={total_steps}"
            )
        base_scheduler = _FlatLR(optimizer, last_epoch=last_epoch)
        super().__init__(optimizer, warmup_steps, base_scheduler, last_epoch=last_epoch)


def _validate_wsd_options(spec: SchedulerSpec, *, total_steps: int | None) -> None:
    unknown = sorted(set(spec.extra) - {"decay_ratio", "decay_style"})
    if unknown:
        raise ValueError(f"unknown wsd scheduler options: {unknown}")
    decay_ratio = float(spec.extra.get("decay_ratio", 0.1))
    if not 0.0 <= decay_ratio <= 1.0:
        raise ValueError(f"wsd decay_ratio must be in [0, 1], got {decay_ratio}")
    decay_style = spec.extra.get("decay_style", "linear")
    if decay_style not in ("linear", "cosine"):
        raise ValueError(
            f"wsd decay_style must be 'linear' or 'cosine', got {decay_style!r}"
        )
    if total_steps is not None:
        warmup = warmup_steps_from_ratio(spec.warmup_ratio, total_steps)
        decay_steps = int(decay_ratio * total_steps)
        if warmup + decay_steps > total_steps:
            raise ValueError(
                f"wsd warmup ({warmup}) + decay ({decay_steps}) exceeds "
                f"total_steps ({total_steps})"
            )


def validate_scheduler_spec(
    spec: SchedulerSpec, *, require_total_steps: bool = True
) -> None:
    """Validate scheduler registration and built-in option combinations."""
    if spec.kind not in _LR_SCHEDULERS:
        raise ValueError(
            f"unsupported lr_scheduler={spec.kind!r}; "
            f"expected one of {sorted(_LR_SCHEDULERS)}"
        )
    if not 0.0 <= spec.warmup_ratio <= 1.0:
        raise ValueError(f"warmup_ratio must be in [0, 1], got {spec.warmup_ratio}")
    if not 0.0 <= spec.min_lr_ratio <= 1.0:
        raise ValueError(f"min_lr_ratio must be in [0, 1], got {spec.min_lr_ratio}")
    if spec.kind in ("cosine", "constant", "linear") and spec.extra:
        raise ValueError(
            f"{spec.kind} scheduler accepts no extra options, got {sorted(spec.extra)}"
        )
    if spec.kind == "constant" and spec.min_lr_ratio != 0.0:
        raise ValueError(
            f"constant scheduler does not use min_lr_ratio={spec.min_lr_ratio}"
        )
    total = spec.total_steps
    if require_total_steps or total is not None:
        if total is None or total <= 0:
            raise ValueError(f"total_steps must be positive, got {total}")
        if not (spec.kind == "cosine" and spec.min_lr_ratio == 0.0):
            warmup = warmup_steps_from_ratio(spec.warmup_ratio, total)
            if not 0 <= warmup < total:
                raise ValueError(
                    "warmup_steps must be in [0, total_steps), got "
                    f"{warmup} for total_steps={total}"
                )
    if spec.kind == "wsd":
        _validate_wsd_options(spec, total_steps=total)


def _cosine_factor(step: int, spec: SchedulerSpec) -> float:
    total = spec.total_steps
    warmup = warmup_steps_from_ratio(spec.warmup_ratio, total)
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    decay_steps = max(1, total - warmup)
    progress = min(1.0, max(0.0, (step - warmup) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return spec.min_lr_ratio + (1.0 - spec.min_lr_ratio) * cosine


def _linear_factor(step: int, spec: SchedulerSpec) -> float:
    total = spec.total_steps
    warmup = warmup_steps_from_ratio(spec.warmup_ratio, total)
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    decay_steps = max(1, total - warmup)
    progress = min(1.0, max(0.0, (step - warmup) / decay_steps))
    return spec.min_lr_ratio + (1.0 - spec.min_lr_ratio) * (1.0 - progress)


def _wsd_factor(step: int, spec: SchedulerSpec) -> float:
    """Warmup-Stable-Decay multiplier.

    Options in ``spec.extra``:
        decay_ratio: Fraction of ``total_steps`` spent decaying (default 0.1).
        decay_style: ``"linear"`` (default) or ``"cosine"``.
    """
    total = spec.total_steps
    warmup = warmup_steps_from_ratio(spec.warmup_ratio, total)
    decay_steps = int(float(spec.extra.get("decay_ratio", 0.1)) * total)
    decay_style = spec.extra.get("decay_style", "linear")
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    stable_end = total - decay_steps
    if decay_steps <= 0 or step < stable_end:
        return 1.0
    progress = min(1.0, max(0.0, (step - stable_end) / decay_steps))
    tail = (
        0.5 * (1.0 + math.cos(math.pi * progress))
        if decay_style == "cosine"
        else 1.0 - progress
    )
    return spec.min_lr_ratio + (1.0 - spec.min_lr_ratio) * tail


class FunctionalLRScheduler(_LRScheduler):
    """Scheduler driven by a pure scalar curve ``factor(step, spec) -> float``.

    Every step's learning rate is computed directly from ``last_epoch`` and
    ``base_lrs``, with no nested scheduler state that can drift on resume.
    """

    def __init__(
        self,
        optimizer,
        spec: SchedulerSpec,
        curve: Callable[[int, SchedulerSpec], float],
        *,
        last_epoch: int = -1,
    ) -> None:
        self.spec = spec
        self._curve = curve
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        factor = float(self._curve(self.last_epoch, self.spec))
        return [base_lr * factor for base_lr in self.base_lrs]

    def state_dict(self):
        return {
            "last_epoch": self.last_epoch,
            "base_lrs": list(self.base_lrs),
            "_last_lr": list(self._last_lr),
        }

    def load_state_dict(self, state_dict):
        self.last_epoch = int(state_dict["last_epoch"])
        self.base_lrs = list(state_dict["base_lrs"])
        self._last_lr = list(state_dict["_last_lr"])
        for group, lr in zip(self.optimizer.param_groups, self._last_lr):
            group["lr"] = lr


def _build_cosine(optimizer, spec: SchedulerSpec) -> _LRScheduler:
    if spec.min_lr_ratio == 0.0:
        return CosineAnnealingWarmupLR(
            optimizer,
            total_steps=spec.total_steps,
            warmup_steps=warmup_steps_from_ratio(spec.warmup_ratio, spec.total_steps),
        )
    return FunctionalLRScheduler(optimizer, spec, _cosine_factor)


def _build_constant(optimizer, spec: SchedulerSpec) -> _LRScheduler:
    return ConstantWarmupLR(
        optimizer,
        total_steps=spec.total_steps,
        warmup_steps=warmup_steps_from_ratio(spec.warmup_ratio, spec.total_steps),
    )


def _build_linear(optimizer, spec: SchedulerSpec) -> _LRScheduler:
    return FunctionalLRScheduler(optimizer, spec, _linear_factor)


def _build_wsd(optimizer, spec: SchedulerSpec) -> _LRScheduler:
    return FunctionalLRScheduler(optimizer, spec, _wsd_factor)


_LR_SCHEDULERS: Dict[str, Callable[[object, SchedulerSpec], _LRScheduler]] = {
    "cosine": _build_cosine,
    "constant": _build_constant,
    "linear": _build_linear,
    "wsd": _build_wsd,
}


def register_lr_scheduler(
    name: str,
    builder: Callable[[object, SchedulerSpec], _LRScheduler],
    *,
    override: bool = False,
) -> None:
    """Register a learning-rate scheduler builder under ``name``."""
    if not name:
        raise ValueError("scheduler name must be non-empty")
    if name in _LR_SCHEDULERS and not override:
        raise ValueError(f"lr_scheduler {name!r} is already registered")
    _LR_SCHEDULERS[name] = builder


def register_lr_curve(
    name: str,
    curve: Callable[[int, SchedulerSpec], float],
    *,
    override: bool = False,
) -> None:
    """Register a pure scalar LR curve ``curve(step, spec) -> multiplier``."""

    def _build(optimizer, spec: SchedulerSpec) -> _LRScheduler:
        return FunctionalLRScheduler(optimizer, spec, curve)

    register_lr_scheduler(name, _build, override=override)


def registered_lr_schedulers() -> Tuple[str, ...]:
    """Registered learning-rate scheduler names, sorted."""
    return tuple(sorted(_LR_SCHEDULERS))


def build_lr_scheduler(optimizer, spec: SchedulerSpec) -> _LRScheduler:
    """Build the learning-rate scheduler described by ``spec``."""
    validate_scheduler_spec(spec, require_total_steps=True)
    return _LR_SCHEDULERS[spec.kind](optimizer, spec)


__all__ = [
    "ConstantWarmupLR",
    "CosineAnnealingWarmupLR",
    "FunctionalLRScheduler",
    "build_lr_scheduler",
    "register_lr_curve",
    "register_lr_scheduler",
    "registered_lr_schedulers",
    "validate_scheduler_spec",
    "warmup_steps_from_ratio",
]
