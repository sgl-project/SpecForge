from warnings import warn

from torch.optim.lr_scheduler import CosineAnnealingLR as _CosineAnnealingLR
from torch.optim.lr_scheduler import LRScheduler as _LRScheduler


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


# --------------------------------------------------------------------------
# Per-update closed forms. ``update`` is the 1-based optimizer update index:
# the value is ``param_groups[*]["lr"]`` while update ``update`` runs (the
# trainer's logged ``lr`` after completing update g is the value for g + 1).


def warmup_updates(warmup_ratio: float, total_steps: int) -> int:
    """Warmup length used by ``BF16Optimizer``: ``int(warmup_ratio * total_steps)``."""
    return int(warmup_ratio * total_steps)


def constant_warmup_lr(update: int, base_lr: float, warmup_updates: int) -> float:
    """``constant`` schedule: ``update / W * base`` for ``update <= W``, else base.

    Same float expression as ``_WarmupScheduler.get_lr`` (``last_epoch + 1 ==
    update``), so the scheduler and this helper agree bit for bit.
    """
    if update < 1:
        raise ValueError(f"update is 1-based, got {update}")
    if update <= warmup_updates:
        return update / warmup_updates * base_lr
    return base_lr


def constant_linear_decay_lr(
    update: int,
    base_lr: float,
    warmup_updates: int,
    decay_start: int,
    decay_end: int,
) -> float:
    """``constant_linear_decay``: ``constant`` through update D, then linear.

    For ``D < update <= T``: ``base_lr * ((T - update + 1) / (T - D))`` (the
    ratio is formed first, so update D+1 is exactly ``base_lr`` -- the value a
    ``constant`` checkpoint at update <= D restores -- and update T is
    ``base_lr * (1 / (T - D))``). Past T the rate is 0.0 (only reached by the
    scheduler step that follows the last update).
    """
    _validate_decay_window(warmup_updates, decay_start, decay_end)
    if update <= decay_start:
        return constant_warmup_lr(update, base_lr, warmup_updates)
    if update > decay_end:
        return 0.0
    return base_lr * ((decay_end - update + 1) / (decay_end - decay_start))


def _validate_decay_window(warmup: int, decay_start: int, decay_end: int) -> None:
    for name, value in (
        ("warmup_updates", warmup),
        ("decay_start", decay_start),
        ("decay_end", decay_end),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an int, got {value!r}")
    if not 0 <= warmup <= decay_start < decay_end:
        raise ValueError(
            "constant_linear_decay requires 0 <= warmup <= decay_start < "
            f"decay_end, got warmup={warmup}, decay_start={decay_start}, "
            f"decay_end={decay_end}"
        )


class _FlatThenLinearDecayLR(_LRScheduler):
    """Post-warmup stage of ``constant_linear_decay``.

    ``last_epoch`` counts post-warmup updates exactly like ``_FlatLR``: value
    ``e`` produces the rate of update ``W + e + 1``. A ``constant`` scheduler
    state therefore loads unchanged (this state is a superset of
    ``_FlatLR``'s); the extra decay fields must match on load.
    """

    DECAY_FIELDS = ("lr_warmup_updates", "lr_decay_start_update", "lr_decay_end_update")

    def __init__(self, optimizer, warmup_steps, decay_start, decay_end, last_epoch=-1):
        _validate_decay_window(warmup_steps, decay_start, decay_end)
        self.lr_warmup_updates = warmup_steps
        self.lr_decay_start_update = decay_start
        self.lr_decay_end_update = decay_end
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        update = self.lr_warmup_updates + self.last_epoch + 1
        return [
            constant_linear_decay_lr(
                update,
                base_lr,
                self.lr_warmup_updates,
                self.lr_decay_start_update,
                self.lr_decay_end_update,
            )
            for base_lr in self.base_lrs
        ]

    def load_state_dict(self, state_dict):
        for key in self.DECAY_FIELDS:
            if key in state_dict and state_dict[key] != getattr(self, key):
                raise ValueError(
                    f"checkpoint scheduler has {key}={state_dict[key]} but this "
                    f"run has {key}={getattr(self, key)}"
                )
        super().load_state_dict(state_dict)


class ConstantLinearDecayWarmupLR(_WarmupScheduler):
    """``constant`` (linear warmup to base, then flat) through update
    ``decay_start``, then linear decay to ``base / (decay_end - decay_start)`` at
    update ``decay_end``. See :func:`constant_linear_decay_lr`."""

    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        *,
        decay_start: int,
        decay_end: int,
        last_epoch: int = -1,
    ):
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        if not 0 <= warmup_steps < total_steps:
            raise ValueError(
                "warmup_steps must be in [0, total_steps), got "
                f"{warmup_steps} for total_steps={total_steps}"
            )
        if decay_end > total_steps:
            raise ValueError(f"decay_end={decay_end} exceeds total_steps={total_steps}")
        base_scheduler = _FlatThenLinearDecayLR(
            optimizer, warmup_steps, decay_start, decay_end, last_epoch=last_epoch
        )
        super().__init__(optimizer, warmup_steps, base_scheduler, last_epoch=last_epoch)

    def completed_updates(self) -> int:
        """Optimizer updates already applied (from the restored counters)."""
        return completed_updates_from_state(self.state_dict())


def completed_updates_from_state(state_dict) -> int:
    """Updates a saved ``constant``/``constant_linear_decay`` scheduler has seen.

    Before warmup ends the outer ``last_epoch`` counts updates; from update W on
    ``finished`` is set and the nested ``last_epoch`` counts updates past W.
    """
    if state_dict.get("finished"):
        return int(state_dict["warmup_epochs"]) + int(
            state_dict["after_scheduler_dict"]["last_epoch"]
        )
    return int(state_dict["last_epoch"])


__all__ = [
    "ConstantLinearDecayWarmupLR",
    "ConstantWarmupLR",
    "CosineAnnealingWarmupLR",
    "completed_updates_from_state",
    "constant_linear_decay_lr",
    "constant_warmup_lr",
    "warmup_updates",
]
