import math
from warnings import warn

from torch.optim.lr_scheduler import LRScheduler as _LRScheduler


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

        # Restore optimizer's lr from _last_lr: load_state_dict doesn't update
        # the optimizer's lr, so without this the resumed run would use the
        # freshly constructed scheduler's initial lr until its next step.
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
        _check_warmup_steps(total_steps, warmup_steps)
        base_scheduler = _CosineLR(
            optimizer,
            total_steps - warmup_steps,
            eta_min=eta_min,
            last_epoch=last_epoch,
        )
        super().__init__(optimizer, warmup_steps, base_scheduler, last_epoch=last_epoch)


def _check_warmup_steps(total_steps: int, warmup_steps: int) -> None:
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError(
            "warmup_steps must be in [0, total_steps), got "
            f"{warmup_steps} for total_steps={total_steps}"
        )


class _CosineLR(_LRScheduler):
    """Closed-form cosine decay from ``base_lrs`` to ``eta_min`` over ``T_max`` steps.

    Unlike ``torch.optim.lr_scheduler.CosineAnnealingLR``, the lr never depends
    on the current ``group["lr"]``, so handing over from warmup (or resuming)
    cannot inflate it. ``T_max``/``eta_min`` keep torch's attribute names so
    checkpoints saved with the torch scheduler still load.
    """

    def __init__(self, optimizer, T_max: int, eta_min: float = 0.0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        progress = min(self.last_epoch, self.T_max) / self.T_max
        factor = (1 + math.cos(math.pi * progress)) / 2
        return [
            self.eta_min + (base_lr - self.eta_min) * factor
            for base_lr in self.base_lrs
        ]


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
        _check_warmup_steps(total_steps, warmup_steps)
        base_scheduler = _FlatLR(optimizer, last_epoch=last_epoch)
        super().__init__(optimizer, warmup_steps, base_scheduler, last_epoch=last_epoch)


__all__ = ["ConstantWarmupLR", "CosineAnnealingWarmupLR"]
