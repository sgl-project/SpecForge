import logging
import math

import torch
import torch.distributed as dist

from specforge.lr_scheduler import (
    ConstantLinearDecayWarmupLR,
    ConstantWarmupLR,
    CosineAnnealingWarmupLR,
    completed_updates_from_state,
    constant_linear_decay_lr,
    warmup_updates,
)
from specforge.utils import print_on_rank0

logger = logging.getLogger(__name__)


def _sum_of_squares(tensors):
    """FP32 sum of squared L2 norms of ``tensors``.

    On CUDA one ``_foreach_norm`` launch per (device, dtype) group replaces
    three kernels per tensor; the result matches up to FP32 summation order.
    Grouping matters because a mixed-dtype list falls back to one kernel per
    tensor. Other devices keep the per-tensor reduction.
    """
    if all(tensor.is_cuda for tensor in tensors):
        groups = {}
        for tensor in tensors:
            groups.setdefault((tensor.device, tensor.dtype), []).append(tensor)
        norms = [
            norm
            for group in groups.values()
            for norm in torch._foreach_norm(group, 2.0, dtype=torch.float32)
        ]
        return torch.stack(norms).square().sum()
    return torch.stack([tensor.float().square().sum() for tensor in tensors]).sum()


class BF16Optimizer:
    """AdamW over fp32 master copies of the bf16 trainable params, with grad
    clipping and configurable warmup scheduling."""

    #: ``step(loss_denominator=...)`` validates the caller's global loss
    #: denominator in the same host read as the grad norm (see TrainerCore).
    checks_loss_denominator = True

    def __init__(
        self,
        model,
        lr,
        weight_decay=0.0,
        max_grad_norm=0.5,
        total_steps=800_000,
        warmup_ratio=0.015,
        lr_scheduler="cosine",
        offload_master=False,
        lr_decay_start_update=None,
        lr_decay_end_update=None,
    ):
        # defaults copied from EAGLE traineagle3 ds_config.json
        self.model = model
        self.model_params = [p for p in model.parameters() if p.requires_grad]
        self.max_grad_norm = max_grad_norm
        self.offload_master = bool(offload_master)
        self.fp32_params = [
            (
                p.detach().to(device="cpu", dtype=torch.float32).clone()
                if self.offload_master
                else p.detach().clone().to(torch.float32)
            )
            for p in self.model_params
        ]
        for mp in self.fp32_params:
            mp.requires_grad = True
        # One fused kernel updates every CUDA master; CPU-offloaded masters keep
        # the default AdamW implementation.
        self._adamw_fused = (
            True
            if self.fp32_params and all(mp.is_cuda for mp in self.fp32_params)
            else None
        )
        self.optimizer = torch.optim.AdamW(
            self.fp32_params,
            lr=lr,
            weight_decay=weight_decay,
            fused=self._adamw_fused,
        )
        self.last_grad_norm = None
        self._grad_norm_process_group = None
        self._reduce_grad_norm_across_ranks = True
        scheduler_types = {
            "constant": ConstantWarmupLR,
            "constant_linear_decay": ConstantLinearDecayWarmupLR,
            "cosine": CosineAnnealingWarmupLR,
        }
        if lr_scheduler not in scheduler_types:
            raise ValueError(
                f"unsupported lr_scheduler={lr_scheduler!r}; "
                f"expected one of {sorted(scheduler_types)}"
            )
        self.lr_scheduler_type = lr_scheduler
        warmup_steps = warmup_updates(warmup_ratio, total_steps)
        decay_kwargs = {}
        self.lr_decay_window = None
        if lr_scheduler == "constant_linear_decay":
            if lr_decay_start_update is None or lr_decay_end_update is None:
                raise ValueError(
                    "lr_scheduler='constant_linear_decay' requires "
                    "lr_decay_start_update and lr_decay_end_update"
                )
            self.lr_decay_window = {
                "warmup_updates": warmup_steps,
                "decay_start_update": int(lr_decay_start_update),
                "decay_end_update": int(lr_decay_end_update),
            }
            decay_kwargs = dict(
                decay_start=int(lr_decay_start_update),
                decay_end=int(lr_decay_end_update),
            )
        elif lr_decay_start_update is not None or lr_decay_end_update is not None:
            raise ValueError(
                "lr_decay_start_update/lr_decay_end_update apply only to "
                f"lr_scheduler='constant_linear_decay', not {lr_scheduler!r}"
            )
        # Set when a cooldown run branches from a ``constant`` checkpoint and
        # carried through every later checkpoint of that run.
        self.lr_schedule_branch = None
        self.scheduler = scheduler_types[lr_scheduler](
            self.optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            **decay_kwargs,
        )

    def configure_grad_norm_reduction(
        self, *, process_group=None, enabled: bool = True
    ) -> None:
        """Configure the group that owns disjoint gradient shards.

        FSDP backends disable the reduction for replicated/NO_SHARD parameters.
        """
        self._grad_norm_process_group = process_group
        self._reduce_grad_norm_across_ranks = enabled

    def _reduce_grad_norm(self, total_norm_sq):
        """All-reduce the squared L2 norm across shard ranks and derive the
        clip coefficient.

        ``total_norm_sq`` must already live on a device the process group can
        reduce (e.g. CUDA for NCCL). Returns ``(total_norm, clip_coef)``.
        """
        if (
            self._reduce_grad_norm_across_ranks
            and dist.is_available()
            and dist.is_initialized()
        ):
            dist.all_reduce(
                total_norm_sq,
                op=dist.ReduceOp.SUM,
                group=self._grad_norm_process_group,
            )
        total_norm = total_norm_sq.sqrt()
        clip_coef = torch.clamp(self.max_grad_norm / (total_norm + 1e-6), max=1.0)
        return total_norm, clip_coef

    def _grad_norm_and_clip_coefficient(self):
        """Compute the global grad norm from the model params on their own
        device, where NCCL can reduce it safely, without materialising master
        gradients first."""
        grads = [p.grad.detach() for p in self.model_params if p.grad is not None]
        if grads:
            total_norm_sq = _sum_of_squares(grads)
        else:
            device = self.model_params[0].device if self.model_params else "cpu"
            total_norm_sq = torch.zeros((), dtype=torch.float32, device=device)
        return self._reduce_grad_norm(total_norm_sq)

    def _clip_grad_norm(self):
        """Clip already-populated FP32 master gradients in place.

        Convenience entry point for optimizer tests and custom loops. When
        masters are CPU-offloaded, only the scalar norm is moved to the model
        device so a NCCL process group can still participate in the reduction.
        """
        grads = [master.grad for master in self.fp32_params if master.grad is not None]
        if grads:
            local_norm_sq = torch.stack(
                [grad.float().square().sum() for grad in grads]
            ).sum()
        else:
            master_device = self.fp32_params[0].device if self.fp32_params else "cpu"
            local_norm_sq = torch.zeros((), dtype=torch.float32, device=master_device)

        reduction_device = (
            self.model_params[0].device if self.model_params else local_norm_sq.device
        )
        total_norm, clip_coef = self._reduce_grad_norm(
            local_norm_sq.to(reduction_device)
        )
        for grad in grads:
            coefficient = (
                clip_coef
                if clip_coef.device == grad.device
                else float(clip_coef.item())
            )
            grad.mul_(coefficient)
        return total_norm

    def _clear_grads(self) -> None:
        with torch.no_grad():
            for p in self.model_params:
                p.grad = None
            for mp in self.fp32_params:
                mp.grad = None

    def step(self, *, loss_denominator=None):
        """Clip, update, and return the pre-clip global grad norm.

        The step synchronizes with the host once: the grad norm (plus the CPU
        clip coefficient when masters are offloaded) and the optional
        ``loss_denominator`` -- the all-reduced global loss denominator the
        caller already scaled the gradients by -- are read in one transfer.
        Either check fails before Adam, scheduler, or copy-back state changes.
        """
        grad_norm, clip_coefficient = self._grad_norm_and_clip_coefficient()
        checked = [grad_norm]
        if self.offload_master:
            checked.append(clip_coefficient)
        if loss_denominator is not None:
            checked.append(loss_denominator)
        # FP32 is exact for the norm and clip coefficient; the denominator
        # check needs only its sign and finiteness.
        host_values = torch.stack(
            [
                value.detach().reshape(()).to(grad_norm.device, torch.float32)
                for value in checked
            ]
        ).tolist()
        if loss_denominator is not None:
            denominator = host_values[-1]
            if not math.isfinite(denominator) or denominator <= 0:
                # Gradients were already scaled by an invalid factor.
                self._clear_grads()
                raise ValueError("global loss denominator must be finite and positive")
        if not math.isfinite(host_values[0]):
            # The norm is already all-reduced, so every rank fails before Adam,
            # scheduler, global-step, or durable-ack state can advance. Returning
            # here would make the controller record an optimizer update that did
            # not happen and permanently discard its training window.
            self._clear_grads()
            self.last_grad_norm = grad_norm.detach()
            raise FloatingPointError(
                "refusing optimizer step with non-finite global grad norm "
                f"(max_grad_norm={self.max_grad_norm})"
            )
        with torch.no_grad():
            model_grads, master_grads = [], []
            for p, mp in zip(self.model_params, self.fp32_params):
                if p.grad is None:
                    mp.grad = None
                    continue
                if self.offload_master:
                    master_grad = p.grad.detach().to(
                        device=mp.device,
                        dtype=torch.float32,
                    )
                    master_grad.mul_(host_values[1])
                else:
                    master_grad = torch.empty_like(mp)
                    model_grads.append(p.grad.detach())
                    master_grads.append(master_grad)
                mp.grad = master_grad
            if master_grads:
                torch._foreach_copy_(master_grads, model_grads)
                torch._foreach_mul_(master_grads, clip_coefficient)
        self.last_grad_norm = grad_norm.detach()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        with torch.no_grad():
            if self.offload_master:
                for p, mp in zip(self.model_params, self.fp32_params):
                    p.data.copy_(mp.data.to(device=p.device, dtype=p.dtype))
            elif self.model_params:
                torch._foreach_copy_(
                    [p.data for p in self.model_params],
                    [mp.data for mp in self.fp32_params],
                )
            for p in self.model_params:
                p.grad = None
        return self.last_grad_norm

    def _restore_adamw_implementation(self) -> None:
        """Re-apply this run's AdamW kernel choice after a checkpoint load.

        ``Optimizer.load_state_dict`` adopts the saved param-group flags, so a
        checkpoint from an unfused run would silently disable ``fused`` here
        (and a fused one would enable it for CPU masters). Fused AdamW reads
        its step counters on the parameter device; unfused keeps them on CPU.
        """
        for group in self.optimizer.param_groups:
            group["fused"] = self._adamw_fused
            if self._adamw_fused:
                group["foreach"] = None
        for param, state in self.optimizer.state.items():
            step = state.get("step")
            if isinstance(step, torch.Tensor):
                state["step"] = (
                    step.to(device=param.device, dtype=torch.float32)
                    if self._adamw_fused
                    else step.cpu()
                )

    def load_state_dict(self, state_dict):
        """Restore optimizer/scheduler state and, when present, the rank-local
        fp32 master params; without them the masters are re-cloned from the
        bf16 weights and the resume is not numerically faithful."""
        saved_scheduler_type = state_dict.get("lr_scheduler_type", "cosine")
        branch = self._check_lr_schedule_resume(state_dict, saved_scheduler_type)
        saved_max_grad_norm = state_dict.get("max_grad_norm")
        if saved_max_grad_norm is not None and float(saved_max_grad_norm) != float(
            self.max_grad_norm
        ):
            raise ValueError(
                "checkpoint optimizer used max_grad_norm="
                f"{saved_max_grad_norm} but this run has "
                f"max_grad_norm={self.max_grad_norm}"
            )
        # offload_master is a pure device-placement choice: restored fp32
        # masters and Adam moments are relocated to the current master device,
        # so toggling it on resume is safe and intentionally not gated here.
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        self._restore_adamw_implementation()
        print_on_rank0("Successfully loaded optimizer state_dict.")
        self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        print_on_rank0("Successfully loaded scheduler state_dict.")
        if self.lr_decay_window is not None:
            self._check_restored_learning_rate()
        if branch is not None and saved_scheduler_type == "constant":
            print_on_rank0(
                "Branched lr_scheduler constant -> constant_linear_decay after "
                f"update {branch['branch_update']}: {branch}"
            )
        self.lr_schedule_branch = branch
        saved_fp32 = state_dict.get("fp32_params")
        if saved_fp32 is not None:
            if len(saved_fp32) != len(self.fp32_params):
                raise ValueError(
                    f"checkpoint carries {len(saved_fp32)} fp32 master params "
                    f"but this rank has {len(self.fp32_params)}"
                )
            with torch.no_grad():
                for i, (saved, mp) in enumerate(zip(saved_fp32, self.fp32_params)):
                    if saved.shape != mp.shape:
                        raise ValueError(
                            f"fp32 master param {i} shape mismatch: checkpoint "
                            f"{tuple(saved.shape)} vs current {tuple(mp.shape)}"
                        )
                    mp.data.copy_(saved.to(mp.device, mp.dtype))
        else:
            logger.warning(
                "checkpoint has no fp32_params; re-cloning master params from "
                "bf16 weights — resume will not be numerically faithful"
            )
            with torch.no_grad():
                for p, mp in zip(self.model_params, self.fp32_params):
                    mp.data.copy_(p.detach().to(device=mp.device, dtype=mp.dtype))

    def state_dict(self):
        state = {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "lr_scheduler_type": self.lr_scheduler_type,
            "max_grad_norm": self.max_grad_norm,
            # rank-local fp32 masters; without them a resume re-quantizes from bf16
            "fp32_params": [t.detach().cpu() for t in self.fp32_params],
        }
        if self.lr_decay_window is not None:
            # Only the cooldown schedule adds keys; other types keep the
            # historical checkpoint layout.
            state["lr_decay_window"] = dict(self.lr_decay_window)
            state["lr_schedule_branch"] = (
                None
                if self.lr_schedule_branch is None
                else dict(self.lr_schedule_branch)
            )
        return state

    def _check_lr_schedule_resume(self, state_dict, saved_scheduler_type):
        """Refuse a scheduler change except the cooldown branch.

        Returns the branch metadata this run carries after the load (``None``
        unless the run is a ``constant_linear_decay`` branch of a ``constant``
        run).
        """
        current = self.lr_scheduler_type
        if saved_scheduler_type == current:
            if current != "constant_linear_decay":
                return None
            saved_window = state_dict.get("lr_decay_window")
            if saved_window != self.lr_decay_window:
                raise ValueError(
                    "checkpoint optimizer used constant_linear_decay window "
                    f"{saved_window} but this run has {self.lr_decay_window}; "
                    "a resumed cooldown must keep identical warmup/D/T"
                )
            branch = state_dict.get("lr_schedule_branch")
            if branch is not None:
                self._check_branch_window(branch)
                branch = dict(branch)
            return branch
        if saved_scheduler_type == "constant" and current == "constant_linear_decay":
            window = self.lr_decay_window
            scheduler_state = state_dict["scheduler_state_dict"]
            if scheduler_state.get("after_scheduler_type") != "_FlatLR":
                raise ValueError(
                    "checkpoint lr_scheduler='constant' has nested scheduler "
                    f"{scheduler_state.get('after_scheduler_type')!r}, "
                    "expected '_FlatLR'"
                )
            saved_warmup = int(scheduler_state["warmup_epochs"])
            if saved_warmup != window["warmup_updates"]:
                raise ValueError(
                    f"checkpoint warmup={saved_warmup} updates but this run has "
                    f"warmup={window['warmup_updates']}; the cooldown branch must "
                    "keep the constant run's warmup"
                )
            restored_update = completed_updates_from_state(scheduler_state)
            if restored_update > window["decay_start_update"]:
                raise ValueError(
                    "cannot branch lr_scheduler constant -> constant_linear_decay "
                    f"from update {restored_update}: it is past "
                    f"lr_decay_start_update={window['decay_start_update']}"
                )
            return {
                "from_type": "constant",
                "branch_update": restored_update,
                "decay_start_update": window["decay_start_update"],
                "decay_end_update": window["decay_end_update"],
            }
        raise ValueError(
            "checkpoint optimizer used lr_scheduler="
            f"{saved_scheduler_type!r} but this run has "
            f"lr_scheduler={current!r}"
        )

    def _check_branch_window(self, branch):
        window = self.lr_decay_window
        expected = {
            "from_type": "constant",
            "decay_start_update": window["decay_start_update"],
            "decay_end_update": window["decay_end_update"],
        }
        recorded = {key: branch.get(key) for key in expected}
        if recorded != expected or not isinstance(branch.get("branch_update"), int):
            raise ValueError(
                f"checkpoint lr_schedule_branch {branch} does not match this "
                f"run's constant_linear_decay window {window}"
            )

    def _check_restored_learning_rate(self):
        """The restored rate must be the cooldown rate of the next update (for a
        ``constant`` checkpoint at update <= D it is by construction)."""
        window = self.lr_decay_window
        next_update = self.scheduler.completed_updates() + 1
        for group in self.optimizer.param_groups:
            expected = constant_linear_decay_lr(
                next_update,
                group["initial_lr"],
                window["warmup_updates"],
                window["decay_start_update"],
                window["decay_end_update"],
            )
            if group["lr"] != expected:
                raise ValueError(
                    f"restored lr={group['lr']!r} for update {next_update} differs "
                    f"from constant_linear_decay lr={expected!r}"
                )

    def lr_schedule(self) -> dict:
        """The schedule's type, peak rates and update boundaries, for asserting
        per-update rates with :mod:`specforge.lr_scheduler`'s closed forms."""
        schedule = {
            "type": self.lr_scheduler_type,
            "base_lrs": [group["initial_lr"] for group in self.optimizer.param_groups],
            "warmup_updates": int(self.scheduler.warmup_epochs),
        }
        if self.lr_decay_window is not None:
            schedule.update(
                decay_start_update=self.lr_decay_window["decay_start_update"],
                decay_end_update=self.lr_decay_window["decay_end_update"],
                branch=(
                    None
                    if self.lr_schedule_branch is None
                    else dict(self.lr_schedule_branch)
                ),
            )
        return schedule

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]
