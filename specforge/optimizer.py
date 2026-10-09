import logging
import math

import torch
import torch.distributed as dist

from specforge.lr_scheduler import ConstantWarmupLR, CosineAnnealingWarmupLR
from specforge.utils import print_on_rank0

logger = logging.getLogger(__name__)

_SCHEDULER_TYPES = {
    "constant": ConstantWarmupLR,
    "cosine": CosineAnnealingWarmupLR,
}


def _rank_local_parameter_ids(model) -> frozenset:
    """Ids of parameters each rank owns a disjoint slice of.

    A module opts its own parameters in by setting
    ``_specforge_rank_local_parameters = True``; everything else in the model
    is treated as replicated with identical gradients on every rank.
    """
    if model is None:
        return frozenset()
    ids = set()
    for module in model.modules():
        if getattr(module, "_specforge_rank_local_parameters", False):
            ids.update(id(parameter) for parameter in module.parameters())
    return frozenset(ids)


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
        # Expert parallelism hands this optimizer a process group holding a
        # MIX of rank-local and replicated parameters; these stay inert until
        # configure_grad_norm_reduction(partition_replicated=True) says so.
        self._partition_replicated = False
        self._rank_local_param_ids = frozenset()
        self._replicated_owner = {}
        self._master_index = list(range(len(self.model_params)))
        if lr_scheduler not in _SCHEDULER_TYPES:
            raise ValueError(
                f"unsupported lr_scheduler={lr_scheduler!r}; "
                f"expected one of {sorted(_SCHEDULER_TYPES)}"
            )
        self.lr_scheduler_type = lr_scheduler
        self._total_steps = total_steps
        self._warmup_steps = int(warmup_ratio * total_steps)
        self.scheduler = self._build_scheduler()

    def _build_scheduler(self):
        return _SCHEDULER_TYPES[self.lr_scheduler_type](
            self.optimizer,
            total_steps=self._total_steps,
            warmup_steps=self._warmup_steps,
        )

    def configure_grad_norm_reduction(
        self,
        *,
        process_group=None,
        enabled: bool = True,
        partition_replicated: bool = False,
    ) -> None:
        """Configure the group that owns disjoint gradient shards.

        FSDP backends disable the reduction for replicated/NO_SHARD parameters.

        ``partition_replicated`` says the group holds a *mix*: some parameters
        are rank-local (each rank owns a disjoint slice) and the rest are
        replicated with identical gradients on every rank. Expert parallelism
        is exactly that — summing every square over the group would count the
        replicated parameters once per rank and inflate the norm by up to
        ``sqrt(group_size)``, which then clips every step that much harder.
        Modules mark their rank-local parameters by setting
        ``_specforge_rank_local_parameters = True`` on themselves.
        """
        self._grad_norm_process_group = process_group
        self._reduce_grad_norm_across_ranks = enabled
        self._partition_replicated = partition_replicated
        self._rank_local_param_ids = (
            _rank_local_parameter_ids(self.model)
            if partition_replicated
            else frozenset()
        )
        if partition_replicated:
            self._shard_replicated_state()

    def _shard_replicated_state(self) -> None:
        """Keep master and Adam state for only this rank's share of the replicas.

        Under expert parallelism the rank-local parameters are already disjoint,
        but every rank holds an identical copy of the replicated ones — and an
        identical fp32 master and both Adam moments, which it updates
        identically. Assigning each replicated parameter to one owner recovers
        ``(group - 1) / group`` of that state; the owner broadcasts the updated
        weights after each step, so every rank ends the step with the same
        parameters it would have had.

        The split is by parameter count, greedy and deterministic, so every rank
        computes the same assignment without communicating.
        """
        if not (dist.is_available() and dist.is_initialized()):
            return
        try:
            group_size = dist.get_world_size(self._grad_norm_process_group)
            group_rank = dist.get_rank(self._grad_norm_process_group)
        except (RuntimeError, ValueError):
            return
        if group_size <= 1:
            return

        local_ids = self._rank_local_param_ids
        replicated = [
            index
            for index, parameter in enumerate(self.model_params)
            if id(parameter) not in local_ids
        ]
        if not replicated:
            return
        load = [0] * group_size
        owner = {}
        # Largest first so the greedy assignment balances instead of trailing
        # one huge tensor onto whichever rank happens to come last.
        for index in sorted(
            replicated, key=lambda i: self.model_params[i].numel(), reverse=True
        ):
            target = min(range(group_size), key=lambda r: (load[r], r))
            owner[index] = target
            load[target] += self.model_params[index].numel()

        keep = sorted(
            index
            for index in range(len(self.model_params))
            if index not in owner or owner[index] == group_rank
        )
        self._replicated_owner = owner
        self._rebuild_masters(keep)

    def _rebuild_masters(self, keep) -> None:
        """Restrict the fp32 masters to ``keep`` and rebuild AdamW over them.

        Safe only before the first step: AdamW state is empty at configure time,
        so nothing is discarded by replacing the parameter group.
        """
        self._master_index = list(keep)
        self.fp32_params = [
            (
                self.model_params[index]
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .clone()
                if self.offload_master
                else self.model_params[index].detach().clone().to(torch.float32)
            )
            for index in self._master_index
        ]
        for master in self.fp32_params:
            master.requires_grad = True
        defaults = self.optimizer.defaults
        self.optimizer = torch.optim.AdamW(
            self.fp32_params,
            lr=defaults["lr"],
            weight_decay=defaults["weight_decay"],
            betas=defaults["betas"],
            fused=self._adamw_fused,
        )
        # Rebuild rather than re-point: the scheduler wrote initial_lr into the
        # old parameter groups and applied its first value there, so a swapped
        # optimizer would start at the raw constructor learning rate and be one
        # warmup step out for the whole run.
        self.scheduler = self._build_scheduler()

    def _broadcast_replicated(self) -> None:
        """Publish each replicated parameter from the rank that updated it."""
        if not self._replicated_owner:
            return
        group = self._grad_norm_process_group
        by_owner = {}
        for index, rank in self._replicated_owner.items():
            by_owner.setdefault(rank, []).append(index)
        for rank in sorted(by_owner):
            indices = sorted(by_owner[rank])
            flat = torch.cat(
                [self.model_params[index].data.reshape(-1) for index in indices]
            )
            source = rank
            resolve = getattr(dist, "get_global_rank", None)
            if callable(resolve) and group is not None:
                try:
                    source = resolve(group, rank)
                except (RuntimeError, ValueError):
                    source = rank
            # One collective per owner rather than one per parameter; every rank
            # issues them in the same order, which the collective requires.
            dist.broadcast(flat, src=source, group=group)
            offset = 0
            for index in indices:
                parameter = self.model_params[index]
                count = parameter.numel()
                parameter.data.copy_(flat[offset : offset + count].view_as(parameter))
                offset += count

    def _reduce_grad_norm(self, total_norm_sq, replicated_norm_sq=None):
        """All-reduce the squared L2 norm across shard ranks and derive the
        clip coefficient.

        ``total_norm_sq`` must already live on a device the process group can
        reduce (e.g. CUDA for NCCL). When ``replicated_norm_sq`` is given it
        carries the squares of the parameters every rank holds a copy of; those
        are summed and divided back by the group size so they contribute
        exactly once, while ``total_norm_sq`` (the rank-local slices) is summed
        normally. Returns ``(total_norm, clip_coef)``.
        """
        if (
            self._reduce_grad_norm_across_ranks
            and dist.is_available()
            and dist.is_initialized()
        ):
            if replicated_norm_sq is None:
                dist.all_reduce(
                    total_norm_sq,
                    op=dist.ReduceOp.SUM,
                    group=self._grad_norm_process_group,
                )
            else:
                # One collective for both halves keeps the collective count and
                # ordering identical to the un-partitioned path.
                packed = torch.stack([total_norm_sq, replicated_norm_sq])
                dist.all_reduce(
                    packed,
                    op=dist.ReduceOp.SUM,
                    group=self._grad_norm_process_group,
                )
                group_size = dist.get_world_size(self._grad_norm_process_group)
                total_norm_sq = packed[0] + packed[1] / group_size
        elif replicated_norm_sq is not None:
            total_norm_sq = total_norm_sq + replicated_norm_sq
        total_norm = total_norm_sq.sqrt()
        clip_coef = torch.clamp(self.max_grad_norm / (total_norm + 1e-6), max=1.0)
        return total_norm, clip_coef

    def _grad_norm_and_clip_coefficient(self):
        """Compute the global grad norm from the model params on their own
        device, where NCCL can reduce it safely, without materialising master
        gradients first."""
        device = self.model_params[0].device if self.model_params else "cpu"
        zero = torch.zeros((), dtype=torch.float32, device=device)
        local_ids = self._rank_local_param_ids
        rank_local, replicated = [], []
        for parameter in self.model_params:
            if parameter.grad is None:
                continue
            if not local_ids or id(parameter) in local_ids:
                rank_local.append(parameter.grad.detach())
            else:
                replicated.append(parameter.grad.detach())
        total_norm_sq = _sum_of_squares(rank_local) if rank_local else zero
        replicated_norm_sq = (
            _sum_of_squares(replicated) if replicated else (zero if local_ids else None)
        )
        return self._reduce_grad_norm(total_norm_sq, replicated_norm_sq)

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
            # _master_index is the identity unless a replicated parameter was
            # assigned to another owner by _shard_replicated_state.
            for mp, index in zip(self.fp32_params, self._master_index):
                p = self.model_params[index]
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
            mastered = [self.model_params[index] for index in self._master_index]
            if self.offload_master:
                for p, mp in zip(mastered, self.fp32_params):
                    p.data.copy_(mp.data.to(device=p.device, dtype=p.dtype))
            elif mastered:
                torch._foreach_copy_(
                    [p.data for p in mastered],
                    [mp.data for mp in self.fp32_params],
                )
            # Parameters this rank does not master still accumulated a gradient
            # and still need it cleared before the next accumulation window.
            for p in self.model_params:
                p.grad = None
        # Only after every owner has written its update back into p.data.
        self._broadcast_replicated()
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
        if saved_scheduler_type != self.lr_scheduler_type:
            raise ValueError(
                "checkpoint optimizer used lr_scheduler="
                f"{saved_scheduler_type!r} but this run has "
                f"lr_scheduler={self.lr_scheduler_type!r}"
            )
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
                for mp, index in zip(self.fp32_params, self._master_index):
                    p = self.model_params[index]
                    mp.data.copy_(p.detach().to(device=mp.device, dtype=mp.dtype))

    def state_dict(self):
        return {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "lr_scheduler_type": self.lr_scheduler_type,
            "max_grad_norm": self.max_grad_norm,
            # rank-local fp32 masters; without them a resume re-quantizes from bf16
            "fp32_params": [t.detach().cpu() for t in self.fp32_params],
        }

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]
