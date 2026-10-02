from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.distributed as dist

from specforge.lr_scheduler import ConstantWarmupLR, CosineAnnealingWarmupLR
from specforge.muon import (
    ADAMW_OPTIMIZER,
    DEFAULT_MUON_EXCLUDED_MODULES,
    MUON_OPTIMIZER,
    SUPPORTED_OPTIMIZERS,
    FSDPShardedMuon,
    MuonParameterMetadata,
    MuonParameterPartition,
    partition_parameters_for_muon,
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


@dataclass
class _SchedulerCollection:
    """Expose multiple schedulers through the existing scheduler interface."""

    schedulers: Mapping[str, torch.optim.lr_scheduler.LRScheduler]

    _FORMAT_VERSION = 1

    def step(self) -> None:
        for scheduler in self.schedulers.values():
            scheduler.step()

    def state_dict(self) -> dict:
        return {
            "format_version": self._FORMAT_VERSION,
            "schedulers": {
                name: scheduler.state_dict()
                for name, scheduler in self.schedulers.items()
            },
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if state_dict.get("format_version") != self._FORMAT_VERSION:
            raise ValueError(
                "Unsupported hybrid scheduler state format: "
                f"{state_dict.get('format_version')!r}"
            )
        scheduler_states = state_dict.get("schedulers")
        if not isinstance(scheduler_states, dict):
            raise ValueError("Hybrid scheduler state is missing 'schedulers'")
        if set(scheduler_states) != set(self.schedulers):
            raise ValueError(
                "Hybrid scheduler groups do not match: "
                f"expected={sorted(self.schedulers)}, "
                f"received={sorted(scheduler_states)}"
            )
        for name, scheduler in self.schedulers.items():
            scheduler.load_state_dict(scheduler_states[name])


class BF16Optimizer:
    """FP32-master AdamW or hybrid Muon, with clipping and warmup scheduling."""

    _HYBRID_STATE_FORMAT_VERSION = 2

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
        *,
        optimizer_type: str = ADAMW_OPTIMIZER,
        muon_lr: float | None = None,
        muon_weight_decay: float = 0.1,
        muon_momentum: float = 0.95,
        muon_nesterov: bool = True,
        muon_ns_steps: int = 5,
        muon_adjust_lr_fn: str = "match_rms_adamw",
        muon_excluded_module_names: Sequence[str] = DEFAULT_MUON_EXCLUDED_MODULES,
        muon_metadata: MuonParameterMetadata | None = None,
    ):
        optimizer_type = optimizer_type.lower()
        if optimizer_type not in SUPPORTED_OPTIMIZERS:
            raise ValueError(
                f"Unknown optimizer_type={optimizer_type!r}; "
                f"expected one of {SUPPORTED_OPTIMIZERS}"
            )
        if optimizer_type == MUON_OPTIMIZER and offload_master:
            raise ValueError("Muon does not support optimizer CPU offload")

        self.optimizer_type = optimizer_type
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
        if optimizer_type == ADAMW_OPTIMIZER:
            self.optimizer = torch.optim.AdamW(
                self.fp32_params,
                lr=lr,
                weight_decay=weight_decay,
                fused=self._adamw_fused,
            )
            self._optimizers = {ADAMW_OPTIMIZER: self.optimizer}
        else:
            partition = partition_parameters_for_muon(
                model,
                excluded_module_names=muon_excluded_module_names,
                metadata=muon_metadata,
            )
            master_by_name = dict(
                zip(
                    (
                        name
                        for name, parameter in model.named_parameters()
                        if parameter.requires_grad
                    ),
                    self.fp32_params,
                )
            )
            self._init_muon(
                partition=partition,
                master_by_name=master_by_name,
                adamw_lr=lr,
                adamw_weight_decay=weight_decay,
                muon_lr=lr if muon_lr is None else muon_lr,
                muon_weight_decay=muon_weight_decay,
                muon_momentum=muon_momentum,
                muon_nesterov=muon_nesterov,
                muon_ns_steps=muon_ns_steps,
                muon_adjust_lr_fn=muon_adjust_lr_fn,
            )
        self.last_grad_norm = None
        self._grad_norm_process_group = None
        self._reduce_grad_norm_across_ranks = True
        scheduler_types = {
            "constant": ConstantWarmupLR,
            "cosine": CosineAnnealingWarmupLR,
        }
        if lr_scheduler not in scheduler_types:
            raise ValueError(
                f"unsupported lr_scheduler={lr_scheduler!r}; "
                f"expected one of {sorted(scheduler_types)}"
            )
        self.lr_scheduler_type = lr_scheduler
        schedulers = {
            name: scheduler_types[lr_scheduler](
                optimizer,
                total_steps=total_steps,
                warmup_steps=int(warmup_ratio * total_steps),
            )
            for name, optimizer in self._optimizers.items()
        }
        self.scheduler = (
            schedulers[ADAMW_OPTIMIZER]
            if optimizer_type == ADAMW_OPTIMIZER
            else _SchedulerCollection(schedulers)
        )

    def _init_muon(
        self,
        *,
        partition: MuonParameterPartition,
        master_by_name: Mapping[str, torch.Tensor],
        adamw_lr: float,
        adamw_weight_decay: float,
        muon_lr: float,
        muon_weight_decay: float,
        muon_momentum: float,
        muon_nesterov: bool,
        muon_ns_steps: int,
        muon_adjust_lr_fn: str,
    ) -> None:
        if not partition.muon:
            raise ValueError(
                "Muon mode found no eligible hidden nn.Linear weight matrices"
            )
        muon_parameters = [master_by_name[item.name] for item in partition.muon]
        adamw_parameters = [master_by_name[item.name] for item in partition.adamw]
        locally_sharded = any(
            tuple(parameter.shape) != tuple(item.logical_shape)
            for parameter, item in zip(muon_parameters, partition.muon)
        )
        muon_kwargs = dict(
            lr=muon_lr,
            weight_decay=muon_weight_decay,
            momentum=muon_momentum,
            nesterov=muon_nesterov,
            ns_steps=muon_ns_steps,
            adjust_lr_fn=muon_adjust_lr_fn,
        )
        self.optimizer = (
            FSDPShardedMuon(
                muon_parameters,
                [item.logical_shape for item in partition.muon],
                **muon_kwargs,
            )
            if locally_sharded
            else torch.optim.Muon(muon_parameters, **muon_kwargs)
        )

        self.aux_optimizer = (
            torch.optim.AdamW(
                adamw_parameters,
                lr=adamw_lr,
                weight_decay=adamw_weight_decay,
                fused=self._adamw_fused,
            )
            if adamw_parameters
            else None
        )
        self._optimizers = {MUON_OPTIMIZER: self.optimizer}
        if self.aux_optimizer is not None:
            self._optimizers[ADAMW_OPTIMIZER] = self.aux_optimizer
        parameter_layout = {
            item.name: (
                item.name,
                tuple(item.logical_shape),
                tuple(item.parameter.shape),
                name,
            )
            for name, parameters in (
                (MUON_OPTIMIZER, partition.muon),
                (ADAMW_OPTIMIZER, partition.adamw),
            )
            for item in parameters
        }
        # FP32 masters follow model order, which may interleave optimizer groups.
        self._parameter_layout = [parameter_layout[name] for name in master_by_name]

    def configure_grad_norm_reduction(
        self, *, process_group=None, enabled: bool = True
    ) -> None:
        """Configure the group that owns disjoint gradient shards.

        FSDP backends disable the reduction for replicated/NO_SHARD parameters.
        """
        self._grad_norm_process_group = process_group
        self._reduce_grad_norm_across_ranks = enabled
        if isinstance(self.optimizer, FSDPShardedMuon):
            if not enabled:
                raise RuntimeError(
                    "Flattened Muon parameters require sharded gradient reduction"
                )
            self.optimizer.configure_process_group(process_group)

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
        for optimizer in self._optimizers.values():
            optimizer.step()
            optimizer.zero_grad()
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

    def _restore_adamw_implementation(self, optimizer) -> None:
        """Re-apply this run's AdamW kernel choice after a checkpoint load.

        ``Optimizer.load_state_dict`` adopts the saved param-group flags, so a
        checkpoint from an unfused run would silently disable ``fused`` here
        (and a fused one would enable it for CPU masters). Fused AdamW reads
        its step counters on the parameter device; unfused keeps them on CPU.
        """
        for group in optimizer.param_groups:
            group["fused"] = self._adamw_fused
            if self._adamw_fused:
                group["foreach"] = None
        for param, state in optimizer.state.items():
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
        checkpoint_type = state_dict.get("optimizer_type", ADAMW_OPTIMIZER)
        if self.optimizer_type == ADAMW_OPTIMIZER:
            if checkpoint_type != ADAMW_OPTIMIZER:
                raise ValueError("Cannot load a Muon optimizer state into AdamW")
            self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        else:
            self._load_hybrid_optimizer_state(state_dict)
        if ADAMW_OPTIMIZER in self._optimizers:
            self._restore_adamw_implementation(self._optimizers[ADAMW_OPTIMIZER])
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
                for p, mp in zip(self.model_params, self.fp32_params):
                    mp.data.copy_(p.detach().to(device=mp.device, dtype=mp.dtype))

    def _load_hybrid_optimizer_state(self, state_dict: dict) -> None:
        if state_dict.get("optimizer_type") != MUON_OPTIMIZER:
            raise ValueError("Cannot load a non-Muon optimizer state into Muon")
        optimizer_state = state_dict.get("optimizer_state_dict")
        if not isinstance(optimizer_state, dict):
            raise ValueError("Muon checkpoint is missing 'optimizer_state_dict'")
        if optimizer_state.get("format_version") != self._HYBRID_STATE_FORMAT_VERSION:
            raise ValueError(
                "Unsupported hybrid optimizer state format: "
                f"{optimizer_state.get('format_version')!r}"
            )

        if optimizer_state.get("parameter_layout") != self._parameter_layout:
            raise ValueError(
                "Muon parameter layout differs from the checkpoint; "
                "parameter order, shapes, and optimizer assignment must match"
            )

        saved_optimizers = optimizer_state.get("optimizers")
        if not isinstance(saved_optimizers, dict):
            raise ValueError("Muon checkpoint is missing optimizer group states")
        if set(saved_optimizers) != set(self._optimizers):
            raise ValueError(
                "Muon optimizer groups do not match: "
                f"expected={sorted(self._optimizers)}, "
                f"received={sorted(saved_optimizers)}"
            )
        for name, optimizer in self._optimizers.items():
            optimizer.load_state_dict(saved_optimizers[name])

    def state_dict(self):
        common_state = {
            "scheduler_state_dict": self.scheduler.state_dict(),
            "lr_scheduler_type": self.lr_scheduler_type,
            "max_grad_norm": self.max_grad_norm,
            "fp32_params": [tensor.detach().cpu() for tensor in self.fp32_params],
        }
        if self.optimizer_type == ADAMW_OPTIMIZER:
            return {
                "optimizer_state_dict": self.optimizer.state_dict(),
                **common_state,
            }
        return {
            "optimizer_type": MUON_OPTIMIZER,
            "optimizer_state_dict": {
                "format_version": self._HYBRID_STATE_FORMAT_VERSION,
                "parameter_layout": self._parameter_layout,
                "optimizers": {
                    name: optimizer.state_dict()
                    for name, optimizer in self._optimizers.items()
                },
            },
            **common_state,
        }

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]

    def get_learning_rates(self) -> dict[str, float]:
        return {
            name: float(optimizer.param_groups[0]["lr"])
            for name, optimizer in self._optimizers.items()
        }
