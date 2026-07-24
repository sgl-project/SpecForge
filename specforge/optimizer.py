from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.distributed as dist

from specforge.lr_scheduler import CosineAnnealingWarmupLR
from specforge.muon import (
    ADAMW_OPTIMIZER,
    DEFAULT_MUON_EXCLUDED_MODULES,
    MUON_OPTIMIZER,
    SUPPORTED_OPTIMIZERS,
    FSDPShardedMuon,
    MuonParameterMetadata,
    NamedTrainableParameter,
    partition_parameters_for_muon,
)
from specforge.utils import print_on_rank0

logger = logging.getLogger(__name__)


@dataclass
class _SchedulerCollection:
    """Expose multiple schedulers through the existing scheduler interface."""

    schedulers: Mapping[str, CosineAnnealingWarmupLR]

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
    """FP32-master AdamW optimizer with opt-in hybrid Muon support.

    The default AdamW path preserves the historical state format and training
    behavior. Muon mode applies Muon to hidden linear matrices and an auxiliary
    AdamW to embeddings, heads, normalization parameters, and biases.
    """

    _HYBRID_STATE_FORMAT_VERSION = 1

    def __init__(
        self,
        model,
        lr,
        weight_decay=0.0,
        max_grad_norm=0.5,
        total_steps=800_000,
        warmup_ratio=0.015,
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
            raise ValueError(
                "Muon does not support optimizer CPU offload because its "
                "Newton-Schulz step must run on the accelerator"
            )

        self.model = model
        self.optimizer_type = optimizer_type
        self.max_grad_norm = max_grad_norm
        self.offload_master = bool(offload_master)
        self.last_grad_norm = None
        self._grad_norm_process_group = None
        self._reduce_grad_norm_across_ranks = True

        logical_shapes = (
            muon_metadata.logical_shapes_by_id if muon_metadata is not None else {}
        )
        named_parameters = tuple(
            NamedTrainableParameter(
                name=name,
                parameter=parameter,
                logical_shape=torch.Size(
                    logical_shapes.get(id(parameter), parameter.shape)
                ),
            )
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        if not named_parameters:
            raise ValueError(
                "Cannot construct an optimizer with no trainable parameters"
            )

        self._model_param_names = tuple(item.name for item in named_parameters)
        self._logical_shapes_by_name = {
            item.name: item.logical_shape for item in named_parameters
        }
        self.model_params = [item.parameter for item in named_parameters]
        self.fp32_params = [
            self._new_master_parameter(item.parameter) for item in named_parameters
        ]
        master_by_name = {
            item.name: master
            for item, master in zip(named_parameters, self.fp32_params)
        }

        if optimizer_type == ADAMW_OPTIMIZER:
            self._init_adamw(
                named_parameters=named_parameters,
                lr=lr,
                weight_decay=weight_decay,
                total_steps=total_steps,
                warmup_ratio=warmup_ratio,
            )
        else:
            partition = partition_parameters_for_muon(
                model,
                excluded_module_names=muon_excluded_module_names,
                metadata=muon_metadata,
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
                total_steps=total_steps,
                warmup_ratio=warmup_ratio,
            )

    def _new_master_parameter(self, parameter: torch.Tensor) -> torch.Tensor:
        master = (
            parameter.detach().to(device="cpu", dtype=torch.float32).clone()
            if self.offload_master
            else parameter.detach().clone().to(torch.float32)
        )
        master.requires_grad = True
        return master

    def _init_adamw(
        self,
        *,
        named_parameters: Sequence[NamedTrainableParameter],
        lr: float,
        weight_decay: float,
        total_steps: int,
        warmup_ratio: float,
    ) -> None:
        self.optimizer = torch.optim.AdamW(
            self.fp32_params, lr=lr, weight_decay=weight_decay
        )
        self.aux_optimizer = None
        self._optimizers = {ADAMW_OPTIMIZER: self.optimizer}
        self._parameter_group_names = {
            ADAMW_OPTIMIZER: tuple(item.name for item in named_parameters)
        }
        self.scheduler = CosineAnnealingWarmupLR(
            self.optimizer,
            total_steps=total_steps,
            warmup_steps=int(warmup_ratio * total_steps),
        )

    def _init_muon(
        self,
        *,
        partition,
        master_by_name: Mapping[str, torch.Tensor],
        adamw_lr: float,
        adamw_weight_decay: float,
        muon_lr: float,
        muon_weight_decay: float,
        muon_momentum: float,
        muon_nesterov: bool,
        muon_ns_steps: int,
        muon_adjust_lr_fn: str,
        total_steps: int,
        warmup_ratio: float,
    ) -> None:
        if not partition.muon:
            raise ValueError(
                "Muon mode found no eligible hidden nn.Linear weight matrices"
            )
        muon_class = getattr(torch.optim, "Muon", None)
        if muon_class is None:
            raise RuntimeError(
                "optimizer_type='muon' requires torch.optim.Muon (PyTorch >= 2.9)"
            )

        muon_parameters = [master_by_name[item.name] for item in partition.muon]
        adamw_parameters = [master_by_name[item.name] for item in partition.adamw]
        locally_sharded = any(
            tuple(parameter.shape) != tuple(item.logical_shape)
            for parameter, item in zip(muon_parameters, partition.muon)
        )
        if locally_sharded:
            self.optimizer = FSDPShardedMuon(
                muon_parameters,
                [item.logical_shape for item in partition.muon],
                lr=muon_lr,
                weight_decay=muon_weight_decay,
                momentum=muon_momentum,
                nesterov=muon_nesterov,
                ns_steps=muon_ns_steps,
                adjust_lr_fn=muon_adjust_lr_fn,
            )
        else:
            self.optimizer = muon_class(
                muon_parameters,
                lr=muon_lr,
                weight_decay=muon_weight_decay,
                momentum=muon_momentum,
                nesterov=muon_nesterov,
                ns_steps=muon_ns_steps,
                adjust_lr_fn=muon_adjust_lr_fn,
            )

        self.aux_optimizer = (
            torch.optim.AdamW(
                adamw_parameters,
                lr=adamw_lr,
                weight_decay=adamw_weight_decay,
            )
            if adamw_parameters
            else None
        )
        self._optimizers = {MUON_OPTIMIZER: self.optimizer}
        if self.aux_optimizer is not None:
            self._optimizers[ADAMW_OPTIMIZER] = self.aux_optimizer
        self._parameter_group_names = partition.names()
        self.scheduler = _SchedulerCollection(
            {
                name: CosineAnnealingWarmupLR(
                    optimizer,
                    total_steps=total_steps,
                    warmup_steps=int(warmup_ratio * total_steps),
                )
                for name, optimizer in self._optimizers.items()
            }
        )

    def configure_grad_norm_reduction(
        self, *, process_group=None, enabled: bool = True
    ) -> None:
        """Configure collectives for the group that owns parameter shards."""

        self._grad_norm_process_group = process_group
        self._reduce_grad_norm_across_ranks = enabled
        if isinstance(self.optimizer, FSDPShardedMuon):
            if not enabled:
                raise RuntimeError(
                    "Flattened Muon parameters require sharded gradient reduction"
                )
            self.optimizer.configure_process_group(process_group)

    def _reduce_grad_norm(self, total_norm_sq):
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
        grads = [p.grad.detach() for p in self.model_params if p.grad is not None]
        if grads:
            total_norm_sq = torch.stack(
                [grad.float().square().sum() for grad in grads]
            ).sum()
        else:
            device = self.model_params[0].device if self.model_params else "cpu"
            total_norm_sq = torch.zeros((), dtype=torch.float32, device=device)
        return self._reduce_grad_norm(total_norm_sq)

    def _clip_grad_norm(self):
        """Clip populated master gradients; retained for custom loops/tests."""

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

    def step(self):
        grad_norm, clip_coefficient = self._grad_norm_and_clip_coefficient()
        cpu_clip_coefficient = (
            float(clip_coefficient.item()) if self.offload_master else None
        )
        with torch.no_grad():
            for name, parameter, master in zip(
                self._model_param_names, self.model_params, self.fp32_params
            ):
                if parameter.grad is None:
                    master.grad = None
                    continue
                if parameter.grad.shape != master.shape:
                    raise RuntimeError(
                        "Optimizer gradient shape changed after distributed "
                        f"wrapping for {name!r}: gradient={tuple(parameter.grad.shape)}, "
                        f"master={tuple(master.shape)}"
                    )
                master_grad = parameter.grad.detach().to(
                    device=master.device, dtype=torch.float32
                )
                master_grad.mul_(
                    cpu_clip_coefficient
                    if cpu_clip_coefficient is not None
                    else clip_coefficient
                )
                master.grad = master_grad

        self.last_grad_norm = grad_norm.detach()
        for optimizer in self._optimizers.values():
            optimizer.step()
            optimizer.zero_grad()
        self.scheduler.step()

        with torch.no_grad():
            for parameter, master in zip(self.model_params, self.fp32_params):
                parameter.data.copy_(
                    master.data.to(device=parameter.device, dtype=parameter.dtype)
                )
                parameter.grad = None
        return self.last_grad_norm

    def _validate_max_grad_norm(self, state_dict: dict) -> None:
        saved_max_grad_norm = state_dict.get("max_grad_norm")
        if saved_max_grad_norm is not None and float(saved_max_grad_norm) != float(
            self.max_grad_norm
        ):
            raise ValueError(
                "checkpoint optimizer used max_grad_norm="
                f"{saved_max_grad_norm} but this run has "
                f"max_grad_norm={self.max_grad_norm}"
            )

    def _restore_fp32_params(self, state_dict: dict) -> None:
        saved_fp32 = state_dict.get("fp32_params")
        if saved_fp32 is None:
            logger.warning(
                "checkpoint has no fp32_params; re-cloning master params from "
                "bf16 weights — resume will not be numerically faithful"
            )
            saved_fp32 = [parameter.detach() for parameter in self.model_params]
        if len(saved_fp32) != len(self.fp32_params):
            raise ValueError(
                f"checkpoint carries {len(saved_fp32)} fp32 master params "
                f"but this rank has {len(self.fp32_params)}"
            )
        with torch.no_grad():
            for index, (saved, master) in enumerate(zip(saved_fp32, self.fp32_params)):
                if saved.shape != master.shape:
                    raise ValueError(
                        f"fp32 master param {index} shape mismatch: checkpoint "
                        f"{tuple(saved.shape)} vs current {tuple(master.shape)}"
                    )
                master.data.copy_(saved.to(master.device, master.dtype))

    def load_state_dict(self, state_dict):
        self._validate_max_grad_norm(state_dict)
        checkpoint_type = state_dict.get("optimizer_type", ADAMW_OPTIMIZER)
        if self.optimizer_type == ADAMW_OPTIMIZER:
            if checkpoint_type != ADAMW_OPTIMIZER:
                raise ValueError("Cannot load a Muon optimizer state into AdamW")
            self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        else:
            self._load_hybrid_optimizer_state(state_dict)
        print_on_rank0("Successfully loaded optimizer state_dict.")
        self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        print_on_rank0("Successfully loaded scheduler state_dict.")
        self._restore_fp32_params(state_dict)

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

        saved_names = optimizer_state.get("parameter_group_names")
        current_names = {
            name: list(names) for name, names in self._parameter_group_names.items()
        }
        if saved_names != current_names:
            raise ValueError(
                "Muon parameter partition differs from the checkpoint; refusing "
                "an order-dependent optimizer-state load"
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
                "parameter_group_names": {
                    name: list(names)
                    for name, names in self._parameter_group_names.items()
                },
                "optimizers": {
                    name: optimizer.state_dict()
                    for name, optimizer in self._optimizers.items()
                },
            },
            **common_state,
        }

    def get_learning_rate(self):
        primary_name = (
            MUON_OPTIMIZER if self.optimizer_type == MUON_OPTIMIZER else ADAMW_OPTIMIZER
        )
        return self._optimizers[primary_name].param_groups[0]["lr"]

    def get_learning_rates(self) -> dict[str, float]:
        return {
            name: float(optimizer.param_groups[0]["lr"])
            for name, optimizer in self._optimizers.items()
        }

    def get_parameter_group_summary(self) -> dict[str, dict[str, object]]:
        return {
            group_name: {
                "parameter_count": len(names),
                "numel": sum(
                    math.prod(self._logical_shapes_by_name[name]) for name in names
                ),
                "names": names,
            }
            for group_name, names in self._parameter_group_names.items()
        }


__all__ = ["BF16Optimizer"]
