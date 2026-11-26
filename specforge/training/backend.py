# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""TrainingBackend: model wrapping / backward / optimizer step / state dict.

FSDP-only for now. ``ParallelConfig`` carries the sharding strategy and the
process group the job's topology gives the wrapper.
"""

from __future__ import annotations

import abc
import contextlib
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import torch
import torch.distributed as dist
import torch.nn as nn

from specforge.training.contracts import OptimizerSpec, RankFailure, SchedulerSpec

if TYPE_CHECKING:
    from specforge.distributed import Topology


def _optimizer_params(optimizer: torch.optim.Optimizer) -> List[nn.Parameter]:
    """Parameters in the exact flat index order ``optimizer.state_dict()`` uses."""
    return [param for group in optimizer.param_groups for param in group["params"]]


def _foreach_scale_(tensors: List[torch.Tensor], factor: torch.Tensor) -> None:
    """``tensor.mul_(factor)`` for every tensor, one fused launch per group.

    A same-device CUDA 0-dim factor is cast to each tensor's dtype by the
    per-tensor ``mul_``; casting it once up front keeps ``_foreach_mul_`` on
    its fused path (which requires matching dtypes) and bitwise identical.
    """
    groups: Dict[Tuple[torch.device, torch.dtype], List[torch.Tensor]] = {}
    for tensor in tensors:
        groups.setdefault((tensor.device, tensor.dtype), []).append(tensor)
    for (device, dtype), group in groups.items():
        scalar = factor
        if (
            isinstance(factor, torch.Tensor)
            and device.type == "cuda"
            and factor.device == device
        ):
            scalar = factor.to(dtype)
        torch._foreach_mul_(group, scalar)


@dataclass
class ParallelConfig:
    """How the backend wraps the program, and over which process group.

    The group comes from the job's topology, built once by
    :func:`specforge.distributed.init_distributed`. Re-deriving groups here can
    change collective ordering and deadlock a multi-rank run.
    """

    world_size: int = 1
    sharding_strategy: str = "SHARD_GRAD_OP"
    param_dtype: torch.dtype = torch.bfloat16
    fsdp_process_group: Any = None
    ddp_bucket_cap_mb: Optional[float] = None

    @classmethod
    def from_topology(
        cls,
        topology: Topology,
        *,
        sharding_strategy: str = "SHARD_GRAD_OP",
        param_dtype: torch.dtype = torch.bfloat16,
        ddp_bucket_cap_mb: Optional[float] = None,
    ) -> "ParallelConfig":
        """Wrap over the topology's world group."""
        return cls(
            world_size=topology.world_size,
            sharding_strategy=sharding_strategy,
            param_dtype=param_dtype,
            fsdp_process_group=topology.world_group,
            ddp_bucket_cap_mb=ddp_bucket_cap_mb,
        )


class TrainingBackend(abc.ABC):
    name: str
    #: How backward reduces gradients across ``process_group``: ``"mean"``
    #: (DDP/FSDP1/FSDP2 average) or ``"sum"`` (gradient-summing runtimes).
    #: The step runner cancels ``"mean"`` by multiplying by world size.
    grad_reduction: Literal["mean", "sum"] = "mean"
    #: Whether ``step(loss_denominator=...)`` validates the global loss
    #: denominator inside the optimizer's single host read, letting the step
    #: runner skip its own synchronizing check.
    checks_loss_denominator: bool = False

    @property
    @abc.abstractmethod
    def process_group(self) -> Any:
        """The group whose ranks share an update; None is the default group.

        The loss denominator, the world-size rescale and the metrics are
        reduced over it, so it has to be the group the gradients synchronize
        over. It has no default: a wrong group trains on wrong gradients.
        """

    #: Whether every rank owns the same complete optimizer state, which a
    #: checkpoint then saves once.
    optimizer_state_is_replicated: bool = False

    def learning_rate(self) -> Any:
        """The optimizer's current rate, or None when it does not report one."""
        return None

    def close(self) -> None:
        """Release runtime-owned resources (e.g. auxiliary process groups).

        Must be idempotent: callers invoke it from ``finally`` blocks.
        """

    def accumulation_context(self, *, is_boundary: bool = True):
        return contextlib.nullcontext()

    @abc.abstractmethod
    def prepare_model(
        self,
        model: nn.Module,
        *,
        wrap: bool = True,
        optimizer_target: Optional[nn.Module] = None,
        frozen_modules: Tuple[nn.Module, ...] = (),
        blocks: Tuple[str, ...] = (),
    ) -> nn.Module:
        """Register the program and wrap it for data-parallel training.

        Args:
            model: The program.
            wrap: Whether to wrap it for data-parallel training.
            optimizer_target: The module whose parameters the optimizer takes;
                the wrapped module when omitted.
            frozen_modules: Frozen modules of ``model`` that stay replicated.
                They are target tables used only inside the loss: sharding them
                would all-gather them before every optimizer window without
                saving optimizer memory.
            blocks: Class names of the modules a sharding backend wraps as one
                unit each, which overlaps their all-gather and reduce-scatter
                with the decoder's compute. Empty: one root unit.

        Returns:
            The module the runner calls.
        """

    @abc.abstractmethod
    def backward(self, loss: torch.Tensor, *, is_boundary: bool = True) -> None: ...

    def scale_gradients(self, factor: torch.Tensor) -> None:
        """Scale synchronized gradients before clipping and stepping."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement gradient scaling"
        )

    @abc.abstractmethod
    def step(self) -> Optional[torch.Tensor]: ...

    @abc.abstractmethod
    def state_dict(self) -> dict: ...

    @abc.abstractmethod
    def load_state_dict(self, state: dict) -> None: ...


class FSDPTrainingBackend(TrainingBackend):
    """FSDP1 backend for the canonical SpecForge training math: FSDP with
    ``use_orig_params=True`` / bf16 mixed precision over the configured process
    group, optimizer targeting the inner trainable submodule."""

    name = "fsdp"

    def __init__(
        self,
        parallel_config: ParallelConfig,
        *,
        optimizer_factory=None,
        optimizer_spec: Optional[OptimizerSpec] = None,
        scheduler_spec: Optional[SchedulerSpec] = None,
        max_grad_norm: Optional[float] = None,
    ) -> None:
        if optimizer_factory is not None and optimizer_spec is not None:
            raise ValueError(
                "pass either optimizer_factory or optimizer_spec, not both"
            )
        self.parallel_config = parallel_config
        self._optimizer_factory = optimizer_factory
        self._optimizer_spec = optimizer_spec
        self._scheduler_spec = scheduler_spec
        self._max_grad_norm = max_grad_norm
        self.module: Optional[nn.Module] = None
        self.optimizer = None
        self.scheduler = None
        self.last_grad_norm: Optional[torch.Tensor] = None
        self._wrapped = False
        self._wrapper_kind = "none"
        self.auto_wrap_block_classes = set()
        self.ignored_frozen_modules = tuple()

    @property
    def process_group(self) -> Any:
        return self.parallel_config.fsdp_process_group

    def learning_rate(self) -> Any:
        if self.optimizer is None:
            return None
        get_learning_rate = getattr(self.optimizer, "get_learning_rate", None)
        if callable(get_learning_rate):
            return get_learning_rate()
        if (
            isinstance(self.optimizer, torch.optim.Optimizer)
            and self.optimizer.param_groups
        ):
            return self.optimizer.param_groups[0]["lr"]
        return None

    @property
    def optimizer_state_is_replicated(self) -> bool:
        """Whether every rank owns the same complete optimizer state."""
        return self._wrapper_kind == "ddp"

    def prepare_model(
        self,
        model: nn.Module,
        *,
        wrap: bool = True,
        optimizer_target: Optional[nn.Module] = None,
        frozen_modules: Tuple[nn.Module, ...] = (),
        blocks: Tuple[str, ...] = (),
    ) -> nn.Module:
        """Register and wrap the trainable module unless ``wrap=False``.

        Replicated ``NO_SHARD`` recipes use DDP; sharded recipes use FSDP.

        Raises:
            ValueError: A module in ``frozen_modules`` holds a trainable
                parameter, or ``blocks`` names no module class of ``model``.
        """
        if any(
            parameter.requires_grad
            for module in frozen_modules
            for parameter in module.parameters()
        ):
            raise ValueError("frozen modules must contain only frozen parameters")
        # Record pre-sharding parameter ranks by object identity before FSDP1
        # flattens shards to 1-D (and inserts ``_fsdp_wrapped_module`` into names).
        pre_wrap_target = optimizer_target if optimizer_target is not None else model
        orig_ndims = {
            id(p): p.ndim for p in pre_wrap_target.parameters() if p.requires_grad
        }
        if not wrap:
            self.module = model
            self._wrapped = False
            self._wrapper_kind = "none"
        else:
            import functools

            pc = self.parallel_config
            block_classes = {
                type(module)
                for module in model.modules()
                if type(module).__name__ in blocks
            }
            # A draft class may name a block only some of its models hold (a
            # one-layer P-EAGLE has the first stage and no decoder layer), so
            # a name without a module is not an error. Names that match
            # nothing are: the model would train as one root unit.
            if blocks and not block_classes:
                raise ValueError(
                    f"blocks {sorted(blocks)} name no module class of the model"
                )
            if pc.sharding_strategy == "NO_SHARD":
                # PyTorch deprecated FSDP's NO_SHARD mode in favor of DDP.
                # DDP gives this small draft model replicated-param execution
                # without per-block parameter all-gathers.
                from torch.nn.parallel import DistributedDataParallel as DDP

                device_ids = None
                output_device = None
                model_device = next(model.parameters()).device
                if model_device.type in ("cuda", "npu"):
                    device_ids = [model_device.index]
                    output_device = model_device.index
                model = DDP(
                    model,
                    device_ids=device_ids,
                    output_device=output_device,
                    process_group=pc.fsdp_process_group,
                    broadcast_buffers=False,
                    gradient_as_bucket_view=True,
                    bucket_cap_mb=pc.ddp_bucket_cap_mb,
                )
                self._wrapper_kind = "ddp"
            else:
                from torch.distributed.fsdp import BackwardPrefetch
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
                from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

                sharding = getattr(ShardingStrategy, pc.sharding_strategy)
                fsdp_kwargs = dict(
                    use_orig_params=True,
                    mixed_precision=MixedPrecision(
                        param_dtype=getattr(model, "fsdp_param_dtype", pc.param_dtype),
                        buffer_dtype=torch.float32,
                        # The program casts what it computes with; the wrapper
                        # leaves its inputs as the runner placed them, as DDP does.
                        cast_root_forward_inputs=False,
                    ),
                    sharding_strategy=sharding,
                    process_group=pc.fsdp_process_group,
                )
                if frozen_modules:
                    fsdp_kwargs["ignored_modules"] = frozen_modules
                if block_classes:
                    fsdp_kwargs.update(
                        auto_wrap_policy=functools.partial(
                            transformer_auto_wrap_policy,
                            transformer_layer_cls=block_classes,
                        ),
                        forward_prefetch=True,
                        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                        limit_all_gathers=True,
                    )
                model = FSDP(model, **fsdp_kwargs)
                self._wrapper_kind = "fsdp"
            self.module = model
            self._wrapped = True
            # What the sharding wrapper was given; DDP takes neither.
            sharded = self._wrapper_kind == "fsdp"
            self.auto_wrap_block_classes = block_classes if sharded else set()
            self.ignored_frozen_modules = frozen_modules if sharded else ()
        target = optimizer_target if optimizer_target is not None else self.module
        if self._optimizer_spec is not None:
            self._build_configured_optimizer(target, param_ndims=orig_ndims)
        elif self._optimizer_factory is not None:
            self.set_optimizer(self._optimizer_factory(target))
        return self.module

    def _build_configured_optimizer(
        self,
        target: nn.Module,
        *,
        param_ndims: Optional[Mapping[int, int]] = None,
    ) -> None:
        from specforge.optimizer import BF16Optimizer

        if self._scheduler_spec is None:
            raise ValueError("optimizer_spec requires scheduler_spec")
        if self._max_grad_norm is None:
            raise ValueError("optimizer_spec requires max_grad_norm")
        self.set_optimizer(
            BF16Optimizer(
                target,
                optimizer_spec=self._optimizer_spec,
                scheduler_spec=self._scheduler_spec,
                max_grad_norm=self._max_grad_norm,
                param_ndims=param_ndims,
            )
        )

    def set_optimizer(self, optimizer, *, scheduler=None) -> None:
        self.optimizer = optimizer
        self.scheduler = scheduler
        self._configure_optimizer_grad_norm()

    def _configure_optimizer_grad_norm(self) -> None:
        configure = getattr(self.optimizer, "configure_grad_norm_reduction", None)
        if configure is not None:
            configure(
                process_group=self.parallel_config.fsdp_process_group,
                enabled=(
                    self._wrapped
                    and self.parallel_config.sharding_strategy != "NO_SHARD"
                ),
            )

    def backward(self, loss: torch.Tensor, *, is_boundary: bool = True) -> None:
        """Backward one micro-step with one gradient collective per window.

        FSDP defers reduction here; DDP requires ``accumulation_context``
        around both forward and backward to defer its reducer.
        """
        if is_boundary or not self._wrapped or self._wrapper_kind == "ddp":
            loss.backward()
        else:
            with self.module.no_sync():
                loss.backward()

    def accumulation_context(self, *, is_boundary: bool = True):
        if self._wrapper_kind == "ddp" and not is_boundary:
            return self.module.no_sync()
        return contextlib.nullcontext()

    def scale_gradients(self, factor: torch.Tensor) -> None:
        if self.module is None:
            raise RuntimeError("scale_gradients called before prepare_model")
        with torch.no_grad():
            _foreach_scale_(
                [p.grad for p in self.module.parameters() if p.grad is not None],
                factor,
            )

    @property
    def _runtime_owns_step(self) -> bool:
        """Whether this runtime owns grad norm, clipping, checks and scheduling."""
        return (
            isinstance(self.optimizer, torch.optim.Optimizer)
            and self._max_grad_norm is not None
        )

    @property
    def checks_loss_denominator(self) -> bool:
        if self._runtime_owns_step:
            return True
        return bool(getattr(self.optimizer, "checks_loss_denominator", False))

    def _global_grad_norm(self, params: Sequence[nn.Parameter]) -> torch.Tensor:
        """Compute the pre-clip global L2 gradient norm over ``params``."""
        from torch.distributed.tensor import DTensor

        grads = [p.grad for p in params if p.grad is not None]
        if not grads:
            device = params[0].device if params else "cpu"
            return torch.zeros((), dtype=torch.float32, device=device)
        if any(isinstance(grad, DTensor) for grad in grads):
            norm = torch.nn.utils.get_total_norm(grads, norm_type=2.0)
            return norm.full_tensor() if isinstance(norm, DTensor) else norm
        from specforge.optimizer import _sum_of_squares

        total_norm_sq = _sum_of_squares([grad.detach() for grad in grads])
        if (
            self._wrapped
            and self.parallel_config.sharding_strategy != "NO_SHARD"
            and dist.is_available()
            and dist.is_initialized()
        ):
            dist.all_reduce(
                total_norm_sq, op=dist.ReduceOp.SUM, group=self.process_group
            )
        return total_norm_sq.sqrt()

    def _step_torch_optimizer(
        self,
        params: Sequence[nn.Parameter],
        *,
        loss_denominator: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the runtime-owned update for a direct ``torch.optim.Optimizer``.

        Computes global grad norm, validates it (and optional
        ``loss_denominator``) in one host read before mutating any state,
        clips gradients, steps the optimizer, clears gradients, and advances
        the LR scheduler when present.
        """
        from specforge.optimizer import validate_step_scalars

        assert self._max_grad_norm is not None
        grad_norm = self._global_grad_norm(params)

        def clear_grads() -> None:
            self.optimizer.zero_grad(set_to_none=True)

        try:
            validate_step_scalars(
                grad_norm,
                max_grad_norm=self._max_grad_norm,
                loss_denominator=loss_denominator,
                on_error=clear_grads,
            )
        except FloatingPointError:
            self.last_grad_norm = grad_norm.detach()
            raise
        torch.nn.utils.clip_grads_with_norm_(params, self._max_grad_norm, grad_norm)
        self.last_grad_norm = grad_norm.detach()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self.scheduler is not None:
            self.scheduler.step()
        return self.last_grad_norm

    def step(
        self, *, loss_denominator: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """Run the optimizer step, which clips and returns the global grad norm.

        ``loss_denominator`` is forwarded only when set; it requires an
        optimizer or runtime-owned update that ``checks_loss_denominator``.
        """
        if self.optimizer is None:
            raise RuntimeError(
                f"{type(self).__name__}.step called before optimizer is set"
            )
        if self._runtime_owns_step:
            return self._step_torch_optimizer(
                _optimizer_params(self.optimizer),
                loss_denominator=loss_denominator,
            )
        if loss_denominator is None:
            return self.optimizer.step()
        return self.optimizer.step(loss_denominator=loss_denominator)

    def _torch_optimizer_state_dict(
        self, *, include_params: bool = False, format_key: Optional[str] = None
    ) -> dict:
        """Serialize a runtime-owned ``torch.optim.Optimizer`` + optional scheduler."""
        from specforge.optimizer import local_tensor

        optimizer_state = self.optimizer.state_dict()
        optimizer_state["state"] = {
            index: {
                key: (
                    local_tensor(value).detach().cpu().clone()
                    if torch.is_tensor(value)
                    else value
                )
                for key, value in state.items()
            }
            for index, state in optimizer_state["state"].items()
        }
        payload: Dict[str, Any] = {
            "optimizer_state_dict": optimizer_state,
            "max_grad_norm": self._max_grad_norm,
        }
        if self.scheduler is not None:
            payload["scheduler_state_dict"] = self.scheduler.state_dict()
            payload["lr_scheduler_type"] = (
                self._scheduler_spec.kind if self._scheduler_spec is not None else None
            )
        if include_params:
            payload["fp32_params"] = [
                local_tensor(param.detach()).cpu().clone()
                for param in _optimizer_params(self.optimizer)
            ]
        if format_key is not None:
            payload[format_key] = True
        return payload

    def _load_torch_optimizer_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *,
        restore_params: bool = False,
        format_key: Optional[str] = None,
    ) -> None:
        """Restore a runtime-owned ``torch.optim.Optimizer`` + optional scheduler."""
        from specforge.optimizer import (
            like_parameter,
            local_tensor,
            restore_optimizer_kernel_state,
        )

        if format_key is not None and format_key not in state_dict:
            raise ValueError(
                f"this optimizer state was not written by the {self.name} runtime; "
                "resume it with the runtime that wrote it"
            )
        if self.scheduler is not None and self._scheduler_spec is not None:
            saved_scheduler_type = state_dict.get("lr_scheduler_type", "cosine")
            if saved_scheduler_type != self._scheduler_spec.kind:
                raise ValueError(
                    "checkpoint optimizer used lr_scheduler="
                    f"{saved_scheduler_type!r} but this run has "
                    f"lr_scheduler={self._scheduler_spec.kind!r}"
                )
        saved_max_grad_norm = state_dict.get("max_grad_norm")
        if (
            self._max_grad_norm is not None
            and saved_max_grad_norm is not None
            and float(saved_max_grad_norm) != float(self._max_grad_norm)
        ):
            raise ValueError(
                "checkpoint optimizer used max_grad_norm="
                f"{saved_max_grad_norm} but this run has "
                f"max_grad_norm={self._max_grad_norm}"
            )
        params = _optimizer_params(self.optimizer)
        if restore_params:
            saved = state_dict["fp32_params"]
            if len(saved) != len(params):
                raise ValueError(
                    f"checkpoint carries {len(saved)} fp32 parameters but this rank "
                    f"has {len(params)}"
                )
            with torch.no_grad():
                for index, (value, param) in enumerate(zip(saved, params)):
                    local = local_tensor(param)
                    if value.shape != local.shape:
                        raise ValueError(
                            f"fp32 parameter {index} shape mismatch: checkpoint "
                            f"{tuple(value.shape)} vs current {tuple(local.shape)}"
                        )
                    local.copy_(value.to(local.device, local.dtype))
        optimizer_state = dict(state_dict["optimizer_state_dict"])
        optimizer_state["state"] = {
            index: {
                key: like_parameter(
                    value.clone() if torch.is_tensor(value) else value,
                    params[int(index)],
                )
                for key, value in state.items()
            }
            for index, state in optimizer_state["state"].items()
        }
        self.optimizer.load_state_dict(optimizer_state)
        restore_optimizer_kernel_state(self.optimizer)
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])

    def _optimizer_state_dict(self) -> Any:
        if self.optimizer is None:
            return None
        if self._runtime_owns_step:
            return self._torch_optimizer_state_dict()
        return self.optimizer.state_dict()

    def _load_optimizer_state_dict(self, state: Mapping[str, Any]) -> None:
        if self._runtime_owns_step:
            self._load_torch_optimizer_state_dict(state)
            return
        self.optimizer.load_state_dict(state)

    def state_dict(self) -> dict:
        """Full training state ``{"model", "optimizer", "rng"}`` for resume.

        ``model`` is gathered rank0-only (``{}`` on other ranks when wrapped).
        FSDP optimizer state is rank-local; DDP optimizer state is replicated.
        RNG state is always rank-local.
        """
        if self.module is None:
            raise RuntimeError("state_dict called before prepare_model")
        # Gathering a sharded model is a collective: a rank that fails inside
        # it is already apart from the others.
        model = self._module_state_dict()
        try:
            # This rank's own work, once the gather is over.
            optimizer = self._optimizer_state_dict()
            rng = self._rng_state()
        except Exception as error:
            raise RankFailure(f"{type(error).__name__}: {error}") from error
        return {"model": model, "optimizer": optimizer, "rng": rng}

    def load_state_dict(self, state: dict) -> None:
        """Restore whichever of module weights / optimizer / RNG the state carries."""
        if state.get("model") is not None:
            self._load_module_state_dict(state["model"])
        try:
            # This rank's own state: its failure is this rank's alone.
            if self.optimizer is not None and state.get("optimizer") is not None:
                self._load_optimizer_state_dict(state["optimizer"])
            if state.get("rng") is not None:
                self._set_rng_state(state["rng"])
        except Exception as error:
            raise RankFailure(f"{type(error).__name__}: {error}") from error

    def _full_state_ctx(self, state_dict_config=None):
        """FULL_STATE_DICT context for a wrapped module; a no-op when unwrapped."""
        if self._wrapper_kind != "fsdp":
            return contextlib.nullcontext()
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import StateDictType

        return FSDP.state_dict_type(
            self.module, StateDictType.FULL_STATE_DICT, state_dict_config
        )

    def _module_state_dict(self) -> dict:
        if self._wrapper_kind == "ddp":
            if dist.is_initialized() and dist.get_rank() != 0:
                return {}
            return self.module.module.state_dict()
        if self._wrapper_kind != "fsdp":
            return self.module.state_dict()
        from torch.distributed.fsdp import FullStateDictConfig

        # gather to rank0 CPU only — materializing the full model on every
        # rank's GPU is wasted memory when only rank0 writes it.
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with self._full_state_ctx(cfg):
            return self.module.state_dict()

    def _load_module_state_dict(self, model_state: dict) -> None:
        # every rank loads the full state dict read from the shared file.
        if self._wrapper_kind == "ddp":
            self.module.module.load_state_dict(model_state)
            return
        with self._full_state_ctx():
            self.module.load_state_dict(model_state)

    @staticmethod
    def _rng_state() -> dict:
        # Persist only the bound accelerator's RNG state so a checkpoint is
        # independent of how many CUDA/NPU devices are visible at load time.
        from specforge.utils import get_device_type

        device_type = get_device_type()
        accelerator_state = None
        module = getattr(torch, device_type, None)
        if (
            device_type in ("cuda", "npu")
            and module is not None
            and module.is_available()
        ):
            accelerator_state = module.get_rng_state(module.current_device())
        return {
            "torch": torch.get_rng_state(),
            "device_type": device_type,
            "cuda": accelerator_state if device_type == "cuda" else None,
            "npu": accelerator_state if device_type == "npu" else None,
        }

    @staticmethod
    def _set_rng_state(rng: dict) -> None:
        if "device_type" not in rng:
            raise ValueError("Checkpoint RNG state does not record device_type")
        device_type = rng["device_type"]
        torch.set_rng_state(rng["torch"])
        state = rng.get(device_type) if device_type is not None else None
        module = getattr(torch, device_type, None) if device_type is not None else None
        if state is None or module is None or not module.is_available():
            return
        module.set_rng_state(state, module.current_device())


__all__ = ["ParallelConfig", "TrainingBackend", "FSDPTrainingBackend"]
