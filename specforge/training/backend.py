# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""TrainingBackend: model wrapping / backward / optimizer step / state dict.

FSDP-only for now. ``ParallelConfig`` carries the process groups created by the
single distributed lifecycle: trainer TP (fixed at one by public builders) plus
draft DP/USP topology.
"""

from __future__ import annotations

import abc
import contextlib
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn


@dataclass
class ParallelConfig:
    """Snapshot of the trainer-TP and draft-DP/SP layout.

    Process-group handles are created once by :func:`init_distributed` and
    carried into the trainer.  Re-deriving groups here can change collective
    ordering and deadlock a multi-rank run.
    """

    world_size: int = 1
    tp_size: int = 1
    sp_ulysses_size: int = 1
    sp_ring_size: int = 1
    sharding_strategy: str = "SHARD_GRAD_OP"
    param_dtype: torch.dtype = torch.bfloat16
    fsdp_process_group: Any = None
    dp_group: Any = None
    draft_dp_group: Any = None
    tp_group: Any = None
    sp_ulysses_group: Any = None
    sp_ring_group: Any = None
    draft_sp_group: Any = None
    device_mesh: Any = None
    tp_device_mesh: Any = None
    extra: dict = field(default_factory=dict)

    @property
    def sp_size(self) -> int:
        return self.sp_ulysses_size * self.sp_ring_size

    @classmethod
    def from_distributed(
        cls,
        *,
        tp_size: int = 1,
        sp_ulysses_size: int = 1,
        sp_ring_size: int = 1,
        sharding_strategy: str = "SHARD_GRAD_OP",
        param_dtype: torch.dtype = torch.bfloat16,
    ) -> "ParallelConfig":
        """Carry every group built by :func:`specforge.distributed.init_distributed`."""
        # Env override for the FSDP sharding strategy — e.g. FSDP_SHARDING=NO_SHARD
        # runs DDP-style (full params replicated, one grad all-reduce, no param
        # all-gather). Default unchanged when the env var is unset.
        sharding_strategy = os.environ.get("FSDP_SHARDING", sharding_strategy)
        if not dist.is_initialized():
            return cls(
                world_size=1,
                tp_size=tp_size,
                sp_ulysses_size=sp_ulysses_size,
                sp_ring_size=sp_ring_size,
                sharding_strategy=sharding_strategy,
                param_dtype=param_dtype,
            )
        handles: Dict[str, Any] = {}
        try:
            from specforge import distributed as sfdist

            for name, getter in (
                ("dp_group", "get_dp_group"),
                ("draft_dp_group", "get_draft_dp_group"),
                ("tp_group", "get_tp_group"),
                ("sp_ulysses_group", "get_sp_ulysses_group"),
                ("sp_ring_group", "get_sp_ring_group"),
                ("draft_sp_group", "get_draft_sp_group"),
                ("device_mesh", "get_device_mesh"),
                ("tp_device_mesh", "get_tp_device_mesh"),
            ):
                fn = getattr(sfdist, getter, None)
                if fn is None:
                    continue
                try:
                    handles[name] = fn()
                except Exception as exc:
                    logging.getLogger(__name__).warning(
                        "ParallelConfig.from_distributed: %s() unavailable: %s",
                        getter,
                        exc,
                    )
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "ParallelConfig.from_distributed: distributed handles unavailable: %s",
                exc,
            )
        return cls(
            world_size=dist.get_world_size(),
            tp_size=tp_size,
            sp_ulysses_size=sp_ulysses_size,
            sp_ring_size=sp_ring_size,
            sharding_strategy=sharding_strategy,
            param_dtype=param_dtype,
            fsdp_process_group=dist.group.WORLD,
            **handles,
        )


class TrainingBackend(abc.ABC):
    name: str

    @abc.abstractmethod
    def prepare_model(self, model: nn.Module) -> nn.Module: ...

    @abc.abstractmethod
    def backward(self, loss: torch.Tensor, *, is_boundary: bool = True) -> None: ...

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
    ) -> None:
        self.parallel_config = parallel_config
        self._optimizer_factory = optimizer_factory
        self.module: Optional[nn.Module] = None
        self.optimizer = None
        self._wrapped = False
        self._wrapper_kind = "none"
        self.auto_wrap_block_classes = set()
        self.ignored_frozen_modules = tuple()

    @property
    def optimizer_state_is_replicated(self) -> bool:
        """Whether every rank owns the same complete optimizer state."""
        return self._wrapper_kind == "ddp"

    @staticmethod
    def _frozen_target_modules(model: nn.Module) -> tuple[nn.Module, ...]:
        """Return frozen target tables that should remain replicated.

        DFlash-family wrappers carry the target embedding and LM head only for
        inference inside the loss.  Sharding those frozen tables makes FSDP
        all-gather them before every optimizer window without saving optimizer
        memory, which is the wrong trade-off for the current trainer recipes.
        """
        modules = []
        for name in ("lm_head", "embed_tokens"):
            module = getattr(model, name, None)
            if not isinstance(module, nn.Module):
                continue
            parameters = tuple(module.parameters())
            if parameters and not any(
                parameter.requires_grad for parameter in parameters
            ):
                modules.append(module)
        return tuple(modules)

    def prepare_model(
        self,
        model: nn.Module,
        *,
        wrap: bool = True,
        optimizer_target: Optional[nn.Module] = None,
    ) -> nn.Module:
        """Register and wrap the trainable module unless ``wrap=False``.

        Replicated ``NO_SHARD`` recipes use DDP; sharded recipes use FSDP.
        """
        if self._optimizer_factory is not None:
            capture_metadata = getattr(
                self._optimizer_factory, "capture_parameter_metadata", None
            )
            if callable(capture_metadata):
                capture_metadata(
                    optimizer_target if optimizer_target is not None else model
                )
        if not wrap:
            self.module = model
            self._wrapped = False
            self._wrapper_kind = "none"
        else:
            import functools

            pc = self.parallel_config
            ignored_frozen_modules = self._frozen_target_modules(model)
            # DFlash-family models expose their transformer block class through
            # ``_no_split_modules``.  Preserve the legacy per-block FSDP policy:
            # it overlaps all-gather/reduce-scatter with decoder compute and is
            # required for the memory envelope of the larger recipes.  EAGLE
            # models do not advertise block classes and retain a single root
            # FSDP unit.
            block_names = set(
                getattr(optimizer_target, "_no_split_modules", None) or ()
            )
            block_classes = {
                type(module)
                for module in model.modules()
                if type(module).__name__ in block_names
            }
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
                        param_dtype=pc.param_dtype, buffer_dtype=torch.float32
                    ),
                    sharding_strategy=sharding,
                    process_group=pc.fsdp_process_group,
                )
                if ignored_frozen_modules:
                    fsdp_kwargs["ignored_modules"] = ignored_frozen_modules
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
            self.auto_wrap_block_classes = (
                block_classes if self._wrapper_kind == "fsdp" else set()
            )
            self.ignored_frozen_modules = ignored_frozen_modules
        if self._optimizer_factory is not None:
            target = optimizer_target if optimizer_target is not None else self.module
            self.optimizer = self._optimizer_factory(target)
            self._configure_optimizer_grad_norm()
        return self.module

    def set_optimizer(self, optimizer) -> None:
        self.optimizer = optimizer
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

        Non-boundary micro-steps run under the FSDP/DDP ``no_sync()`` context;
        the boundary backward reduces the accumulated sum once.
        """
        if is_boundary or not self._wrapped:
            loss.backward()
        else:
            with self.module.no_sync():
                loss.backward()

    def step(self) -> Optional[torch.Tensor]:
        """Run the optimizer step, which clips and returns the global grad norm."""
        if self.optimizer is None:
            raise RuntimeError(
                "FSDPTrainingBackend.step called before optimizer is set"
            )
        return self.optimizer.step()

    def state_dict(self) -> dict:
        """Full training state ``{"model", "optimizer", "rng"}`` for resume.

        ``model`` is gathered rank0-only (``{}`` on other ranks when wrapped).
        FSDP optimizer state is rank-local; DDP optimizer state is replicated.
        RNG state is always rank-local.
        """
        if self.module is None:
            raise RuntimeError("state_dict called before prepare_model")
        return {
            "model": self._module_state_dict(),
            "optimizer": (
                self.optimizer.state_dict() if self.optimizer is not None else None
            ),
            "rng": self._rng_state(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore whichever of module weights / optimizer / RNG the state carries."""
        if state.get("model") is not None:
            self._load_module_state_dict(state["model"])
        if self.optimizer is not None and state.get("optimizer") is not None:
            self.optimizer.load_state_dict(state["optimizer"])
        if state.get("rng") is not None:
            self._set_rng_state(state["rng"])

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
            # Named keys keep old CUDA checkpoints readable and make NPU state
            # inspectable without understanding a new opaque payload.
            "cuda": accelerator_state if device_type == "cuda" else None,
            "npu": accelerator_state if device_type == "npu" else None,
        }

    @staticmethod
    def _set_rng_state(rng: dict) -> None:
        torch.set_rng_state(rng["torch"])
        # ``device_type``/``npu`` are new. Falling back to the legacy ``cuda``
        # key preserves every checkpoint written before NPU support.
        device_type = rng.get("device_type")
        if device_type is None:
            device_type = "cuda" if rng.get("cuda") is not None else None
        state = rng.get(device_type) if device_type is not None else None
        module = getattr(torch, device_type, None) if device_type is not None else None
        if state is None or module is None or not module.is_available():
            return
        module.set_rng_state(state, module.current_device())


__all__ = ["ParallelConfig", "TrainingBackend", "FSDPTrainingBackend"]
