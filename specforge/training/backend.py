# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""TrainingBackend: model wrapping / backward / optimizer step / state dict.

FSDP-only for now. ``ParallelConfig`` carries the parallel handles created by
``init_distributed`` rather than re-deriving them; distributed imports stay
lazy so the module is importable without a GPU.
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
    """Handles describing the active parallel layout. Carried, not re-derived."""

    world_size: int = 1
    tp_size: int = 1
    sp_ulysses_size: int = 1
    sp_ring_size: int = 1
    sharding_strategy: str = "SHARD_GRAD_OP"
    param_dtype: torch.dtype = torch.bfloat16
    # opaque process-group / device-mesh handles snapshotted from
    # init_distributed, never re-derived (None in single-process).
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
        """Snapshot all parallel handles (DP/TP/SP groups, device meshes) from
        ``init_distributed``; a missing getter is logged, never silently skipped."""
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
                except Exception as exc:  # group not built for this config
                    logging.getLogger(__name__).warning(
                        "ParallelConfig.from_distributed: %s() unavailable: %s",
                        getter,
                        exc,
                    )
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "ParallelConfig.from_distributed: specforge.distributed import failed: %s",
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
    """FSDP1 backend mirroring the legacy SpecForge training math: FSDP with
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

    def prepare_model(
        self,
        model: nn.Module,
        *,
        wrap: bool = True,
        optimizer_target: Optional[nn.Module] = None,
    ) -> nn.Module:
        """Register the trainable module, FSDP-wrapping it unless ``wrap=False``
        (single-rank / equivalence runs where sharding would be a no-op)."""
        if not wrap:
            self.module = model
            self._wrapped = False
        else:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

            pc = self.parallel_config
            sharding = getattr(ShardingStrategy, pc.sharding_strategy)
            model = FSDP(
                model,
                use_orig_params=True,
                mixed_precision=MixedPrecision(
                    param_dtype=pc.param_dtype, buffer_dtype=pc.param_dtype
                ),
                sharding_strategy=sharding,
                process_group=pc.fsdp_process_group,
            )
            self.module = model
            self._wrapped = True
        if self._optimizer_factory is not None:
            target = optimizer_target if optimizer_target is not None else self.module
            self.optimizer = self._optimizer_factory(target)
        return self.module

    def set_optimizer(self, optimizer) -> None:
        self.optimizer = optimizer

    def backward(self, loss: torch.Tensor, *, is_boundary: bool = True) -> None:
        """Backward one micro-step. FSDP reduce-scatters grads on every backward,
        so non-boundary micro-steps run under ``no_sync()`` and the boundary
        backward reduces the accumulated sum once — identical math, one
        collective per optimizer step."""
        if is_boundary or not self._wrapped:
            loss.backward()
        else:
            with self.module.no_sync():
                loss.backward()

    def step(self) -> Optional[torch.Tensor]:
        """Optimizer step + the distributed grad-norm reduction (run_backward_and_update)."""
        if self.optimizer is None:
            raise RuntimeError(
                "FSDPTrainingBackend.step called before optimizer is set"
            )
        grad_norm = self.optimizer.step()
        if grad_norm is not None and dist.is_initialized():
            grad_norm = grad_norm.detach().float()
            if torch.cuda.is_available():
                grad_norm = grad_norm.to(torch.cuda.current_device())
            grad_norm = grad_norm.pow(2)
            dist.all_reduce(grad_norm, op=dist.ReduceOp.SUM)
            grad_norm = grad_norm.sqrt()
        return grad_norm

    def state_dict(self) -> dict:
        """Full training state ``{"model", "optimizer", "rng"}`` for resume.

        ``model`` is gathered rank0-only with CPU offload (``{}`` on other ranks
        when wrapped); ``optimizer``/``rng`` are rank-local and must be persisted
        per rank — restoring rank0's copy everywhere corrupts the other ranks.
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
        if not self._wrapped:
            return contextlib.nullcontext()
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import StateDictType

        return FSDP.state_dict_type(
            self.module, StateDictType.FULL_STATE_DICT, state_dict_config
        )

    def _module_state_dict(self) -> dict:
        if not self._wrapped:
            return self.module.state_dict()
        from torch.distributed.fsdp import FullStateDictConfig

        # gather to rank0 CPU only — materializing the full model on every
        # rank's GPU is wasted memory when only rank0 writes it.
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with self._full_state_ctx(cfg):
            return self.module.state_dict()

    def _load_module_state_dict(self, model_state: dict) -> None:
        # every rank loads the full state dict read from the shared file.
        with self._full_state_ctx():
            self.module.load_state_dict(model_state)

    @staticmethod
    def _rng_state() -> dict:
        # single bound-device CUDA state keeps the checkpoint independent of
        # how many devices happen to be visible at save time.
        return {
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state(torch.cuda.current_device())
                if torch.cuda.is_available()
                else None
            ),
        }

    @staticmethod
    def _set_rng_state(rng: dict) -> None:
        cpu_state = rng.get("torch", rng.get("cpu"))  # "cpu" = legacy key
        if cpu_state is not None:
            torch.set_rng_state(cpu_state)
        cuda_state = rng.get("cuda")
        if cuda_state is None or not torch.cuda.is_available():
            return
        device = torch.cuda.current_device()
        if isinstance(cuda_state, list):  # legacy get_rng_state_all format
            if device >= len(cuda_state):
                raise ValueError(
                    f"legacy RNG checkpoint holds {len(cuda_state)} CUDA states "
                    f"but this rank's bound device index is {device}; it was "
                    "saved with fewer visible devices and cannot be restored here"
                )
            cuda_state = cuda_state[device]
        torch.cuda.set_rng_state(cuda_state, device)


__all__ = ["ParallelConfig", "TrainingBackend", "FSDPTrainingBackend"]
