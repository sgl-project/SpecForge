# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""TrainingBackend: model wrapping / backward / optimizer step / state dict.

Currently implements ``FSDPTrainingBackend`` only. It carries a ``ParallelConfig``
object (the device-mesh / parallel-group handles) rather than a bare FSDP
module, so the existing FSDP + TP + Ulysses/Ring SP setup is preserved exactly.
The backend does not re-derive parallelism — it reads the handles SpecForge's
``init_distributed`` already created.

``torch`` is imported at module load (fine without a GPU); ``specforge.distributed``
and ``yunchang`` are imported lazily inside ``ParallelConfig.from_distributed`` so
this module stays importable in a CPU-only environment.
"""

from __future__ import annotations

import abc
import logging
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
    # opaque process-group / device-mesh handles (None in single-process).
    # The full set is carried so TP + Ulysses/Ring SP survive the trainer split
    # and a later reshard step has real handles to read — the parallelism is
    # NOT re-derived here, only snapshotted from init_distributed.
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
        """Snapshot ALL parallel handles created by ``init_distributed``.

        Captures the TP group + the Ulysses/Ring SP groups + both device meshes,
        not just DP — so the backend genuinely carries the parallel layout. A
        getter that is unexpectedly missing is logged, not silently swallowed.
        """
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
    def backward(self, loss: torch.Tensor) -> None: ...

    @abc.abstractmethod
    def step(self) -> Optional[torch.Tensor]: ...

    @abc.abstractmethod
    def state_dict(self) -> dict: ...

    @abc.abstractmethod
    def load_state_dict(self, state: dict) -> None: ...


class FSDPTrainingBackend(TrainingBackend):
    """FSDP1 backend mirroring today's SpecForge training math.

    Wraps the composite module (e.g. ``OnlineEagle3Model``) in FSDP with
    ``use_orig_params=True`` / bf16 mixed precision / ``SHARD_GRAD_OP`` over the
    configured process group, while the optimizer targets the inner trainable
    submodule (the draft model) — exactly as the legacy script.
    """

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
        """Register the trainable module, FSDP-wrapping it unless ``wrap=False``.

        ``wrap=False`` registers the module without sharding (single-rank /
        equivalence runs where FSDP would be a no-op) so ``state_dict`` and
        ``step`` still work without changing the math.
        """
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

    def backward(self, loss: torch.Tensor) -> None:
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
        if self.module is None:
            raise RuntimeError("state_dict called before prepare_model")
        if not self._wrapped:
            return self.module.state_dict()
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import StateDictType

        with FSDP.state_dict_type(self.module, StateDictType.FULL_STATE_DICT):
            return self.module.state_dict()

    def load_state_dict(self, state: dict) -> None:
        if self.optimizer is not None and "optimizer_state_dict" in state:
            self.optimizer.load_state_dict(state)


__all__ = ["ParallelConfig", "TrainingBackend", "FSDPTrainingBackend"]
