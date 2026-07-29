# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Target-batch ownership for colocated tensor-parallel rollout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeVar

_T = TypeVar("_T")


@dataclass(frozen=True)
class TargetBatchPartition:
    """One rank's contiguous share of a target-TP capture batch.

    Every rank in a target tensor-parallel group captures the same logical
    batch.  The frozen target therefore sees ``size * local_batch_size``
    prompts, while this rank commits and trains only its contiguous share.
    """

    rank: int = 0
    size: int = 1

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError(f"target TP size must be >= 1, got {self.size}")
        if not 0 <= self.rank < self.size:
            raise ValueError(f"target TP rank {self.rank} is outside [0, {self.size})")

    @classmethod
    def from_distributed(cls, configured_size: int) -> "TargetBatchPartition":
        """Resolve the local target-TP rank and verify the configured group."""
        import torch.distributed as dist

        configured_size = int(configured_size)
        if configured_size < 1:
            raise ValueError(
                f"configured target TP size must be >= 1, got {configured_size}"
            )
        if not (dist.is_available() and dist.is_initialized()):
            if configured_size != 1:
                raise RuntimeError(
                    "target tp_size > 1 requires an initialized distributed "
                    "process group"
                )
            return cls()

        from specforge.distributed import get_tp_group

        group = get_tp_group()
        actual_size = dist.get_world_size(group)
        if actual_size != configured_size:
            raise ValueError(
                "configured target TP size does not match the initialized group: "
                f"configured={configured_size}, actual={actual_size}"
            )
        return cls(rank=dist.get_rank(group), size=actual_size)

    def target_batch_size(self, local_batch_size: int) -> int:
        if local_batch_size < 0:
            raise ValueError(
                f"local batch size must be non-negative, got {local_batch_size}"
            )
        return self.size * local_batch_size

    def local_slice(self, target_batch_size: int) -> slice:
        if target_batch_size % self.size:
            raise ValueError(
                f"target batch size {target_batch_size} is not divisible by "
                f"target TP size {self.size}"
            )
        local_batch_size = target_batch_size // self.size
        start = self.rank * local_batch_size
        return slice(start, start + local_batch_size)

    def select(self, values: Sequence[_T]) -> list[_T]:
        return list(values[self.local_slice(len(values))])


__all__ = ["TargetBatchPartition"]
