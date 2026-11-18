# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The plain step runner: one process, the program as built, no wrapper."""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from torch import nn

from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
from specforge.training.contracts import ProgramSpec, TrainingProgram
from specforge.training.runner import DataParallelRunner


class PlainRunner(DataParallelRunner):
    """Step runner of a job with one rank.

    The window's bookkeeping, the normalization, the replay's RNG and the
    optimizer are the reference runner's. What differs is what one process
    does not need: the program is not wrapped, no process group is read and
    no collective is issued, inside an initialized process group or not.

    Args:
        program: The program, placed on its device.
        spec: What the program declares.
        optimizer_factory: Builds the optimizer over ``optimizer_target``.
        optimizer_target: The module the optimizer trains; the program when
            omitted.
        world_size: The job's number of ranks, as the caller resolved it.

    Raises:
        ValueError: The job has more than one rank: every rank would train on
            its own data without ever meeting the others.
    """

    def __init__(
        self,
        program: TrainingProgram,
        spec: ProgramSpec,
        *,
        optimizer_factory: Callable[[nn.Module], Any],
        optimizer_target: Optional[nn.Module] = None,
        world_size: int,
    ) -> None:
        if world_size != 1:
            raise ValueError(
                f"the plain runner trains one process; this job has {world_size} ranks"
            )
        backend = FSDPTrainingBackend(
            ParallelConfig(), optimizer_factory=optimizer_factory
        )
        backend.prepare_model(program, wrap=False, optimizer_target=optimizer_target)
        super().__init__(program, backend, spec)

    def _world_size(self) -> int:
        return 1

    def _reduce(self, values: torch.Tensor, op: Any) -> torch.Tensor:
        return values

    def agree(self, error: Optional[str], *, value: Any = None) -> None:
        if error:
            raise RuntimeError(f"rank 0: {error}")


__all__ = ["PlainRunner"]
