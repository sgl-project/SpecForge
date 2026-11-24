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
    does not need: the program is not wrapped and the runner issues no
    collective of its own, inside an initialized process group or not. What
    the program's forward does is the program's: a model-parallel axis, or a
    compiler that coordinates ranks.

    Args:
        program: The program, placed on its device. Its weights are loaded
            (``load_weights``) before the optimizer is built.
        spec: What the program declares.
        optimizer_factory: Builds the optimizer over ``optimizer_target``.
        optimizer_target: The module the optimizer trains; the program when
            omitted.
        world_size: The job's number of ranks, as the caller resolved it.

    Raises:
        ValueError: The job has more than one rank: every rank would train on
            its own data without ever meeting the others. Or the program sits
            on another accelerator than the process's current one, whose
            generator is the one a checkpoint saves and restores.
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
        device = next(program.parameters()).device
        if device.type != "cpu":
            current = torch.get_device_module(device.type).current_device()
            if device.index not in (None, current):
                raise ValueError(
                    f"the program is on {device} and the process's current "
                    f"{device.type} device is {current}"
                )
        # The optimizer clones its fp32 masters when it is built.
        program.load_weights()
        backend = FSDPTrainingBackend(
            ParallelConfig(), optimizer_factory=optimizer_factory
        )
        backend.prepare_model(
            program,
            wrap=False,
            optimizer_target=optimizer_target,
            # Checked as the reference checks them: none may be trainable.
            frozen=program.frozen,
        )
        super().__init__(program, backend, spec)

    def _world_size(self) -> int:
        return 1

    def _reduce(self, values: torch.Tensor, op: Any) -> torch.Tensor:
        return values

    def agree(self, error: Optional[str], *, value: Any = None) -> None:
        if error is not None:
            raise RuntimeError(f"rank 0: {error}")


__all__ = ["PlainRunner"]
