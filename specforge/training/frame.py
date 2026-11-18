# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The step frame: walks one optimizer window, the same way for every runner.

It issues no collective and touches no gradient, optimizer or RNG; those are
the runner's. It decides only the order in which a runner sees the window.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Dict, Iterable, Iterator, List, Tuple

import torch

from specforge.training.contracts import IncompleteWindow, StepMetrics, StepRunner

if TYPE_CHECKING:
    from specforge.runtime.contracts import TrainBatch
    from specforge.training.strategies.base import StepContext


def run_window(
    runner: StepRunner,
    window: Iterable["TrainBatch"],
    ctx: "StepContext",
) -> StepMetrics:
    """Consume one optimizer window and apply its update.

    Args:
        runner: Executes each micro-batch and applies the update.
        window: At most ``ctx.accumulation_steps`` micro-batches, consumed
            lazily: each is handed to the runner as soon as it arrives.
        ctx: Schedule state, the same for every micro-batch of the window.

    Returns:
        The applied update's metrics.

    Raises:
        IncompleteWindow: The stream ended inside the window. No update is
            applied, and the runner is left mid-window: it cannot start
            another one.
    """
    if runner.replay_statistics:
        return _run_replayed_window(runner, window, ctx)
    pending = None
    received = 0
    for received, batch in _bounded(window, ctx):
        pending = runner.forward_backward(
            batch, ctx, is_boundary=received == ctx.accumulation_steps
        )
    # The frame's own count decides, whatever a runner returned on the way.
    if received != ctx.accumulation_steps:
        raise _incomplete(received, ctx)
    return runner.apply_step(pending)


def _bounded(
    window: Iterable["TrainBatch"], ctx: "StepContext"
) -> Iterator[Tuple[int, "TrainBatch"]]:
    """Number the window's micro-batches from one; refuse one too many."""
    for received, batch in enumerate(window, start=1):
        if received > ctx.accumulation_steps:
            raise ValueError(
                f"an optimizer window holds at most {ctx.accumulation_steps} "
                "micro-batches"
            )
        yield received, batch


def _run_replayed_window(
    runner: StepRunner,
    window: Iterable["TrainBatch"],
    ctx: "StepContext",
) -> StepMetrics:
    """Pool the window's statistics in a prepass, then replay it for gradients."""
    names = tuple(runner.replay_statistics)
    kept: List["TrainBatch"] = []
    sums: Dict[str, torch.Tensor] = {}
    for _, batch in _bounded(window, ctx):
        local = runner.prepass(batch, ctx)
        for name in names:
            value = local[name].detach()
            # The pooled sums are reduced in place; they must not be the
            # program's own tensor.
            sums[name] = sums[name] + value if name in sums else value.clone()
        # The replay reads the inputs again; the device keeps one micro-batch.
        kept.append(
            replace(
                batch,
                tensors={
                    key: value.detach().cpu() for key, value in batch.tensors.items()
                },
            )
        )
    if len(kept) != ctx.accumulation_steps:
        raise _incomplete(len(kept), ctx)
    replay = replace(
        ctx, replay_values={name: runner.reduce_sum(sums[name]) for name in names}
    )
    pending = None
    for index, batch in enumerate(kept, start=1):
        pending = runner.forward_backward(
            batch, replay, is_boundary=index == ctx.accumulation_steps
        )
    return runner.apply_step(pending)


def _incomplete(received: int, ctx: "StepContext") -> IncompleteWindow:
    return IncompleteWindow(
        "training stream ended with incomplete gradient accumulation: "
        f"received {received} of {ctx.accumulation_steps} "
        "micro-batches after the last optimizer step; no partial "
        "optimizer step or durable acknowledgement was committed"
    )
