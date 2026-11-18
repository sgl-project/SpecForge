# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""What the run loop, the step frame and a step runner exchange.

The frame (:mod:`specforge.training.frame`) walks one optimizer window and a
runner executes each micro-batch of it. Everything that crosses between them
is declared here, so a second runner needs nothing from the first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Protocol, Sequence

import torch

if TYPE_CHECKING:
    from specforge.runtime.contracts import TrainBatch
    from specforge.training.strategies.base import StepContext


class IncompleteWindow(RuntimeError):
    """The training stream ended inside an optimizer window."""


class PendingStep:
    """A window whose gradients are complete and whose update is not applied.

    The boundary call of :meth:`StepRunner.forward_backward` returns it and
    :meth:`StepRunner.apply_step` takes it. It can be taken once, so an update
    cannot be applied twice or without the backward that produced it.
    """

    def __init__(self, window: Any) -> None:
        self._window = window
        self._taken = False

    def take(self) -> Any:
        """Hand the window to the runner that created this handle."""
        if self._taken:
            raise RuntimeError("an optimizer step can be applied only once")
        self._taken = True
        window, self._window = self._window, None
        return window


class StepMetrics:
    """What one optimizer update reports.

    Metric tensors stay on the training device until a consumer asks for host
    values, so an update that is not logged does not synchronize with the host.
    """

    def __init__(
        self,
        values: Mapping[str, Any],
        *,
        learning_rate: Optional[float] = None,
    ) -> None:
        self._values = dict(values)
        #: The optimizer's rate once this update is applied; None when the
        #: optimizer does not report one.
        self.learning_rate = learning_rate
        self._host_values: Optional[Dict[str, float]] = None

    def materialize(self) -> Dict[str, float]:
        """Copy every metric to the host in one synchronization, once."""
        if self._host_values is None:
            self._host_values = materialize_metrics(self._values)
        return self._host_values

    @property
    def loss(self) -> float:
        return self.materialize()["loss"]

    @property
    def grad_norm(self) -> Optional[float]:
        """The gradient norm before clipping; None when the runner has none."""
        return self.materialize().get("grad_norm")


def materialize_metrics(values: Mapping[str, Any]) -> Dict[str, float]:
    """Materialize scalar metrics with at most one device-to-host transfer."""
    host_values: Dict[str, float] = {}
    tensor_names = []
    tensors = []
    for name, value in values.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(
                    f"metric {name!r} must be scalar before materialization"
                )
            tensor_names.append(name)
            tensors.append(value.detach().float().reshape(()))
        else:
            host_values[name] = float(value)

    if tensors:
        device = tensors[0].device
        packed = torch.stack([tensor.to(device) for tensor in tensors])
        materialized = packed.cpu().tolist()
        host_values.update(
            {name: float(value) for name, value in zip(tensor_names, materialized)}
        )
    return host_values


class StepRunner(Protocol):
    """Executes the micro-batches of an optimizer window and applies its update.

    A runner owns the wrapped model, the step's collectives across samples,
    normalization, clipping, the optimizer and the RNG. It does not decide
    which micro-batch comes next: the frame does. Every method runs on every
    rank, in the same order.
    """

    #: Names of the statistics its program needs pooled over the whole window
    #: before the gradient pass. A window of such a program costs a second
    #: forward and host copies of its inputs.
    replay_statistics: Sequence[str]

    def forward_backward(
        self, batch: "TrainBatch", ctx: "StepContext", *, is_boundary: bool
    ) -> Optional[PendingStep]:
        """Run one micro-batch's forward and backward.

        Args:
            batch: This rank's micro-batch.
            ctx: Schedule state for the window; ``ctx.accumulation_steps`` is
                its length and ``ctx.replay_values`` holds the pooled replay
                statistics when this call replays a prepass.
            is_boundary: Whether this is the window's last micro-batch.

        Returns:
            None inside the window; at its boundary, the handle for
            :meth:`apply_step`.
        """

    def apply_step(self, pending: PendingStep) -> StepMetrics:
        """Normalize, clip and apply the window's update; report its metrics."""

    def prepass(self, batch: "TrainBatch", ctx: "StepContext") -> Dict[str, Any]:
        """Run one micro-batch in training mode without gradients.

        Every replay statistic is delivered as zero. The runner remembers the
        RNG state the call started from and restores it for the replayed
        forward at the same position of the window.

        Returns:
            This rank's sums of each declared replay statistic, by name.
        """

    def reduce_sum(self, values: torch.Tensor) -> torch.Tensor:
        """Sum ``values`` over the data-parallel ranks."""
