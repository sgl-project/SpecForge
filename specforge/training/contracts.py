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

from typing import (
    TYPE_CHECKING,
    Any,
    ContextManager,
    Dict,
    Mapping,
    Optional,
    Protocol,
    Sequence,
)

import torch

if TYPE_CHECKING:
    from specforge.runtime.contracts import TrainBatch
    from specforge.training.strategies.base import StepContext, StepOutput


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
        has_grad_norm: bool = False,
        learning_rate: Any = None,
    ) -> None:
        self._values = dict(values)
        self._has_grad_norm = has_grad_norm
        self._learning_rate = learning_rate
        self._host_values: Optional[Dict[str, float]] = None

    @property
    def learning_rate(self) -> Optional[float]:
        """The optimizer's rate once this update is applied.

        None when the optimizer does not report one. Converted on access: an
        optimizer may report a device scalar.
        """
        return None if self._learning_rate is None else float(self._learning_rate)

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
        """The optimizer's gradient norm before clipping; None when it reports none."""
        if not self._has_grad_norm:
            return None
        return self.materialize()["grad_norm"]


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

    A runner owns the wrapped model and its mode, the step's collectives
    across samples, normalization, clipping, the optimizer and the RNG. It
    does not decide which micro-batch comes next: the frame does. Every method
    runs on every rank, in the same order, with one exception: inside an
    evaluation pass a rank calls :meth:`evaluate` once for each batch of its
    own shard, which may be none.
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

    def prepass(
        self, batch: "TrainBatch", ctx: "StepContext"
    ) -> Dict[str, torch.Tensor]:
        """Run one micro-batch in training mode without gradients.

        Every replay statistic is delivered as zero. The runner remembers the
        RNG state the call started from and restores it for the replayed
        forward at the same position of the window.

        Returns:
            This rank's sums of each declared replay statistic, by name.
        """

    def evaluation(self) -> ContextManager[None]:
        """Delimit one evaluation pass.

        The model is in evaluation mode from entry to exit, between batches
        as well, and back in the mode it was found in afterwards, also when
        the pass raises. Every rank enters it, with or without batches.
        """

    def evaluate(self, batch: "TrainBatch", ctx: "StepContext") -> "StepOutput":
        """Run one batch of an evaluation pass, without gradients.

        The result is the program's own, normalized on this rank. The call
        reduces nothing across samples: whoever pools a pass does it once at
        the end through :meth:`reduce_sum` and :meth:`reduce_max`. A rank may
        therefore hold fewer evaluation batches than another, or none, unless
        the forward itself communicates: a sharded wrapper does, and so does
        a component that exchanges data on a model-parallel axis, whose ranks
        then have to evaluate the same batches. Detailed metrics are collected
        when the program declares that its evaluation reports them, whatever
        ``ctx.collect_detailed_metrics`` says.
        """

    def reduce_sum(self, values: torch.Tensor) -> torch.Tensor:
        """Sum ``values`` over the data-parallel ranks.

        ``values`` may be on the host, as a rank with nothing to report builds
        them; the result is on the device the ranks communicate on.
        """

    def reduce_max(self, values: torch.Tensor) -> torch.Tensor:
        """Take the maximum of ``values`` over the data-parallel ranks.

        It lets every rank size a buffer for the widest rank's data when its
        own shard was empty. ``values`` may be on the host, as for
        :meth:`reduce_sum`.
        """
