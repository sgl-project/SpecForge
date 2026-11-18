# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The DDP / FSDP step runner: one micro-batch per call over the wrapped model.

The collectives across samples of a training step are issued here or in the
backend this runner drives: gradient synchronization, the loss denominators
of either reduction, the gradient norm, the window's metrics and the replay
statistics.
"""

from __future__ import annotations

import contextlib
import math
from collections import deque
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Deque, Dict, Mapping, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from specforge.runtime.contracts import TrainBatch
from specforge.training.backend import TrainingBackend
from specforge.training.contracts import (
    CheckpointState,
    PendingStep,
    ProgramSpec,
    RankFailure,
    StepContext,
    StepMetrics,
    TrainingProgram,
)

if TYPE_CHECKING:
    from specforge.algorithms.objectives.base import ObjectiveResult


def _check_metrics(values: Dict[str, Any], sums: Dict[str, Any]) -> None:
    """Raise for metrics the reduction cannot pack.

    A name used for both a ratio and a sum, a ratio that is not a pair, or a
    pair whose sides differ in size.
    """
    if values.keys() & sums.keys():
        raise ValueError("ratio and sum metric names must be distinct")
    for name in sorted(values):
        pair = values[name]
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise TypeError(
                f"ratio metric {name!r} must be a (numerator, denominator) pair"
            )
        numerator, denominator = (torch.as_tensor(side).numel() for side in pair)
        if numerator != denominator:
            raise ValueError(
                f"ratio metric {name!r} shape mismatch: "
                f"({numerator},) vs ({denominator},)"
            )


def _reduce_ratio_metrics(
    values: Dict[str, Any],
    *,
    device: torch.device,
    process_group: Any,
    reduce: bool,
    sums: Optional[Dict[str, Any]] = None,
) -> Dict[str, torch.Tensor]:
    """Reduce ratios and additive telemetry in one collective.

    Ratios with no observations retain the existing zero convention; log their
    counts through ``sums`` when consumers need to distinguish missing data.
    """

    sums = sums or {}
    if not values and not sums:
        return {}
    _check_metrics(values, sums)
    normalized = [
        (
            name,
            torch.as_tensor(values[name][0]).detach().float().flatten().to(device),
            torch.as_tensor(values[name][1]).detach().float().flatten().to(device),
        )
        for name in sorted(values)
    ]

    normalized_sums = [
        (name, torch.as_tensor(sums[name]).detach().float().flatten().to(device))
        for name in sorted(sums)
    ]
    packed = torch.cat(
        [
            tensor
            for _, numerator, denominator in normalized
            for tensor in (numerator, denominator)
        ]
        + [tensor for _, tensor in normalized_sums]
    )
    if reduce and dist.is_available() and dist.is_initialized():
        world = (
            dist.get_world_size()
            if process_group is None
            else dist.get_world_size(group=process_group)
        )
        if world > 1:
            if process_group is None:
                dist.all_reduce(packed)
            else:
                dist.all_reduce(packed, group=process_group)

    output: Dict[str, torch.Tensor] = {}
    cursor = 0
    for name, numerator, _denominator in normalized:
        width = numerator.numel()
        summed_numerator = packed[cursor : cursor + width]
        cursor += width
        summed_denominator = packed[cursor : cursor + width]
        cursor += width
        ratios = summed_numerator / summed_denominator.clamp_min(1e-12)
        if width == 1:
            output[name] = ratios.reshape(())
        else:
            output.update({f"{name}_{index}": ratios[index] for index in range(width)})
    for name, tensor in normalized_sums:
        width = tensor.numel()
        total = packed[cursor : cursor + width]
        cursor += width
        if width == 1:
            output[name] = total.reshape(())
        else:
            output.update({f"{name}_{index}": total[index] for index in range(width)})
    return output


@dataclass(frozen=True)
class _RngState:
    """The generators a forward draws from, as one prepass found them."""

    device: torch.device
    cpu: torch.Tensor
    accelerator: Optional[torch.Tensor]


@dataclass(frozen=True)
class _Window:
    """One completed window, between its last backward and its update."""

    #: The loss denominator, summed over the window and the ranks.
    denominator: torch.Tensor
    #: This rank's ratio pairs and additive telemetry, summed over the window.
    ratio_totals: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    sum_totals: Dict[str, torch.Tensor]
    accumulation_steps: int


class DataParallelRunner:
    """Step runner for a model wrapped by DDP or FSDP over data-parallel ranks.

    Under the ``window`` reduction the update's objective is
    ``sum(numerator) / sum(denominator)`` over the window's micro-batches and
    the ranks. Each numerator is divided by the window length before its
    backward, the wrapper averages gradients over the ranks, and the
    accumulated gradients are rescaled once by
    ``world size * window length / global denominator``.

    Under ``microbatch`` every term's denominator is pooled over the ranks
    between a micro-batch's forward and its backward, in one reduction. The
    loss that is differentiated is ``world size * sum_t c_t * numerator_t /
    pooled denominator_t``, which the wrapper's average over the ranks turns
    into the micro-batch's global mean; its denominator is one, so the
    window's rescale is by one.

    Args:
        program: The program as the backend wrapped it; its forward runs
            through the wrapper.
        backend: The wrapper's backward and the optimizer.
        spec: What the program declares.
    """

    def __init__(
        self, program: TrainingProgram, backend: TrainingBackend, spec: ProgramSpec
    ) -> None:
        self.program = program
        self.backend = backend
        self.spec = spec
        # DDP hides the program's own methods behind ``module``; FSDP forwards
        # them.
        self._unwrapped: TrainingProgram = (
            program.module if isinstance(program, DistributedDataParallel) else program
        )
        self._ratio_totals: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._sum_totals: Dict[str, torch.Tensor] = {}
        # RNG states of the window's prepasses, in the order they ran.
        self._prepass_rng: Deque[_RngState] = deque()
        # Micro-batches of the open window whose backward has run, and the
        # length the window's first micro-batch gave it.
        self._micro = 0
        self._window_steps = 0
        self._pending: Optional[PendingStep] = None
        # Set once a micro-batch has raised inside its forward or backward:
        # gradients and totals may then hold a part of it.
        self._broken: Optional[str] = None
        # Whether the next training call has to set the model's mode, and
        # whether an evaluation pass is open.
        self._mode_unset = True
        self._evaluating = False

    @property
    def replay_statistics(self) -> Tuple[str, ...]:
        return self.spec.replay_statistics

    def _world_size(self) -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size(group=self.backend.process_group)
        return 1

    def reduce_sum(self, values: torch.Tensor) -> torch.Tensor:
        return self._reduce(values, dist.ReduceOp.SUM)

    def reduce_max(self, values: torch.Tensor) -> torch.Tensor:
        return self._reduce(values, dist.ReduceOp.MAX)

    def _reduce(self, values: torch.Tensor, op: Any) -> torch.Tensor:
        if self._world_size() > 1:
            # The ranks exchange tensors of the training device; a rank with
            # nothing to report built its own on the host.
            values = values.to(self._device())
            dist.all_reduce(values, op=op, group=self.backend.process_group)
        return values

    def _device(self) -> torch.device:
        return next(self.program.parameters()).device

    def _forward(
        self, batch: TrainBatch, ctx: StepContext, **replay: Any
    ) -> "ObjectiveResult":
        """Place one batch on the training device and run the program on it."""
        missing = sorted(set(self.spec.inputs) - batch.tensors.keys())
        if missing:
            raise ValueError(
                f"{self.spec.name} batch missing required features {missing}; "
                f"present={sorted(batch.tensors)}"
            )
        device = self._device()
        inputs = batch.tensors
        prepared = self._unwrapped.host_inputs(inputs)
        if prepared:
            if device.type != "cpu":
                # a pageable source makes the non_blocking copy wait for the host
                prepared = {
                    key: value.pin_memory() if value.device.type == "cpu" else value
                    for key, value in prepared.items()
                }
            inputs = {**inputs, **prepared}
        tensors = {
            key: value.to(device, non_blocking=True) for key, value in inputs.items()
        }
        return self.program(
            tensors,
            step=ctx.global_step,
            total_steps=ctx.total_steps,
            collect_detailed_metrics=ctx.collect_detailed_metrics,
            **replay,
        )

    def checkpoint_state(self) -> CheckpointState:
        # Every rank takes part: gathering a sharded model is a collective.
        state = self.backend.state_dict()
        # DDP ranks hold identical parameters, gradients, optimizer moments
        # and scheduler state, so that large state is saved once. Sharded
        # optimizer state is each rank's own, and so is every rank's RNG.
        replicated = self.backend.optimizer_state_is_replicated
        try:
            # This rank's own work, once the gather is over.
            return CheckpointState(
                # Only the rank that writes the shared file selects the
                # draft's weights; what the others hold of the model is
                # partial under sharding.
                draft=(
                    self._unwrapped.draft_state(state["model"])
                    if self._first_rank()
                    else {}
                ),
                rank={
                    "optimizer": None if replicated else state["optimizer"],
                    "rng": state["rng"],
                },
                replicated=state["optimizer"] if replicated else None,
                kernels=dict(self._unwrapped.kernel_record()),
            )
        except Exception as error:
            raise RankFailure(f"{type(error).__name__}: {error}") from error

    @staticmethod
    def _first_rank() -> bool:
        """Whether this is the process that writes a checkpoint's shared file."""
        return not (dist.is_available() and dist.is_initialized()) or (
            dist.get_rank() == 0
        )

    def load_resume(self, state: Mapping[str, Any]) -> None:
        self.backend.load_state_dict(state)

    def agree(self, error: Optional[str], *, value: Any = None) -> None:
        if dist.is_available() and dist.is_initialized():
            outcomes = [None] * dist.get_world_size(group=self.backend.process_group)
            dist.all_gather_object(
                outcomes, (error, value), group=self.backend.process_group
            )
        else:
            outcomes = [(error, value)]
        failures = [
            f"rank {rank}: {failure}"
            for rank, (failure, _) in enumerate(outcomes)
            if failure is not None
        ]
        if failures:
            raise RuntimeError("; ".join(failures))
        first, *others = [reported for _, reported in outcomes]
        if any(reported != first for reported in others):
            raise RuntimeError(f"ranks disagree: {[first, *others]}")

    def _train_mode(self) -> None:
        """Put the whole model in training mode before a training call.

        Once at the start and once after every evaluation pass, as the run
        loop used to. The call is recursive on purpose: a draft can sit in
        another mode than the module that wraps it, and an objective reads the
        draft's own flag to choose its normalization.
        """
        if self._mode_unset or not self.program.training:
            self.program.train()
            self._mode_unset = False

    @contextlib.contextmanager
    def evaluation(self):
        was_training = self.program.training
        self.program.eval()
        self._evaluating = True
        try:
            yield
        finally:
            self._evaluating = False
            self.program.train(was_training)
            self._mode_unset = True

    def evaluate(self, batch: TrainBatch, ctx: StepContext) -> "ObjectiveResult":
        if not self._evaluating:
            raise RuntimeError("evaluate belongs inside an evaluation() pass")
        ctx = replace(ctx, collect_detailed_metrics=self.spec.evaluation_diagnostics)
        # A program with a statistic is told there is none pooled: it then
        # uses the batch's own.
        arguments = (
            self._unwrapped.replay_arguments({}) if self.replay_statistics else {}
        )
        with torch.no_grad():
            return self._forward(batch, ctx, **arguments)

    def prepass(self, batch: TrainBatch, ctx: StepContext) -> Dict[str, torch.Tensor]:
        self._check_training_call()
        if not self.replay_statistics:
            # Its RNG state would be kept for a replay that never comes and
            # handed to the next gradient pass instead.
            raise RuntimeError("this program declares no replay statistic")
        if self._micro or self._pending is not None:
            raise RuntimeError("a prepass belongs before the window's gradient pass")
        self._hold_window_length(ctx, opening=not self._prepass_rng)
        self._train_mode()
        device = self._device()
        accelerator = None
        if device.type != "cpu":
            accelerator = torch.get_device_module(device.type).get_rng_state(
                device.index
            )
        state = _RngState(device, torch.get_rng_state(), accelerator)
        with torch.no_grad():
            statistics = self._forward(
                batch,
                ctx,
                prepass=True,
                **self._unwrapped.replay_arguments({}, prepass=True),
            ).statistics
        # Kept only for a prepass that ran: each one pairs with one replay.
        self._prepass_rng.append(state)
        return statistics

    def forward_backward(
        self, batch: TrainBatch, ctx: StepContext, *, is_boundary: bool
    ) -> Optional[PendingStep]:
        self._check_training_call()
        if self._pending is not None:
            raise RuntimeError(
                "the previous window's optimizer step has not been applied"
            )
        self._hold_window_length(
            ctx, opening=self._micro == 0 and not self._prepass_rng
        )
        remaining = ctx.accumulation_steps - self._micro
        if self.replay_statistics:
            # A program that declares a replay statistic has no other gradient
            # pass: skipping the prepass would train on a rank-local value.
            if set(ctx.replay_values) != set(self.replay_statistics):
                raise RuntimeError(
                    f"replay statistics {list(self.replay_statistics)} were not "
                    "pooled over the window: the gradient pass has to be the "
                    "replay of a prepass"
                )
            if len(self._prepass_rng) != remaining:
                raise RuntimeError(
                    f"the window has {remaining} micro-batches left to replay "
                    f"and {len(self._prepass_rng)} prepasses to pair them with"
                )
        elif ctx.replay_values:
            raise RuntimeError("this program declares no replay statistic")
        if is_boundary != (remaining == 1):
            raise RuntimeError(
                f"the window of {ctx.accumulation_steps} micro-batches has "
                f"{remaining} left; is_boundary={is_boundary} does not close it"
            )
        try:
            self._forward_backward(batch, ctx, is_boundary=is_boundary)
        except BaseException as error:
            self._broken = f"{type(error).__name__}: {error}"
            raise
        self._micro += 1
        if not is_boundary:
            return None
        self._micro = 0
        window = _Window(
            denominator=self.reduce_sum(self._ratio_totals["loss"][1].clone()),
            ratio_totals=self._ratio_totals,
            sum_totals=self._sum_totals,
            accumulation_steps=ctx.accumulation_steps,
        )
        self._ratio_totals = {}
        self._sum_totals = {}
        self._pending = PendingStep(window)
        return self._pending

    def _forward_backward(
        self, batch: TrainBatch, ctx: StepContext, *, is_boundary: bool
    ) -> None:
        """One micro-batch's forward, its share of the window's totals, its backward."""
        self._train_mode()
        # A program that declares a replay statistic turns the pooled sums
        # into its forward's arguments; that arithmetic stays out of the
        # wrapper, which may cast what passes through its forward.
        replay = (
            self._unwrapped.replay_arguments(ctx.replay_values)
            if self.replay_statistics
            else {}
        )
        # DDP defers its gradient synchronization only when the scope encloses
        # the forward as well as the backward.
        with self.backend.accumulation_context(is_boundary=is_boundary):
            with self._replayed_rng():
                result = self._forward(batch, ctx, **replay)
            numerator, denominator = self._loss_terms(result)
            if numerator.numel() != 1 or denominator.numel() != 1:
                raise ValueError("the loss numerator and denominator must be scalars")
            ratio_metrics = dict(result.ratios)
            ratio_metrics["loss"] = (
                numerator.detach().reshape(()),
                denominator.detach().reshape(()),
            )
            # Checked against the window so far, before this backward: found
            # at the boundary, a malformed metric would raise after the update
            # was applied and before it was counted or acknowledged.
            _check_metrics(
                {**self._ratio_totals, **ratio_metrics},
                {**self._sum_totals, **result.sums},
            )
            self._accumulate_ratio_metrics(ratio_metrics)
            for name, value in result.sums.items():
                self._sum_totals[name] = (
                    self._sum_totals.get(name, 0) + torch.as_tensor(value).detach()
                )
            loss = numerator.reshape(()) / ctx.accumulation_steps
            self.backend.backward(loss, is_boundary=is_boundary)

    def _loss_terms(
        self, result: "ObjectiveResult"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One micro-batch's loss numerator and denominator under the declared reduction."""
        pooled = self.spec.reduction == "microbatch"
        if pooled != bool(result.terms):
            raise ValueError(
                f"{self.spec.name} declares the {self.spec.reduction!r} reduction "
                f"and returned {len(result.terms)} loss terms from a training call"
            )
        if not pooled:
            return result.numerator, result.denominator
        # Every rank holds the same terms in the same order.
        denominators = self.reduce_sum(
            torch.stack([denominator for _, _, denominator in result.terms])
        )
        world_size = self._world_size()
        loss = sum(
            coefficient
            * (world_size * numerator / denominator.where(denominator > 0, 1))
            for (coefficient, numerator, _), denominator in zip(
                result.terms, denominators
            )
        )
        return loss, denominators.new_ones(())

    def apply_step(self, pending: PendingStep) -> StepMetrics:
        if pending is not self._pending:
            raise RuntimeError("this optimizer step belongs to another window")
        window: _Window = pending.take()
        denominator = window.denominator
        # A capable optimizer validates the global denominator in its single
        # host read; otherwise it is checked here with a synchronization.
        # Either way an invalid denominator raises before any state updates.
        defer_check = self.backend.checks_loss_denominator
        if not defer_check:
            denominator_value = denominator.item()
            if not math.isfinite(denominator_value) or denominator_value <= 0:
                raise ValueError("global loss denominator must be finite and positive")
        # new_full fills on the device; new_tensor would be a synchronizing H2D.
        scale = (
            denominator.new_full((), self._world_size() * window.accumulation_steps)
            / denominator
        )
        self.backend.scale_gradients(scale)
        if defer_check:
            grad_norm = self.backend.step(loss_denominator=denominator)
        else:
            grad_norm = self.backend.step()
        # The update is applied; a step that raised above leaves the handle in
        # place, so the runner starts no window on top of its gradients.
        self._pending = None
        # Ratios and sums are reduced as counts, in one collective.
        metrics: Dict[str, Any] = _reduce_ratio_metrics(
            window.ratio_totals,
            # Where the program computes: a metric side it built on the host
            # would otherwise pull every other one there.
            device=self._device(),
            process_group=self.backend.process_group,
            reduce=True,
            sums=window.sum_totals,
        )
        if grad_norm is not None:
            metrics["grad_norm"] = (
                grad_norm.detach().float().mean()
                if isinstance(grad_norm, torch.Tensor)
                else float(grad_norm)
            )
        return StepMetrics(
            metrics,
            has_grad_norm=grad_norm is not None,
            learning_rate=self.backend.learning_rate(),
        )

    def _check_training_call(self) -> None:
        if self._broken is not None:
            raise RuntimeError(
                "this runner cannot continue: a micro-batch raised inside its "
                f"forward or backward ({self._broken})"
            )
        if self._evaluating:
            raise RuntimeError("a training call inside an evaluation pass")

    def _hold_window_length(self, ctx: StepContext, *, opening: bool) -> None:
        """Keep the length the window's first call gave it, prepass or backward."""
        if opening:
            self._window_steps = ctx.accumulation_steps
        elif ctx.accumulation_steps != self._window_steps:
            raise RuntimeError(
                f"the open window accumulates {self._window_steps} micro-batches, "
                f"not {ctx.accumulation_steps}"
            )

    @contextlib.contextmanager
    def _replayed_rng(self):
        """Give a replayed forward the RNG state its prepass started from.

        The generators continue afterwards from where the prepasses left them.
        """
        if not self._prepass_rng:
            yield
            return
        state = self._prepass_rng.popleft()
        device = state.device
        devices = [] if state.accelerator is None else [device.index]
        with torch.random.fork_rng(devices=devices, device_type=device.type):
            torch.set_rng_state(state.cpu)
            if state.accelerator is not None:
                torch.get_device_module(device.type).set_rng_state(
                    state.accelerator, device.index
                )
            yield

    def _accumulate_ratio_metrics(self, values: Dict[str, Any]) -> None:
        for name, (raw_numerator, raw_denominator) in values.items():
            numerator = torch.as_tensor(raw_numerator).detach()
            denominator = torch.as_tensor(raw_denominator).detach()
            previous = self._ratio_totals.get(name)
            if previous is not None:
                numerator = previous[0] + numerator
                denominator = previous[1] + denominator
            self._ratio_totals[name] = (numerator, denominator)


__all__ = ["DataParallelRunner"]
