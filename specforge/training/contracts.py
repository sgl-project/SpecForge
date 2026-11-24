# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""What the run loop, the step frame, a step runner and a program exchange.

The frame (:mod:`specforge.training.frame`) walks one optimizer window and a
runner executes each micro-batch of it on a program. Everything that crosses
between them is declared here, so a second runner needs nothing from the
first, and a runner needs no name of an objective or an architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    ContextManager,
    Dict,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

import torch

if TYPE_CHECKING:
    from specforge.algorithms.objectives.base import ObjectiveResult
    from specforge.runtime.contracts import TrainBatch


#: Key of ``TrainBatch.metadata`` under which the run loop records the epoch a
#: training batch belongs to, as it pulls the batch. A window can span an epoch
#: boundary, so the epoch belongs to the batch, not to the window.
EPOCH_METADATA_KEY = "epoch"
#: The epoch an evaluation forward reports: one value for every pass, so a
#: draw keyed on the epoch is the same at every evaluation.
EVALUATION_EPOCH = -1


@dataclass(frozen=True)
class StepContext:
    """Schedule state for one optimizer window; values only.

    ``collect_detailed_metrics`` gates expensive diagnostics to log steps.
    ``replay_values`` holds each replay statistic's sums over the optimizer
    window and the data-parallel ranks, by name; it is empty except while the
    window is replayed for gradients.
    """

    global_step: int = 0
    total_steps: Optional[int] = None
    collect_detailed_metrics: bool = True
    accumulation_steps: int = 1
    replay_values: Mapping[str, torch.Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class OptimizerSpec:
    """Declarative optimizer configuration consumed by the shared registry.

    The runtime constructs the optimizer over the wrapped trainable module so
    parameter names and sharded/DTensor layouts stay intact.
    """

    kind: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    param_group_policy: str = "default"
    offload_master: bool = False
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SchedulerSpec:
    """Declarative learning-rate schedule consumed by the shared registry."""

    kind: str = "cosine"
    total_steps: Optional[int] = None
    warmup_ratio: float = 0.015
    min_lr_ratio: float = 0.0
    extra: Mapping[str, Any] = field(default_factory=dict)


#: Runtimes ``training.runtime`` names, and how each holds the trainable draft.
RUNTIMES = ("fsdp", "fsdp2")


@dataclass(frozen=True)
class PrecisionSpec:
    """How the trainable draft's numbers are stored, computed, reduced and written.

    Resolved once for a run (``resolve``) and handed to the trainer. The
    runtime refuses a combination it cannot hold, and checks the trainable
    parameters are stored as declared and the optimizer keeps the declared
    masters. Under ``fsdp`` the separate fp32 master copy is
    ``BF16Optimizer``'s, and the artifact is written in the parameter dtype.
    Frozen modules (bound target tables, the teacher head) keep the dtype they
    were built in and are not described here.
    """

    #: What the trainable parameters are stored in.
    parameter_dtype: torch.dtype
    #: What the runtime gives the parameters in for forward and backward. A
    #: module may cast further inside (Domino's GRU runs fp16 on Ascend).
    compute_dtype: torch.dtype
    #: What gradients accumulate in across micro-batches and are reduced in
    #: across ranks.
    gradient_dtype: torch.dtype
    #: The optimizer's own copy of the parameters, when it keeps one; None
    #: when the parameters themselves are the masters.
    master_dtype: Optional[torch.dtype]
    #: What checkpoints and exports write the draft's weights in.
    artifact_dtype: torch.dtype
    #: What floating-point buffers are cast to while wrapped; None leaves
    #: buffers in the dtype they were built in (DDP / unwrapped).
    buffer_dtype: Optional[torch.dtype]

    @classmethod
    def resolve(
        cls,
        *,
        runtime: str,
        model_dtype: torch.dtype,
        sharding_strategy: Optional[str] = None,
        gradient_dtype: Optional[torch.dtype] = None,
    ) -> "PrecisionSpec":
        """The precision a runtime trains a draft built in ``model_dtype`` with.

        ``fsdp`` (DDP / FSDP1): parameters, compute, gradients and the artifact
        in ``model_dtype``; ``BF16Optimizer`` keeps fp32 masters; FSDP1 casts
        floating buffers to fp32 while DDP (``NO_SHARD``) leaves them as built.
        ``fsdp2``: fp32 parameters that are the masters, compute in
        ``model_dtype``, fp32 gradients by default (or ``model_dtype`` when
        explicitly requested via ``gradient_dtype``), fp32 floating buffers,
        artifact in ``model_dtype``.
        """
        if not model_dtype.is_floating_point:
            raise ValueError(f"model dtype must be floating point, not {model_dtype}")
        if gradient_dtype is not None and not gradient_dtype.is_floating_point:
            raise ValueError(
                f"gradient dtype must be floating point, not {gradient_dtype}"
            )
        if runtime == "fsdp":
            resolved_gradient_dtype = (
                model_dtype if gradient_dtype is None else gradient_dtype
            )
            if resolved_gradient_dtype != model_dtype:
                raise ValueError(
                    "the fsdp runtime accumulates and reduces gradients in "
                    f"{model_dtype}; got gradient_dtype={resolved_gradient_dtype}"
                )
            return cls(
                parameter_dtype=model_dtype,
                compute_dtype=model_dtype,
                gradient_dtype=resolved_gradient_dtype,
                master_dtype=torch.float32,
                artifact_dtype=model_dtype,
                buffer_dtype=(
                    None if sharding_strategy == "NO_SHARD" else torch.float32
                ),
            )
        if runtime == "fsdp2":
            resolved_gradient_dtype = (
                torch.float32 if gradient_dtype is None else gradient_dtype
            )
            if resolved_gradient_dtype not in (torch.float32, model_dtype):
                raise ValueError(
                    "the fsdp2 runtime supports fp32 or compute-dtype "
                    f"({model_dtype}) gradients; got gradient_dtype="
                    f"{resolved_gradient_dtype}"
                )
            return cls(
                parameter_dtype=torch.float32,
                compute_dtype=model_dtype,
                gradient_dtype=resolved_gradient_dtype,
                master_dtype=None,
                artifact_dtype=model_dtype,
                buffer_dtype=torch.float32,
            )
        raise ValueError(f"unknown runtime {runtime!r}; expected one of {RUNTIMES}")

    def to_record(self) -> Dict[str, Optional[str]]:
        """Plain string representation recorded in checkpoints for resume checks."""
        return {
            "parameter_dtype": str(self.parameter_dtype),
            "compute_dtype": str(self.compute_dtype),
            "gradient_dtype": str(self.gradient_dtype),
            "master_dtype": (
                None if self.master_dtype is None else str(self.master_dtype)
            ),
            "artifact_dtype": str(self.artifact_dtype),
            "buffer_dtype": (
                None if self.buffer_dtype is None else str(self.buffer_dtype)
            ),
        }


@dataclass(frozen=True)
class TrainingRuntime:
    """The data-parallel runtime that trains the draft, and its precision.

    Resolved once for a run and handed to the trainer, which builds that
    runtime; the runtime honours the precision or refuses it.
    """

    #: One of ``RUNTIMES``.
    name: str
    precision: PrecisionSpec

    @classmethod
    def resolve(
        cls,
        name: str,
        *,
        model_dtype: torch.dtype,
        sharding_strategy: Optional[str] = None,
        gradient_dtype: Optional[torch.dtype] = None,
    ) -> "TrainingRuntime":
        return cls(
            name,
            PrecisionSpec.resolve(
                runtime=name,
                model_dtype=model_dtype,
                sharding_strategy=sharding_strategy,
                gradient_dtype=gradient_dtype,
            ),
        )


@dataclass(frozen=True)
class PartitionHints:
    """What a runner that shards parameters has to keep together."""

    #: Class names of the modules that are each one indivisible unit. Empty:
    #: the program is one unit.
    blocks: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ProgramSpec:
    """What a program declares to the frame and a runner, as values."""

    #: The objective's name, for messages.
    name: str
    #: Batch tensors the program requires.
    inputs: Tuple[str, ...]
    #: Statistics it needs pooled over the whole window and every rank before
    #: its gradient pass.
    replay_statistics: Tuple[str, ...] = ()
    #: Additive statistics it emits during training forwards to be summed over
    #: the whole window and every rank and delivered through
    #: :meth:`TrainingProgram.apply_statistics` once the optimizer update
    #: succeeds (before any checkpoint snapshot).
    after_step_statistics: Tuple[str, ...] = ()
    #: Whether its evaluation reports the detailed metrics.
    evaluation_diagnostics: bool = False
    #: How the loss is normalized. "window": the update's objective is
    #: ``sum(numerator) / sum(denominator)`` over the window's micro-batches
    #: and the ranks. "microbatch": it is the mean over the micro-batches of
    #: ``sum_t c_t * sum(numerator_t) / sum(denominator_t)``, the inner sums
    #: over the ranks; the program returns the terms and the runner pools
    #: their denominators between each forward and its backward.
    reduction: Literal["window", "microbatch"] = "window"
    partition: PartitionHints = PartitionHints()
    #: Model-parallel axes the program communicates on, each with its counting
    #: kind: "copy" (same batch, count one rank), "part" (sharded sample, sum
    #: shards), or "own" (disjoint batches with sharded parameters, sum all).
    axes: Tuple[Tuple[str, Literal["copy", "part", "own"]], ...] = ()

    def __post_init__(self) -> None:
        overlap = sorted(set(self.replay_statistics) & set(self.after_step_statistics))
        if overlap:
            raise ValueError(
                "replay_statistics and after_step_statistics must be distinct; "
                f"overlap={overlap}"
            )


class TrainingProgram(Protocol):
    """A draft and the objective that drives it: what a runner executes.

    It supervises, runs the draft and forms the loss terms. It knows nothing
    of the data-parallel ranks, the optimizer or how it is wrapped. It is an
    ``nn.Module``: a runner wraps it, reads its parameters and their device
    and sets its mode. A runner calls the forward through its wrapper and the
    other methods on the program itself.
    """

    #: Frozen modules a runner neither shards nor optimizes (target tables,
    #: the teacher head), as declared when the program was built.
    frozen_modules: Tuple[torch.nn.Module, ...]

    #: The module's mode; a runner sets it with ``train()`` and ``eval()``.
    training: bool

    def train(self, mode: bool = True) -> "TrainingProgram":
        """Set the mode of the whole module, the draft included."""

    def eval(self) -> "TrainingProgram":
        """``train(False)``."""

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """The module's parameters; their device is where batches are placed."""

    def __call__(
        self,
        tensors: Mapping[str, torch.Tensor],
        *,
        step: int,
        total_steps: Optional[int],
        collect_detailed_metrics: bool,
        prepass: bool = False,
        **statistics: Optional[torch.Tensor],
    ) -> "ObjectiveResult":
        """Run the objective on one placed batch.

        The result's numerator and denominator are this rank's; under the
        ``window`` reduction the update's objective is their sums' ratio over
        the window and the ranks. Under ``microbatch`` a training call also
        returns ``result.terms``. No collective across samples is issued
        here. A program that declares a replay statistic also takes
        ``prepass=True``, which asks for this rank's sums of each statistic
        in ``result.statistics``, and in every call the keyword arguments
        that ``replay_arguments`` returned for it. A program that declares
        ``after_step_statistics`` returns each statistic's local sum in
        ``result.statistics`` on every training forward.
        """

    def replay_arguments(
        self, replay_values: Mapping[str, torch.Tensor], *, prepass: bool = False
    ) -> Mapping[str, Optional[torch.Tensor]]:
        """The forward's keyword arguments that carry the declared statistics.

        For a prepass, a replay (``replay_values`` holds the pooled sums) and
        any other call (it is empty). Any arithmetic on the pooled sums, a
        ratio and its zero guard for one, is done here in the precision the
        sums were pooled in, outside the wrapper. Empty for a program without
        a statistic.
        """

    def apply_statistics(
        self, values: Mapping[str, torch.Tensor]
    ) -> Mapping[str, Any]:
        """Apply pooled after-step statistics once an optimizer update succeeds.

        Called on every rank inside :meth:`StepRunner.apply_step`, after the
        optimizer and scheduler advance and before any checkpoint snapshot.
        Must issue no collective and compute identical buffer updates on every
        rank. Returns optional scalar post-step metrics to include in
        :class:`StepMetrics`.
        """

    def host_inputs(
        self,
        tensors: Mapping[str, torch.Tensor],
        *,
        sample_ids: Sequence[str],
        epoch: Optional[int],
    ) -> Mapping[str, torch.Tensor]:
        """Extra batch tensors derived on the host, before the batch is placed.

        ``sample_ids`` are the batch's, one per row; ``epoch`` is the one the
        run loop recorded on the batch (``EPOCH_METADATA_KEY``), or
        ``EVALUATION_EPOCH`` in an evaluation pass.
        """

    def kernel_record(self) -> Dict[str, str]:
        """The kernels realised so far, by the choice they answer."""

    def draft_state(self, state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """The draft's weights under canonical keys, from the program's state dict."""


@dataclass(frozen=True)
class CheckpointState:
    """What a runner hands over when a checkpoint is taken."""

    #: The draft's weights under canonical keys: complete on the first rank,
    #: empty on the others.
    draft: Dict[str, torch.Tensor]
    #: This rank's own state for continuing the run; every rank's is saved.
    rank: Dict[str, Any]
    #: The optimizer state where every rank holds the same one: saved once
    #: and handed back to every rank on resume. None where each rank's is its
    #: own, in ``rank``.
    replicated: Optional[Dict[str, Any]]
    #: The kernels this rank has realised so far, by the choice they answer.
    #: Ranks have to agree on them before a checkpoint records them.
    kernels: Mapping[str, str]


class RankFailure(RuntimeError):
    """This rank failed outside any collective, where its peers will succeed.

    Whoever catches it reports it through ``StepRunner.agree`` and then
    raises its cause, so the other ranks do not wait for this one in their
    next collective. A failure inside a collective is not one: the ranks are
    already apart there, and it propagates so the launcher tears the job down.
    """


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
        self, batch: "TrainBatch", ctx: StepContext, *, is_boundary: bool
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

    def prepass(self, batch: "TrainBatch", ctx: StepContext) -> Dict[str, torch.Tensor]:
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

    def evaluate(self, batch: "TrainBatch", ctx: StepContext) -> "ObjectiveResult":
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

    def checkpoint_state(self) -> CheckpointState:
        """The draft's weights and this rank's state for continuing the run.

        Gathering the weights is a collective when the wrapper shards. Legal
        between an applied step and the next window.
        """

    def load_resume(self, state: Mapping[str, Any]) -> None:
        """Restore the state a checkpoint saved for this rank.

        ``state`` is :attr:`CheckpointState.rank` as it was saved, with the
        replicated part filled in.
        """

    def agree(self, error: Optional[str], *, value: Any = None) -> None:
        """Have every rank learn every rank's outcome.

        One gather of an error and a value per rank. Raises on every rank when
        any rank reports an error or the values differ, so a rank-local
        failure never leaves the others waiting in a later collective. It is
        not for use inside a step.

        Args:
            error: This rank's failure, or None; any string is a failure, an
                empty one included.
            value: Plain data that ``==`` compares to a bool, the same on
                every rank when the ranks agree.
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
