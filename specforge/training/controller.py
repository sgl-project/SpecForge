# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""TrainerCore + TrainerController: the trainer-boundary split.

``TrainerCore`` runs exactly one branch-free step (strategy forward/loss, backend
backward/step) plus the grad-accumulation boundary. ``TrainerController`` owns
the lifecycle: fit / evaluate / save_checkpoint. EAGLE3 and DFlash share this
unchanged — only the strategy differs.
"""

from __future__ import annotations

import itertools
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch

from specforge.runtime.contracts import TrainBatch
from specforge.training.backend import TrainingBackend
from specforge.training.strategies.base import (
    DraftTrainStrategy,
    StepContext,
    StepOutput,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Checkpoint:
    """A saved training checkpoint location (resume target) — deliberately NOT a
    published weight version (weight publication is not implemented)."""

    checkpoint_uri: str
    global_step: int
    epoch: int
    strategy: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepResult:
    """Result of one TrainerCore step; ``optimizer_stepped`` is the authoritative
    grad-accumulation boundary signal."""

    optimizer_stepped: bool
    loss: float
    grad_norm: Optional[float]
    metrics: Dict[str, Any] = field(default_factory=dict)


def _scalar(x: Any) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().float().mean().item())
    if isinstance(x, (list, tuple)) and x:
        return float(torch.stack([t.detach().float() for t in x]).mean().item())
    return float(x)


def _dp_mean_scalars(values: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Average scalar metrics across DP ranks with one collective.

    Uses the established DFlash metric convention (DP mean):
    without it the disagg consumer logs a single rank's local-batch accuracy
    (~1 rank x batch x anchors), which is ~sqrt(world) noisier and spikes because
    each rank's few round-robin refs can be all-easy or all-hard. Reducing across
    ranks recovers the ~world x larger effective sample the stock path logs.
    The caller invokes this only at an optimizer boundary: intermediate
    micro-step metrics are not logged, so synchronizing them only adds latency.
    """
    import torch.distributed as dist

    if not values or not (dist.is_available() and dist.is_initialized()):
        return values
    world = dist.get_world_size()
    if world <= 1:
        return values
    names = list(values)
    packed = torch.stack(
        [values[name].detach().float().reshape(()) for name in names]
    )
    dist.all_reduce(packed)
    packed /= world
    return {name: packed[index] for index, name in enumerate(names)}


_EAGLE3_STRUCTURED_METRIC_KEYS = frozenset(
    {
        "acces",
        "acceptance_rates",
        "plosses",
        "acc_corrects",
        "acc_denoms",
        "metric_losses",
        "metric_loss_denoms",
    }
)


def _metric_vector(values: Any, *, device: torch.device, name: str) -> torch.Tensor:
    """Normalize one per-TTT metric sequence without losing its positions."""
    if isinstance(values, torch.Tensor):
        vector = values.detach().flatten()
    elif isinstance(values, (list, tuple)):
        vector = torch.stack(
            [torch.as_tensor(value).detach().reshape(()) for value in values]
        )
    else:
        raise TypeError(f"{name} must be a tensor or sequence, got {type(values)!r}")
    if vector.numel() == 0:
        raise ValueError(f"{name} must contain at least one TTT position")
    return vector.to(device=device, dtype=torch.float32)


def _reduce_eagle3_metrics(
    raw: Dict[str, Any],
    *,
    device: torch.device,
    process_group: Any,
    ploss_decay: float,
) -> Optional[Dict[str, float]]:
    """Reduce EAGLE3's per-position training telemetry as numerators/counts.

    Accuracy and p-loss are ratios, so averaging rank-local ratios biases the
    result whenever ranks carry different token counts.  Pack every numerator
    and denominator into one collective and form ratios only after the global
    SUM.  Acceptance rate has no separate count in the model contract; weight
    it by the corresponding p-loss token count, matching the evaluator's
    batch-size-invariant convention.
    """
    required = {
        "acc_corrects",
        "acc_denoms",
        "metric_losses",
        "metric_loss_denoms",
    }
    if not required.issubset(raw):
        return None

    corrects = _metric_vector(raw["acc_corrects"], device=device, name="acc_corrects")
    acc_denoms = _metric_vector(raw["acc_denoms"], device=device, name="acc_denoms")
    losses = _metric_vector(raw["metric_losses"], device=device, name="metric_losses")
    loss_denoms = _metric_vector(
        raw["metric_loss_denoms"], device=device, name="metric_loss_denoms"
    )
    length = corrects.numel()
    vectors = {
        "acc_denoms": acc_denoms,
        "metric_losses": losses,
        "metric_loss_denoms": loss_denoms,
    }
    acceptance_rates = None
    if "acceptance_rates" in raw:
        acceptance_rates = _metric_vector(
            raw["acceptance_rates"], device=device, name="acceptance_rates"
        )
        vectors["acceptance_rates"] = acceptance_rates
    mismatched = {
        name: value.numel()
        for name, value in vectors.items()
        if value.numel() != length
    }
    if mismatched:
        raise ValueError(
            "EAGLE3 structured metric lengths must match acc_corrects "
            f"({length}); got {mismatched}"
        )

    # Rows: accuracy numerator/denominator, p-loss numerator/denominator,
    # acceptance numerator/denominator.  The last two rows stay zero when the
    # strategy does not expose acceptance telemetry.
    packed = torch.stack(
        (
            corrects,
            acc_denoms,
            losses * loss_denoms,
            loss_denoms,
            (
                acceptance_rates * loss_denoms
                if acceptance_rates is not None
                else torch.zeros_like(loss_denoms)
            ),
            (
                loss_denoms
                if acceptance_rates is not None
                else torch.zeros_like(loss_denoms)
            ),
        )
    )

    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        world = dist.get_world_size(process_group)
        if world > 1:
            dist.all_reduce(packed, op=dist.ReduceOp.SUM, group=process_group)

    reduced_acc = packed[0] / packed[1].clamp_min(1e-6)
    reduced_ploss = packed[2] / packed[3].clamp_min(1e-6)
    result: Dict[str, float] = {}
    for index, value in enumerate(reduced_acc.tolist()):
        result[f"acc_{index}"] = float(value)
    for index, value in enumerate(reduced_ploss.tolist()):
        result[f"ploss_{index}"] = float(value)

    result["acc"] = float(packed[0].sum().div(packed[1].sum().clamp_min(1e-6)).item())
    weights = torch.tensor(
        [ploss_decay**index for index in range(length)],
        dtype=reduced_ploss.dtype,
        device=reduced_ploss.device,
    )
    result["loss"] = float((reduced_ploss * weights).sum().item())

    if acceptance_rates is not None:
        reduced_acceptance = packed[4] / packed[5].clamp_min(1e-6)
        for index, value in enumerate(reduced_acceptance.tolist()):
            result[f"acceptance_rate_{index}"] = float(value)
        result["acceptance_rate"] = float(reduced_acceptance.mean().item())
    return result


class TrainerCore:
    """One step: forward/loss (strategy) -> backward (backend) -> optimizer boundary."""

    def __init__(
        self,
        strategy: DraftTrainStrategy,
        backend: TrainingBackend,
        *,
        accumulation_steps: int = 1,
    ) -> None:
        self.strategy = strategy
        self.backend = backend
        self.accumulation_steps = max(1, accumulation_steps)
        self._micro = 0

    @property
    def accumulation_remainder(self) -> int:
        """Micro-batches whose gradients have not reached an optimizer step."""
        return self._micro % self.accumulation_steps

    def train_step(
        self, batch: TrainBatch, ctx: Optional[StepContext] = None
    ) -> StepResult:
        out: StepOutput = self.strategy.forward_loss(batch, ctx)
        loss = out.loss / self.accumulation_steps
        self._micro += 1
        # The boundary is known before backward so the backend can defer the FSDP
        # gradient reduction (no_sync) on non-boundary micro-steps.
        stepped = self._micro % self.accumulation_steps == 0
        self.backend.backward(loss, is_boundary=stepped)
        grad_norm = self.backend.step() if stepped else None
        return self._result(out, grad_norm, stepped)

    def _result(self, out: StepOutput, grad_norm, stepped: bool) -> StepResult:
        # EAGLE3 carries per-TTT numerators and denominators.  Preserve those
        # positions and reduce counts before ratios; scalarizing its lists here
        # would both collapse the TTT structure and log one rank's local data.
        metric_device = (
            out.loss.device
            if isinstance(out.loss, torch.Tensor)
            else torch.device("cpu")
        )
        parallel_config = getattr(self.backend, "parallel_config", None)
        process_group = getattr(parallel_config, "fsdp_process_group", None)
        structured = _reduce_eagle3_metrics(
            out.metrics,
            device=metric_device,
            process_group=process_group,
            ploss_decay=float(getattr(self.strategy, "ploss_decay", 1.0)),
        )
        # Structured EAGLE3 metrics are already globally reduced.  Remaining
        # scalar diagnostics are DP-averaged in a single collective at optimizer
        # boundaries; non-boundary results stay rank-local.
        metrics: Dict[str, Any] = dict(structured or {})
        scalar_metrics: Dict[str, torch.Tensor] = {}
        if "loss" not in metrics:
            scalar_metrics["loss"] = out.loss
        if "accuracy" in out.metrics:
            accuracy = out.metrics["accuracy"]
            if isinstance(accuracy, torch.Tensor):
                scalar_metrics["acc"] = accuracy.detach().float().mean().to(
                    out.loss.device
                )
            elif isinstance(accuracy, (int, float)) and not isinstance(
                accuracy, bool
            ):
                scalar_metrics["acc"] = out.loss.detach().new_tensor(float(accuracy))
        # Strategies may expose additional scalar diagnostics without teaching
        # the generic trainer their algorithm-specific names. Move CPU schedule
        # scalars (for example Domino's lambda_base) onto the loss device before
        # the DP reduction so NCCL-backed runs do not all-reduce a CPU tensor.
        reserved_metric_keys = _EAGLE3_STRUCTURED_METRIC_KEYS | {"accuracy", "loss"}
        for key, value in out.metrics.items():
            if key in reserved_metric_keys:
                continue
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue
                scalar = value.detach().reshape(()).to(out.loss.device)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                scalar = out.loss.detach().new_tensor(float(value))
            else:
                continue
            scalar_metrics[key] = scalar
        if stepped:
            scalar_metrics = _dp_mean_scalars(scalar_metrics)
        metrics.update({key: _scalar(value) for key, value in scalar_metrics.items()})
        gn = _scalar(grad_norm) if grad_norm is not None else None
        if gn is not None:
            metrics["grad_norm"] = gn
        return StepResult(
            optimizer_stepped=stepped,
            loss=metrics["loss"],
            grad_norm=gn,
            metrics=metrics,
        )


class TrainerController:
    """Lifecycle: fit / evaluate / checkpoint.

    ``save_checkpoint`` persists resumable draft state and returns a
    :class:`Checkpoint`; ``specforge export`` materializes that state into the
    serving or Hugging Face model format.  Evaluation is configured once at
    construction time, so the public training lifecycle remains one no-argument
    :meth:`Trainer.fit` call.
    """

    def __init__(
        self,
        core: TrainerCore,
        *,
        run_id: str,
        output_dir: str = "./output",
        save_interval: int = 0,
        eval_interval: int = 0,
        eval_data_factory: Optional[
            Callable[[], Optional[Iterable[TrainBatch]]]
        ] = None,
        log_interval: int = 50,
        max_steps: Optional[int] = None,
        total_steps: Optional[int] = None,
        num_epochs: int = 1,
        logger: Optional[Callable[[Dict[str, Any], int], None]] = None,
        ack_fn: Optional[Callable[[List[str], int], None]] = None,
        start_step: int = 0,
        start_epoch: int = 0,
        start_batch: int = 0,
        start_samples: int = 0,
        data_prepositioned: bool = False,
        checkpoint_manager: Optional[Any] = None,
        checkpoint_extra: Optional[Dict[str, Any]] = None,
        profiling_options=None,
    ) -> None:
        if (start_batch == 0) != (start_samples == 0):
            raise ValueError(
                f"start_batch={start_batch} and start_samples={start_samples} "
                f"describe the same mid-epoch position and must be zero or "
                f"nonzero together"
            )
        self.core = core
        self.run_id = run_id
        self.output_dir = output_dir
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.eval_data_factory = eval_data_factory
        self.log_interval = log_interval
        # Injected manager (rotation, best metric) or the lazy default layout.
        self._checkpoint_mgr = checkpoint_manager
        # Extra entries merged into the shared checkpoint payload at save
        # (e.g. dataset_size / accumulation_steps, validated on resume).
        self.checkpoint_extra = dict(checkpoint_extra or {})
        self.max_steps = max_steps
        # Schedule horizon for step-dependent losses (Domino's lambda_base decay);
        # distinct from max_steps, an optional early-stop CAP. Falls back to
        # max_steps; None means schedule-reading strategies decay nothing.
        self.total_steps = total_steps if total_steps is not None else max_steps
        self.num_epochs = num_epochs
        self.logger = logger
        # ack_fn(sample_ids, global_step) records the durable ack transaction at
        # the optimizer-step boundary; None = the loader acks (simple runs).
        self.ack_fn = ack_fn
        # global_step counts OPTIMIZER steps (increments only at a grad-accum
        # boundary) so ack/checkpoint/resume semantics are in true optimizer
        # steps; micro_step counts forward/backward micro-batches.
        self.global_step = start_step
        self.micro_step = 0
        self.epoch = start_epoch
        # Live position within the current epoch, in batches and in SAMPLES
        # (batch-size independent, the persisted form). Nonzero at an epoch start
        # — seeded here on resume, or left over from a mid-epoch max_steps return
        # — makes fit() skip that prefix instead of re-training it.
        self._epoch_batch = start_batch
        self._epoch_samples = start_samples
        # A resumed online queue is rebuilt from the deterministic prompt plan
        # with its trained prefix already removed. Suppress the generic iterable
        # seek exactly once; consuming ``start_batch`` again would skip fresh data.
        self._data_prepositioned = bool(data_prepositioned)
        self.last_metrics: Dict[str, Any] = {}
        self.last_checkpoint_step: Optional[int] = None
        from specforge.training.profiling import ProfilingOptions, StepProfiler

        options = profiling_options or ProfilingOptions()
        self._step_profiler = StepProfiler(
            options,
            output_dir=output_dir,
        )

    def fit(self, data: Iterable[TrainBatch]) -> int:
        if self.max_steps is not None and self.global_step >= self.max_steps:
            logger.info(
                "fit: global_step=%d already at max_steps=%d; nothing to train",
                self.global_step,
                self.max_steps,
            )
            return self.global_step
        module = self.core.strategy.trainable_module()
        module.train()
        # Rank0-broadcast once: rank-local assembly must not let ranks enter or
        # skip the evaluator's collectives independently.
        eval_enabled = self._rank0_decision(
            self.eval_interval > 0 and self.eval_data_factory is not None
        )
        pending_ack: List[str] = []
        for epoch in range(self.epoch, self.num_epochs):
            self.epoch = epoch
            if hasattr(data, "set_epoch"):
                data.set_epoch(epoch)
            stream: Iterable[TrainBatch] = data
            skip = self._epoch_batch
            if skip:
                if self._data_prepositioned:
                    self._data_prepositioned = False
                elif hasattr(data, "seek"):
                    data.seek(skip)
                else:
                    it = iter(data)
                    consumed = sum(1 for _ in itertools.islice(it, skip))
                    if consumed < skip:
                        raise ValueError(
                            f"resume position skips past the end of the data: "
                            f"epoch {epoch} yielded only {consumed} batches, "
                            f"cannot skip {skip}"
                        )
                    stream = it
            _it = iter(stream)
            while True:
                try:
                    batch = next(_it)
                except StopIteration:
                    break
                self._epoch_batch += 1
                self._epoch_samples += len(batch.sample_ids)
                self.micro_step += 1
                if self.ack_fn is not None:
                    pending_ack.extend(batch.sample_ids)
                self._step_profiler.before_micro_step(self.global_step)
                result = self.core.train_step(
                    batch,
                    ctx=StepContext(
                        global_step=self.global_step, total_steps=self.total_steps
                    ),
                )
                self.last_metrics = result.metrics
                # grad accumulated but optimizer has not stepped yet; everything
                # keyed on optimizer steps fires only at the boundary.
                if not result.optimizer_stepped:
                    continue
                self.global_step += 1
                self._step_profiler.after_optimizer_step(self.global_step)
                if self.ack_fn is not None:
                    # durable ack transaction at the optimizer-step boundary
                    self.ack_fn(pending_ack, self.global_step)
                    pending_ack = []
                if self.logger and self.global_step % max(1, self.log_interval) == 0:
                    log_metrics = dict(result.metrics)
                    optimizer = getattr(self.core.backend, "optimizer", None)
                    get_learning_rate = getattr(optimizer, "get_learning_rate", None)
                    if callable(get_learning_rate):
                        log_metrics["lr"] = float(get_learning_rate())
                    self.logger(log_metrics, self.global_step)
                eval_metrics: Optional[Dict[str, Any]] = None
                if eval_enabled and self.global_step % self.eval_interval == 0:
                    eval_metrics = self.evaluate_configured()
                    module.train()
                    if eval_metrics:
                        if self.logger:
                            self.logger(eval_metrics, self.global_step)
                        self.last_metrics = {**self.last_metrics, **eval_metrics}
                # ``is_better`` is collective (rank0 verdict broadcast inside
                # the manager); its guard is rank-identical because eval metrics
                # are DP-reduced. Empty eval metrics skip best tracking.
                interval_hit = bool(
                    self.save_interval and self.global_step % self.save_interval == 0
                )
                is_best = bool(
                    eval_metrics and self._checkpoint_manager().is_better(eval_metrics)
                )
                if interval_hit or is_best:
                    self.save_checkpoint(self.global_step)
                if is_best:
                    self._checkpoint_manager().update_best(
                        self.global_step, eval_metrics
                    )
                if self.max_steps is not None and self.global_step >= self.max_steps:
                    return self.global_step
            self._epoch_batch = 0
            self._epoch_samples = 0
            # Persist the *next* epoch after a naturally exhausted pass.  A
            # checkpoint taken after fit() returns must describe completed
            # work, not epoch ``N`` at batch zero (which would replay that
            # entire epoch on resume).
            self.epoch = epoch + 1
        remainder = self.core.accumulation_remainder
        if remainder:
            raise RuntimeError(
                "training stream ended with incomplete gradient accumulation: "
                f"received {remainder} of {self.core.accumulation_steps} "
                "micro-batches after the last optimizer step; no partial "
                "optimizer step or durable acknowledgement was committed"
            )
        return self.global_step

    def close_profiler(self) -> None:
        """Finalize a partial profiling window on every training exit path."""
        try:
            self._step_profiler.close(self.global_step)
        except Exception:
            logger.exception("failed to finalize the training profiler")

    def evaluate_configured(self) -> Dict[str, Any]:
        """Build one fresh eval pass and close any managed capture stream.

        Fixed offline loaders may simply be returned on every call. Online eval
        factories can return an iterable context manager so each interval gets
        a fresh rollout stream without exposing an extra argument on ``fit``.
        """
        if self.eval_data_factory is None:
            return self.evaluate(None)
        data = self.eval_data_factory()
        if data is None or not hasattr(data, "__enter__"):
            return self.evaluate(data)
        with data as entered:
            return self.evaluate(data if entered is None else entered)

    @torch.no_grad()
    def evaluate(self, data: Optional[Iterable[TrainBatch]]) -> Dict[str, Any]:
        """Full-pass eval via :class:`Evaluator`.

        Returns rank-identical ``eval/*`` metrics, or ``{}`` when zero batches
        were processed globally. ``data=None`` (an empty local shard) still
        joins the evaluator's collectives.
        """
        from specforge.eval import Evaluator

        module = self.core.strategy.trainable_module()
        was_training = module.training
        module.eval()
        # Use the train path's live context so schedule-dependent losses (for
        # example Domino's lambda_base) are not evaluated as if at step zero.
        ctx = StepContext(global_step=self.global_step, total_steps=self.total_steps)
        try:
            return Evaluator().run(
                lambda batch: self.core.strategy.forward_loss(batch, ctx), data
            )
        finally:
            module.train(was_training)

    @staticmethod
    def _rank0_decision(flag: bool) -> bool:
        """Broadcast rank0's verdict for a collective-bearing branch."""
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or torch.distributed.get_world_size() == 1
        ):
            return bool(flag)
        box = [bool(flag)] if torch.distributed.get_rank() == 0 else [False]
        torch.distributed.broadcast_object_list(box, src=0)
        return bool(box[0])

    def _checkpoint_manager(self):
        # Lazily built in its S-home so the runtime seam does not import the domain
        # layer at module load (mirrors _assemble_trainer's lazy Trainer import).
        if self._checkpoint_mgr is None:
            from specforge.training.checkpoint import CheckpointManager

            self._checkpoint_mgr = CheckpointManager(self.output_dir, self.run_id)
        return self._checkpoint_mgr

    def save_checkpoint(self, step: int) -> Checkpoint:
        # Every rank participates: FSDP model gathering is collective and every
        # rank persists its RNG. Sharded optimizer state stays rank-local; the
        # identical DDP optimizer is written once in the shared rank0 payload.
        full = self.core.backend.state_dict()
        mgr = self._checkpoint_manager()
        replicated_optimizer = bool(
            getattr(self.core.backend, "optimizer_state_is_replicated", False)
        )
        shared = None
        if mgr.is_rank0():
            shared = {
                "draft_state_dict": self.core.strategy.checkpoint_state_filter(
                    full["model"]
                ),
                "global_step": step,
                "epoch": self.epoch,
                "epoch_batch": self._epoch_batch,
                "epoch_samples": self._epoch_samples,
                "strategy": self.core.strategy.name,
                "run_id": self.run_id,
                "world_size": (
                    torch.distributed.get_world_size()
                    if torch.distributed.is_initialized()
                    else 1
                ),
                **self.checkpoint_extra,
            }
            if replicated_optimizer:
                # DDP ranks have identical parameters, gradients, optimizer
                # moments, and scheduler state. Persist that large state once;
                # rank files still preserve their distinct RNG streams.
                shared["replicated_optimizer_state"] = full["optimizer"]
        ckpt_dir = mgr.save(
            shared,
            step,
            rank_state={
                "optimizer": None if replicated_optimizer else full["optimizer"],
                "rng": full["rng"],
            },
        )
        self.last_checkpoint_step = step
        return Checkpoint(
            checkpoint_uri=f"file://{os.path.abspath(ckpt_dir)}",
            global_step=step,
            epoch=self.epoch,
            strategy=self.core.strategy.name,
        )


__all__ = ["TrainerCore", "TrainerController", "Checkpoint", "StepResult"]
