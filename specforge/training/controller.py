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


def _dp_mean(x: Any) -> Any:
    """Average a scalar metric tensor across all ranks before it is logged.

    Uses the established DFlash metric convention (DP mean):
    without it the disagg consumer logs a single rank's local-batch accuracy
    (~1 rank x batch x anchors), which is ~sqrt(world) noisier and spikes because
    each rank's few round-robin refs can be all-easy or all-hard. Reducing across
    ranks recovers the ~world x larger effective sample the stock path logs.
    No-op when torch.distributed is unavailable / single-rank / x is not a
    tensor. Called every step on every rank, so the collective stays in lockstep.
    """
    import torch.distributed as dist

    if not isinstance(x, torch.Tensor):
        return x
    if not (dist.is_available() and dist.is_initialized()):
        return x
    world = dist.get_world_size()
    if world <= 1:
        return x
    x = x.detach().clone()
    dist.all_reduce(x)
    return x / world


def _synchronize_device() -> None:
    """Synchronize the active accelerator for opt-in timing diagnostics."""
    from specforge.utils import get_device_type

    device_type = get_device_type()
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize()


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
        import os as _os

        _P = int(_os.environ.get("PROFILE_STEPS", "0"))
        if _P:
            import time as _t

            _synchronize_device()
            _a0 = _t.perf_counter()
        out: StepOutput = self.strategy.forward_loss(batch, ctx)
        loss = out.loss / self.accumulation_steps
        self._micro += 1
        # The boundary is known before backward so the backend can defer the FSDP
        # gradient reduction (no_sync) on non-boundary micro-steps.
        stepped = self._micro % self.accumulation_steps == 0
        if _P:
            _synchronize_device()
            _a1 = _t.perf_counter()
        self.backend.backward(loss, is_boundary=stepped)
        if _P:
            _synchronize_device()
            _a2 = _t.perf_counter()
        grad_norm = self.backend.step() if stepped else None
        if _P:
            _synchronize_device()
            _a3 = _t.perf_counter()
            acc = getattr(self, "_prof2", None)
            if acc is None:
                acc = {
                    "fwd": 0.0,
                    "bwd": 0.0,
                    "opt": 0.0,
                    "n": 0,
                    "B": 0,
                    "padT": 0,
                    "useful": 0.0,
                }
                self._prof2 = acc
            acc["fwd"] += _a1 - _a0
            acc["bwd"] += _a2 - _a1
            acc["opt"] += _a3 - _a2
            acc["n"] += 1
            try:
                _tt = batch.tensors
                _lm = _tt.get("loss_mask")
                _ref = _lm if _lm is not None else _tt.get("input_ids")
                if _ref is not None and _ref.dim() >= 2:
                    acc["B"] += int(_ref.shape[0])
                    acc["padT"] += int(_ref.shape[-1])
                if _lm is not None:
                    acc["useful"] += float(_lm.sum().item())
            except Exception:
                pass
            if acc["n"] >= _P:
                _n = acc["n"]
                print(
                    f"[profile2] n={_n} fwd={acc['fwd']/_n*1000:.1f}ms "
                    f"bwd={acc['bwd']/_n*1000:.1f}ms opt={acc['opt']/_n*1000:.1f}ms | "
                    f"avgB={acc['B']/_n:.1f} avgPadT={acc['padT']/_n:.0f} "
                    f"avgUsefulTok={acc['useful']/_n:.0f}",
                    flush=True,
                )
                self._prof2 = None
        return self._result(out, grad_norm, stepped)

    def _result(self, out: StepOutput, grad_norm, stepped: bool) -> StepResult:
        # DP-average loss AND accuracy across ranks before logging, matching stock
        # DFlash reports a data-parallel mean for loss and accuracy.
        metrics: Dict[str, Any] = {"loss": _scalar(_dp_mean(out.loss))}
        structured_metric_keys = ("acces", "acceptance_rates", "plosses")
        for key in structured_metric_keys:
            if key in out.metrics:
                metrics[key.rstrip("es") if key == "acces" else key] = _scalar(
                    out.metrics[key]
                )
        if "accuracy" in out.metrics:
            # DP-average the accuracy across ranks.
            # so the logged curve is smooth, not a single rank's noisy local batch.
            metrics["acc"] = _scalar(_dp_mean(out.metrics["accuracy"]))
        # Strategies may expose additional scalar diagnostics without teaching
        # the generic trainer their algorithm-specific names. Move CPU schedule
        # scalars (for example Domino's lambda_base) onto the loss device before
        # the DP reduction so NCCL-backed runs do not all-reduce a CPU tensor.
        reserved_metric_keys = set(structured_metric_keys) | {"accuracy", "loss"}
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
            metrics[key] = _scalar(_dp_mean(scalar))
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
        env_steps = int(os.environ.get("PROFILE_TORCH", "0"))
        trace_path = None
        if env_steps > 0 and not options.enabled:
            options = ProfilingOptions(
                enabled=True,
                start_step=int(os.environ.get("PROFILE_TORCH_WARMUP", "40")),
                num_steps=env_steps,
                record_shapes=True,
            )
            trace_path = os.environ.get("PROFILE_TORCH_TRACE") or None
        self._step_profiler = StepProfiler(
            options,
            output_dir=output_dir,
            trace_path=trace_path,
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
        import time as _time

        _PROFILE = int(os.environ.get("PROFILE_STEPS", "0"))
        _prof = {"data": 0.0, "step": 0.0, "n": 0}
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
                if _PROFILE:
                    _synchronize_device()
                    _pt0 = _time.perf_counter()
                try:
                    batch = next(_it)
                except StopIteration:
                    break
                if _PROFILE:
                    _synchronize_device()
                    _pt1 = _time.perf_counter()
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
                if _PROFILE:
                    _synchronize_device()
                    _pt2 = _time.perf_counter()
                    _prof["data"] += _pt1 - _pt0
                    _prof["step"] += _pt2 - _pt1
                    _prof["n"] += 1
                    if _prof["n"] >= _PROFILE:
                        _d = _prof["data"] / _prof["n"] * 1000
                        _s = _prof["step"] / _prof["n"] * 1000
                        _tot = _prof["data"] + _prof["step"]
                        print(
                            f"[profile] microsteps={_prof['n']} "
                            f"data_wait={_d:.1f}ms compute={_s:.1f}ms "
                            f"data_frac={100 * _prof['data'] / _tot:.0f}% "
                            f"step_total={_d + _s:.1f}ms",
                            flush=True,
                        )
                        _prof["data"] = 0.0
                        _prof["step"] = 0.0
                        _prof["n"] = 0
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
        # Every rank participates: the FSDP FULL_STATE_DICT gather is a
        # collective, and the optimizer/RNG parts are rank-local so every rank
        # persists its own (the manager writes the shared payload on rank0 only).
        full = self.core.backend.state_dict()
        mgr = self._checkpoint_manager()
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
        ckpt_dir = mgr.save(
            shared,
            step,
            rank_state={"optimizer": full["optimizer"], "rng": full["rng"]},
        )
        self.last_checkpoint_step = step
        return Checkpoint(
            checkpoint_uri=f"file://{os.path.abspath(ckpt_dir)}",
            global_step=step,
            epoch=self.epoch,
            strategy=self.core.strategy.name,
        )


__all__ = ["TrainerCore", "TrainerController", "Checkpoint", "StepResult"]
