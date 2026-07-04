# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""TrainerCore + TrainerController: the trainer-boundary split.

* ``TrainerCore`` owns exactly one train/eval step plus the grad-accumulation and
  optimizer boundary. It is **branch-free**: it never inspects online/offline or
  ``target_repr`` and never applies a projection — that is the strategy's job. It
  consumes a normalized ``TrainBatch`` and delegates the forward/loss to the
  strategy and the backward/step to the backend.
* ``TrainerController`` owns the lifecycle: ``fit`` / ``evaluate`` /
  ``save_checkpoint`` / weight publication. The training *script* becomes a thin
  launcher that builds these and calls ``fit``.

EAGLE3 and DFlash share this lifecycle unchanged — only the strategy differs.
"""

from __future__ import annotations

import itertools
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch

from specforge.runtime.contracts import TrainBatch
from specforge.runtime.training.backend import TrainingBackend
from specforge.runtime.training.strategy import (
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
        return self._result(out, grad_norm, stepped, mode="train")

    @torch.no_grad()
    def eval_step(
        self, batch: TrainBatch, ctx: Optional[StepContext] = None
    ) -> StepResult:
        out: StepOutput = self.strategy.forward_loss(batch, ctx)
        return self._result(out, None, False, mode="eval")

    def _result(
        self, out: StepOutput, grad_norm, stepped: bool, mode: str
    ) -> StepResult:
        metrics: Dict[str, Any] = {"loss": _scalar(out.loss), "mode": mode}
        for key in ("acces", "acceptance_rates", "plosses"):
            if key in out.metrics:
                metrics[key.rstrip("es") if key == "acces" else key] = _scalar(
                    out.metrics[key]
                )
        if "accuracy" in out.metrics:
            metrics["acc"] = _scalar(out.metrics["accuracy"])
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
    """Lifecycle: fit / evaluate / checkpoint. The training script becomes a
    launcher; weight publishing is not implemented — ``save_checkpoint`` persists
    resume state and returns a :class:`Checkpoint`."""

    def __init__(
        self,
        core: TrainerCore,
        *,
        run_id: str,
        output_dir: str = "./output",
        save_interval: int = 0,
        eval_interval: int = 0,
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
        checkpoint_manager: Optional[Any] = None,
        checkpoint_extra: Optional[Dict[str, Any]] = None,
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
        self.last_metrics: Dict[str, Any] = {}

    def fit(
        self, data: Iterable[TrainBatch], eval_data: Optional[Iterable] = None
    ) -> int:
        if self.max_steps is not None and self.global_step >= self.max_steps:
            logger.info(
                "fit: global_step=%d already at max_steps=%d; nothing to train",
                self.global_step,
                self.max_steps,
            )
            return self.global_step
        module = self.core.strategy.trainable_module()
        module.train()
        pending_ack: List[str] = []
        for epoch in range(self.epoch, self.num_epochs):
            self.epoch = epoch
            if hasattr(data, "set_epoch"):
                data.set_epoch(epoch)
            stream: Iterable[TrainBatch] = data
            skip = self._epoch_batch
            if skip:
                if hasattr(data, "seek"):
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
            for batch in stream:
                self._epoch_batch += 1
                self._epoch_samples += len(batch.sample_ids)
                self.micro_step += 1
                if self.ack_fn is not None:
                    pending_ack.extend(batch.sample_ids)
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
                if self.ack_fn is not None:
                    # durable ack transaction at the optimizer-step boundary
                    self.ack_fn(pending_ack, self.global_step)
                    pending_ack = []
                if self.logger and self.global_step % max(1, self.log_interval) == 0:
                    self.logger(result.metrics, self.global_step)
                if (
                    self.eval_interval
                    and eval_data is not None
                    and self.global_step % self.eval_interval == 0
                ):
                    self.evaluate(eval_data)
                    module.train()
                if self.save_interval and self.global_step % self.save_interval == 0:
                    self.save_checkpoint(self.global_step)
                if self.max_steps is not None and self.global_step >= self.max_steps:
                    return self.global_step
            self._epoch_batch = 0
            self._epoch_samples = 0
        return self.global_step

    @torch.no_grad()
    def evaluate(self, data: Iterable[TrainBatch]) -> Dict[str, float]:
        module = self.core.strategy.trainable_module()
        module.eval()
        agg: Dict[str, list] = {}
        n = 0
        for batch in data:
            rep = self.core.eval_step(batch)
            n += 1
            for k, v in rep.metrics.items():
                if isinstance(v, (int, float)):
                    agg.setdefault(k, []).append(v)
        return {k: sum(vs) / len(vs) for k, vs in agg.items() if vs}

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
        return Checkpoint(
            checkpoint_uri=f"file://{os.path.abspath(ckpt_dir)}",
            global_step=step,
            epoch=self.epoch,
            strategy=self.core.strategy.name,
        )


__all__ = ["TrainerCore", "TrainerController", "Checkpoint", "StepResult"]
