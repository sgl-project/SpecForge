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


@dataclass(frozen=True)
class Checkpoint:
    """A saved training checkpoint location (resume target).

    Deliberately NOT a published "weight version" — the published-weight
    lifecycle (versioning, publisher, serving accept-length gate, hot update) is
    not yet implemented. This record only says where a checkpoint is and at what
    step.
    """

    checkpoint_uri: str
    global_step: int
    epoch: int
    strategy: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepResult:
    """Result of one TrainerCore step.

    ``optimizer_stepped`` is the authoritative grad-accumulation boundary signal —
    callers branch on it rather than sniffing the metrics dict.
    """

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
        return self._result(out, grad_norm, stepped)

    # NB there is deliberately no eval_step: evaluation goes through
    # ``Evaluator.run`` on raw ``strategy.forward_loss`` outputs, because correct
    # acc-len aggregation needs the per-position count tensors that ``_result``'s
    # scalarization strips.

    def _result(self, out: StepOutput, grad_norm, stepped: bool) -> StepResult:
        metrics: Dict[str, Any] = {"loss": _scalar(out.loss), "mode": "train"}
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
    """Lifecycle: fit / evaluate / checkpoint. Script becomes a launcher.

    Weight publishing + the serving accept-length gate are not yet implemented;
    save_checkpoint just persists training state and returns a Checkpoint.
    """

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
    ) -> None:
        self.core = core
        self.run_id = run_id
        self.output_dir = output_dir
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.log_interval = log_interval
        # Inject a configured CheckpointManager (rotation, best metric) or let
        # the lazy default own the plain layout — one configuration mechanism.
        self._checkpoint_mgr = checkpoint_manager
        self.max_steps = max_steps
        # Schedule horizon for step-dependent losses (Domino's lambda_base decay).
        # Distinct from max_steps (an optional early-stop CAP): a run may stop early
        # yet decay over the full planned length. Falls back to max_steps when unset;
        # if BOTH are None a schedule-dependent strategy sees total_steps=None and
        # decays nothing (StepContext-reading strategies must handle that).
        self.total_steps = total_steps if total_steps is not None else max_steps
        self.num_epochs = num_epochs
        self.logger = logger
        # ack_fn(sample_ids, global_step): acks consumed refs at the optimizer-step
        # boundary with the step number, so the controller records the durable
        # {acked, global_step, optimizer marker} transaction. If None, the loader
        # is assumed to ack (e.g. simple/equivalence runs).
        self.ack_fn = ack_fn
        # global_step counts OPTIMIZER steps (increments only at a grad-accum
        # boundary), so ack / checkpoint / resume semantics are in true optimizer
        # steps. micro_step counts forward/backward micro-batches.
        self.global_step = start_step
        self.micro_step = 0
        self.epoch = start_epoch
        # Resume position within the interrupted epoch: ``fit`` skips
        # ``start_batch`` batches of its FIRST epoch (via ``data.seek`` when the
        # stream supports it) so a resumed run continues on the data the
        # uninterrupted run would have seen — the plan's seek()-equivalent. The
        # position is also tracked (and persisted) in SAMPLES, which is
        # batch-size independent; the domain Trainer converts samples back to
        # batches and fails fast on a batch-size drift it cannot divide.
        self._start_batch = start_batch
        self._start_samples = start_samples
        self.epoch_batch = start_batch
        self.epoch_samples = start_samples
        self.last_metrics: Dict[str, Any] = {}

    def fit(
        self, data: Iterable[TrainBatch], eval_data: Optional[Iterable] = None
    ) -> int:
        module = self.core.strategy.trainable_module()
        module.train()
        pending_ack: List[str] = []
        for epoch in range(self.epoch, self.num_epochs):
            self.epoch = epoch
            if hasattr(data, "set_epoch"):
                data.set_epoch(epoch)
            # Resume mid-epoch: reposition the stream past the batches the
            # interrupted run already trained on. ``seek`` (FeatureDataLoader)
            # skips without materializing features; a plain iterable is drained.
            stream: Iterable[TrainBatch] = data
            skip, self._start_batch = self._start_batch, 0
            self.epoch_batch = skip
            self.epoch_samples, self._start_samples = self._start_samples, 0
            if skip:
                if hasattr(data, "seek"):
                    data.seek(skip)
                else:
                    it = iter(data)
                    for _ in itertools.islice(it, skip):
                        pass
                    stream = it
            for batch in stream:
                self.epoch_batch += 1
                self.epoch_samples += len(batch.sample_ids)
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
                eval_metrics: Optional[Dict[str, Any]] = None
                if (
                    self.eval_interval
                    and eval_data is not None
                    and self.global_step % self.eval_interval == 0
                ):
                    eval_metrics = self.evaluate(eval_data)
                    module.train()
                saved = False
                if self.save_interval and self.global_step % self.save_interval == 0:
                    self.save_checkpoint(self.global_step)
                    saved = True
                # Best tracking is part of checkpointing (save_interval > 0) but
                # not tied to cadence alignment: ANY eval that beats the record
                # persists a checkpoint (if this step isn't already saved) and
                # repoints ``best``. save_checkpoint bears collectives, so the
                # is-better verdict MUST be identical on every rank — best_score
                # is rehydrated from rank-local filesystem reads, so rank0 is
                # the single authority and its verdict is broadcast.
                if self.save_interval and eval_metrics is not None:
                    is_best = self._rank0_decision(
                        self._checkpoint_manager().is_better(eval_metrics)
                    )
                    if is_best:
                        if not saved:
                            self.save_checkpoint(self.global_step)
                        self._checkpoint_manager().update_best(
                            self.global_step, eval_metrics, force=True
                        )
                if self.max_steps is not None and self.global_step >= self.max_steps:
                    return self.global_step
        return self.global_step

    @torch.no_grad()
    def evaluate(self, data: Iterable[TrainBatch]) -> Dict[str, Any]:
        """Correct acceptance-length eval: per-position accuracy is aggregated over
        the whole pass (and across DP ranks) before the geometric sum, so the
        returned metrics are identical on every rank (see :class:`Evaluator`)."""
        from specforge.eval import Evaluator

        module = self.core.strategy.trainable_module()
        module.eval()
        return Evaluator().run(
            lambda batch: self.core.strategy.forward_loss(batch), data
        )

    @staticmethod
    def _rank0_decision(flag: bool) -> bool:
        """Make a collective-bearing branch decision identical on every rank.

        Rank0's verdict is broadcast; without this, a rank whose local
        filesystem view diverges (best_meta.json attribute-cache lag,
        node-local dirs) would enter or skip ``save_checkpoint``'s collectives
        alone and hang the group.
        """
        if (
            not torch.distributed.is_initialized()
            or torch.distributed.get_world_size() == 1
        ):
            return bool(flag)
        verdict = [bool(flag)]
        torch.distributed.broadcast_object_list(verdict, src=0)
        return bool(verdict[0])

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
                "epoch_batch": self.epoch_batch,
                "epoch_samples": self.epoch_samples,
                "strategy": self.core.strategy.name,
                "run_id": self.run_id,
                "world_size": (
                    torch.distributed.get_world_size()
                    if torch.distributed.is_initialized()
                    else 1
                ),
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
