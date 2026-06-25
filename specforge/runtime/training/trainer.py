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

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch

from specforge.runtime.contracts import TrainBatch
from specforge.runtime.training.backend import TrainingBackend
from specforge.runtime.training.strategy import DraftTrainStrategy, StepOutput


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

    def train_step(self, batch: TrainBatch) -> StepResult:
        out: StepOutput = self.strategy.forward_loss(batch)
        loss = out.loss / self.accumulation_steps
        self.backend.backward(loss)
        self._micro += 1
        grad_norm = None
        stepped = self._micro % self.accumulation_steps == 0
        if stepped:
            grad_norm = self.backend.step()
        return self._result(out, grad_norm, stepped, mode="train")

    @torch.no_grad()
    def eval_step(self, batch: TrainBatch) -> StepResult:
        out: StepOutput = self.strategy.forward_loss(batch)
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
        num_epochs: int = 1,
        logger: Optional[Callable[[Dict[str, Any], int], None]] = None,
        ack_fn: Optional[Callable[[List[str], int], None]] = None,
        start_step: int = 0,
        start_epoch: int = 0,
    ) -> None:
        self.core = core
        self.run_id = run_id
        self.output_dir = output_dir
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.log_interval = log_interval
        self.max_steps = max_steps
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
            for batch in data:
                self.micro_step += 1
                if self.ack_fn is not None:
                    pending_ack.extend(batch.sample_ids)
                result = self.core.train_step(batch)
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

    def save_checkpoint(self, step: int) -> Checkpoint:
        ckpt_dir = os.path.join(self.output_dir, f"{self.run_id}-step{step}")
        is_rank0 = (
            not torch.distributed.is_initialized()
        ) or torch.distributed.get_rank() == 0
        full_state = self.core.backend.state_dict()
        draft_state = self.core.strategy.checkpoint_state_filter(full_state)
        if is_rank0:
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(
                {
                    "draft_state_dict": draft_state,
                    "global_step": step,
                    "epoch": self.epoch,
                    "strategy": self.core.strategy.name,
                    "run_id": self.run_id,
                },
                os.path.join(ckpt_dir, "training_state.pt"),
            )
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        return Checkpoint(
            checkpoint_uri=f"file://{os.path.abspath(ckpt_dir)}",
            global_step=step,
            epoch=self.epoch,
            strategy=self.core.strategy.name,
        )


__all__ = ["TrainerCore", "TrainerController", "Checkpoint", "StepResult"]
