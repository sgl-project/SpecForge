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

    Matches stock ``train_dflash.py`` (``dist.all_reduce(acc); acc /= world``):
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
        import os as _os

        _P = int(_os.environ.get("PROFILE_STEPS", "0"))
        if _P:
            import time as _t

            torch.cuda.synchronize()
            _a0 = _t.perf_counter()
        out: StepOutput = self.strategy.forward_loss(batch, ctx)
        loss = out.loss / self.accumulation_steps
        self._micro += 1
        # The boundary is known before backward so the backend can defer the FSDP
        # gradient reduction (no_sync) on non-boundary micro-steps.
        stepped = self._micro % self.accumulation_steps == 0
        if _P:
            torch.cuda.synchronize()
            _a1 = _t.perf_counter()
        self.backend.backward(loss, is_boundary=stepped)
        if _P:
            torch.cuda.synchronize()
            _a2 = _t.perf_counter()
        grad_norm = self.backend.step() if stepped else None
        if _P:
            torch.cuda.synchronize()
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

    # Deliberately no eval_step: evaluation runs ``Evaluator.run`` on raw
    # ``forward_loss`` outputs — ``_result`` scalarizes away the per-position
    # count tensors that correct acc-len aggregation needs.

    def _result(self, out: StepOutput, grad_norm, stepped: bool) -> StepResult:
        # DP-average loss AND accuracy across ranks before logging, matching stock
        # train_dflash.py (all_reduce(loss)/world, all_reduce(acc)/world).
        metrics: Dict[str, Any] = {"loss": _scalar(_dp_mean(out.loss))}
        for key in ("acces", "acceptance_rates", "plosses"):
            if key in out.metrics:
                metrics[key.rstrip("es") if key == "acces" else key] = _scalar(
                    out.metrics[key]
                )
        if "accuracy" in out.metrics:
            # DP-average the accuracy across ranks (matches stock train_dflash.py)
            # so the logged curve is smooth, not a single rank's noisy local batch.
            metrics["acc"] = _scalar(_dp_mean(out.metrics["accuracy"]))
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
        # Rank0-broadcast once: a rank-local eval_data view must not let ranks
        # enter or skip the evaluator's collectives alone.
        eval_enabled = self._rank0_decision(eval_data is not None)
        pending_ack: List[str] = []
        import time as _time

        _PROFILE = int(os.environ.get("PROFILE_STEPS", "0"))
        _prof = {"data": 0.0, "step": 0.0, "n": 0}
        _TPROF = int(
            os.environ.get("PROFILE_TORCH", "0")
        )  # active optimizer steps to profile; 0=off
        _tprof_warmup = int(os.environ.get("PROFILE_TORCH_WARMUP", "40"))
        _tprof_rank0 = (
            not torch.distributed.is_initialized()
        ) or torch.distributed.get_rank() == 0
        _tprof = {"p": None, "done": False}
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
            _it = iter(stream)
            while True:
                if _PROFILE:
                    torch.cuda.synchronize()
                    _pt0 = _time.perf_counter()
                try:
                    batch = next(_it)
                except StopIteration:
                    break
                if _PROFILE:
                    torch.cuda.synchronize()
                    _pt1 = _time.perf_counter()
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
                if _PROFILE:
                    torch.cuda.synchronize()
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
                if _TPROF and _tprof_rank0 and not _tprof["done"]:
                    if self.global_step == _tprof_warmup and _tprof["p"] is None:
                        import torch.profiler as _tp

                        _tprof["p"] = _tp.profile(
                            activities=[
                                _tp.ProfilerActivity.CPU,
                                _tp.ProfilerActivity.CUDA,
                            ],
                            record_shapes=True,
                        )
                        _tprof["p"].start()
                        print(f"[tprof] started @ step {self.global_step}", flush=True)
                    elif (
                        _tprof["p"] is not None
                        and self.global_step >= _tprof_warmup + _TPROF
                    ):
                        _p = _tprof["p"]
                        _p.stop()
                        _ka = _p.key_averages()
                        print(
                            "[tprof] ===== top ops by self_cuda_time =====\n"
                            + _ka.table(sort_by="self_cuda_time_total", row_limit=35),
                            flush=True,
                        )
                        try:
                            _ncalls = sum(int(e.count) for e in _ka)
                            _cuda_ms = (
                                sum(
                                    float(getattr(e, "self_cuda_time_total", 0.0))
                                    for e in _ka
                                )
                                / 1000.0
                            )
                            print(
                                f"[tprof] window={_TPROF} steps  "
                                f"total_op_calls={_ncalls}  "
                                f"sum_self_cuda={_cuda_ms:.1f}ms",
                                flush=True,
                            )
                        except Exception as _e:
                            print(f"[tprof] summary err: {_e}", flush=True)
                        try:
                            _p.export_chrome_trace(
                                os.environ.get(
                                    "PROFILE_TORCH_TRACE",
                                    "/workspace/SpecForge-domino/tprof_trace.json",
                                )
                            )
                            print("[tprof] chrome trace exported", flush=True)
                        except Exception as _e:
                            print(f"[tprof] trace err: {_e}", flush=True)
                        _tprof["p"] = None
                        _tprof["done"] = True
                if self.ack_fn is not None:
                    # durable ack transaction at the optimizer-step boundary
                    self.ack_fn(pending_ack, self.global_step)
                    pending_ack = []
                if self.logger and self.global_step % max(1, self.log_interval) == 0:
                    self.logger(result.metrics, self.global_step)
                eval_metrics: Optional[Dict[str, Any]] = None
                if (
                    self.eval_interval
                    and eval_enabled
                    and self.global_step % self.eval_interval == 0
                ):
                    eval_metrics = self.evaluate(eval_data)
                    module.train()
                    if eval_metrics:
                        if self.logger:
                            self.logger(eval_metrics, self.global_step)
                        self.last_metrics = {**self.last_metrics, **eval_metrics}
                # is_better is a collective (rank0 verdict broadcast inside the
                # manager); its guard is rank-identical because eval_metrics is
                # DP-reduced. Empty ({}) eval metrics skip best entirely.
                interval_hit = bool(
                    self.save_interval and self.global_step % self.save_interval == 0
                )
                is_best = bool(
                    self.save_interval
                    and eval_metrics
                    and self._checkpoint_manager().is_better(eval_metrics)
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
        return self.global_step

    @torch.no_grad()
    def evaluate(self, data: Optional[Iterable[TrainBatch]]) -> Dict[str, Any]:
        """Full-pass eval via :class:`Evaluator`: rank-identical ``eval/*``
        metrics, ``{}`` when zero batches were processed globally. ``data=None``
        (empty local shard) still joins the evaluator's collectives."""
        from specforge.eval import Evaluator

        module = self.core.strategy.trainable_module()
        module.eval()
        # Same StepContext as the train path so schedule-dependent losses
        # (Domino's lambda_base) evaluate at the live step, not at step 0.
        ctx = StepContext(global_step=self.global_step, total_steps=self.total_steps)
        return Evaluator().run(
            lambda batch: self.core.strategy.forward_loss(batch, ctx), data
        )

    @staticmethod
    def _rank0_decision(flag: bool) -> bool:
        """Broadcast rank0's verdict so a collective-bearing branch is entered by
        every rank or by none."""
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
