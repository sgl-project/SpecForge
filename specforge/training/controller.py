# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""TrainerController: the run loop over the step frame and a step runner.

It owns the horizon, the optimizer windows it cuts from the training stream,
logging, evaluation and checkpoint cadence, and sample acknowledgements. What
happens inside a window belongs to :mod:`specforge.training.frame` and the
runner.
"""

from __future__ import annotations

import itertools
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

import torch

from specforge.runtime.contracts import TrainBatch
from specforge.training.contracts import RankFailure, StepMetrics, StepRunner
from specforge.training.frame import run_window
from specforge.training.strategies.base import StepContext

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


@dataclass
class _PerfWindow:
    """Wall-clock totals since the last log line."""

    started: float = field(default_factory=time.perf_counter)
    steps: int = 0
    samples: int = 0
    data_wait_s: float = 0.0
    #: Everything the training stream did between two batches: the wait for
    #: the next one and, at an epoch boundary, the loader's epoch setup.
    stream_s: float = 0.0
    train_compute_s: float = 0.0
    durable_ack_s: float = 0.0


def _take(items: List[TrainBatch]) -> Iterator[TrainBatch]:
    """Yield the items and forget them."""
    while items:
        yield items.pop()


class _AsyncAckRunner:
    """Run optimizer-boundary durable acks on ONE background thread, in order.

    ``submit`` first waits for the previous ack, so at most one ack is in
    flight, acks run strictly in optimizer-step order, and every rank issues
    the ack's collectives in the same sequence (the ack's process group must
    not be shared with the training thread). The first failure is sticky: it
    re-raises from every later ``flush``/``submit`` on the training thread, so
    no checkpoint, eval, or further ack proceeds past a failed durable ack.
    """

    def __init__(self, ack_fn: Callable[[List[str], int], None]) -> None:
        self._ack_fn = ack_fn
        self._cv = threading.Condition()
        self._job: Optional[tuple] = None
        self._closed = False
        self._error: Optional[BaseException] = None
        self._exec_s = 0.0
        # CUDA's current device is per thread; pin the ack thread to the
        # trainer's so nothing it touches lands on device 0 by default.
        self._device = (
            torch.cuda.current_device()
            if torch.cuda.is_available() and torch.cuda.is_initialized()
            else None
        )
        self._thread = threading.Thread(
            target=self._run, name="specforge-durable-ack", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        if self._device is not None:
            torch.cuda.set_device(self._device)
        while True:
            with self._cv:
                while self._job is None and not self._closed:
                    self._cv.wait()
                if self._job is None:
                    return
                sample_ids, step = self._job
            started = time.perf_counter()
            error = None
            try:
                self._ack_fn(sample_ids, step)
            except BaseException as exc:
                exc.add_note(
                    f"raised by the background durable ack of optimizer step {step}"
                )
                error = exc
            with self._cv:
                self._exec_s += time.perf_counter() - started
                if error is not None and self._error is None:
                    self._error = error
                self._job = None
                self._cv.notify_all()

    def wait(self) -> Optional[BaseException]:
        """Block until no ack is in flight; return the sticky failure, if any."""
        with self._cv:
            while self._job is not None:
                self._cv.wait()
            return self._error

    def flush(self) -> None:
        error = self.wait()
        if error is not None:
            raise error

    def submit(self, sample_ids: List[str], step: int) -> None:
        self.flush()
        job = (list(sample_ids), step)
        with self._cv:
            if self._closed:
                raise RuntimeError("durable ack runner is closed")
            # Wake the worker BEFORE publishing the job: it re-checks ``_job``
            # only once this block releases the lock. The reverse order lets a
            # SIGTERM unwind (raised between the two statements) strand a
            # published job whose worker never woke, and close() would then
            # wait on it forever.
            self._cv.notify_all()
            self._job = job

    def pop_exec_seconds(self) -> float:
        """Background ack execution time since the previous call."""
        with self._cv:
            elapsed, self._exec_s = self._exec_s, 0.0
        return elapsed

    def close(self) -> Optional[BaseException]:
        """Wait for the in-flight ack, stop the thread, return any failure."""
        error = self.wait()
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        self._thread.join()
        return error


class TrainerController:
    """Lifecycle: fit / evaluate / checkpoint.

    ``save_checkpoint`` persists resumable draft state and returns a
    :class:`Checkpoint`; ``specforge export`` materializes that state into the
    serving or Hugging Face model format.  Evaluation is configured once at
    construction time, so the public training lifecycle remains one no-argument
    :meth:`Trainer.fit` call.

    ``async_ack=True`` runs ``ack_fn`` for optimizer step N on a background
    thread while step N+1 computes. The durable marker (and the consumed
    counter it drives) then lags the trainer by at most one optimizer step:
    the pending ack is flushed before the next ack starts, before eval, before
    every checkpoint (a checkpoint is never ahead of its ack), and before
    ``fit`` returns; an ack failure re-raises on the training thread at that
    flush. ``ack_fn`` must be safe to call off the training thread and must
    not share a process group with training collectives.
    """

    def __init__(
        self,
        runner: StepRunner,
        *,
        strategy_name: str,
        accumulation_steps: int = 1,
        data_parallel_size: int = 1,
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
        async_ack: bool = False,
    ) -> None:
        if (start_batch == 0) != (start_samples == 0):
            raise ValueError(
                f"start_batch={start_batch} and start_samples={start_samples} "
                f"describe the same mid-epoch position and must be zero or "
                f"nonzero together"
            )
        self.runner = runner
        # Recorded in every checkpoint and checked on resume.
        self.strategy_name = strategy_name
        # Micro-batches per optimizer update.
        self.accumulation_steps = max(1, accumulation_steps)
        # Ranks that receive different samples; scales the logged throughput.
        self.data_parallel_size = data_parallel_size
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
        self.async_ack = bool(async_ack)
        # Live only inside fit(); None means acks run inline.
        self._ack_runner: Optional[_AsyncAckRunner] = None
        # Training-thread time blocked on ack flushes outside the boundary
        # submit (eval / checkpoint), folded into perf/durable_ack_time_s.
        self._ack_flush_wait_s = 0.0
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
        self._last_result: Optional[StepMetrics] = None
        self._pending_ack: List[str] = []
        self._perf = _PerfWindow()
        # Set once fit() has raised. Whatever failed (a window, a batch pull,
        # an acknowledgement), the data position is then ahead of the work
        # that was committed, so the run can neither continue nor be saved.
        self._failed: Optional[str] = None
        self._last_eval_metrics: Dict[str, Any] = {}
        self.last_checkpoint_step: Optional[int] = None
        from specforge.training.profiling import ProfilingOptions, StepProfiler

        options = profiling_options or ProfilingOptions()
        self._step_profiler = StepProfiler(
            options,
            output_dir=output_dir,
        )

    @property
    def last_metrics(self) -> Dict[str, Any]:
        metrics = (
            dict(self._last_result.materialize())
            if self._last_result is not None
            else {}
        )
        metrics.update(self._last_eval_metrics)
        return metrics

    def _make_progress_bar(self):
        """Build a rank-0 optimizer-step bar for interactive terminals only."""
        if not sys.stderr.isatty() or (
            self.max_steps is not None and self.global_step >= self.max_steps
        ):
            return None
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return None
        from tqdm import tqdm

        total = self.max_steps if self.max_steps is not None else self.total_steps
        return tqdm(
            total=total,
            initial=self.global_step,
            desc="Training",
            unit="step",
            mininterval=1.0,
            dynamic_ncols=True,
        )

    def fit(self, data: Iterable[TrainBatch]) -> int:
        self._refuse_after_failure("continue")
        progress = self._make_progress_bar()
        if self.async_ack and self.ack_fn is not None:
            self._ack_runner = _AsyncAckRunner(self.ack_fn)
        try:
            step = self._fit(data, progress)
            # Every optimizer boundary this call reached is durable on return.
            self._close_ack_runner(None)
            return step
        except BaseException as exc:
            self._failed = f"{type(exc).__name__}: {exc}"
            self._close_ack_runner(exc)
            raise
        finally:
            if progress is not None:
                progress.close()

    def _refuse_after_failure(self, action: str) -> None:
        if self._failed is not None:
            raise RuntimeError(
                f"training cannot {action}: an earlier fit() of this run failed "
                "after consuming work that was never committed "
                f"({self._failed}); resume from its last checkpoint"
            )

    def _close_ack_runner(self, primary: Optional[BaseException]) -> None:
        runner, self._ack_runner = self._ack_runner, None
        if runner is None:
            return
        # On failure still wait: the in-flight ack's collectives are already
        # joined by every peer's ack thread, and lifecycle cleanup (feature
        # drains, consumer_done) must not race it.
        error = runner.close()
        if error is None:
            return
        if primary is None:
            raise error
        if error is not primary:
            primary.add_note(
                "the background durable ack also failed: "
                f"{type(error).__name__}: {error}"
            )

    def _flush_durable_ack(self) -> None:
        """Wait for the in-flight async ack; re-raise its failure here."""
        if self._ack_runner is None:
            return
        started = time.perf_counter()
        try:
            self._ack_runner.flush()
        finally:
            self._ack_flush_wait_s += time.perf_counter() - started

    def _fit(self, data: Iterable[TrainBatch], progress: Optional[Any]) -> int:
        if self.max_steps is not None and self.global_step >= self.max_steps:
            logger.info(
                "fit: global_step=%d already at max_steps=%d; nothing to train",
                self.global_step,
                self.max_steps,
            )
            return self.global_step
        # Rank0-broadcast once: rank-local assembly must not let ranks enter or
        # skip the evaluator's collectives independently.
        eval_enabled = self._rank0_decision(
            self.eval_interval > 0 and self.eval_data_factory is not None
        )
        self._pending_ack = []
        self._perf = perf = _PerfWindow()
        for window in self._windows(data):
            ctx = StepContext(
                global_step=self.global_step,
                total_steps=self.total_steps,
                collect_detailed_metrics=(
                    self.logger is not None
                    and (self.global_step + 1) % max(1, self.log_interval) == 0
                ),
                accumulation_steps=self.accumulation_steps,
            )
            # The frame pulls the rest of the window while it runs; that time
            # is the data path's, not the step's.
            window_started = time.perf_counter()
            stream_before = perf.stream_s
            result = run_window(self.runner, window, ctx)
            perf.train_compute_s += (time.perf_counter() - window_started) - (
                perf.stream_s - stream_before
            )
            # Everything keyed on optimizer steps fires only here, once the
            # window's update is applied.
            self.global_step += 1
            self._last_result = result
            self._last_eval_metrics = {}
            perf.steps += 1
            self._step_profiler.after_optimizer_step(self.global_step)
            if self.ack_fn is not None:
                # durable ack transaction at the optimizer-step boundary
                durable_ack_started = time.perf_counter()
                if self._ack_runner is None:
                    self.ack_fn(self._pending_ack, self.global_step)
                else:
                    # Waits only for step N-1's ack; step N's overlaps
                    # the next step's compute.
                    self._ack_runner.submit(self._pending_ack, self.global_step)
                perf.durable_ack_s += time.perf_counter() - durable_ack_started
                self._pending_ack = []
            if self.logger and self.global_step % max(1, self.log_interval) == 0:
                log_metrics = dict(result.materialize())
                learning_rate = result.learning_rate
                if learning_rate is not None:
                    log_metrics["lr"] = learning_rate
                perf_elapsed_s = max(time.perf_counter() - perf.started, 1e-12)
                # Time the training thread was BLOCKED on durable acks; in
                # async mode the ack itself runs behind the next step.
                perf.durable_ack_s += self._ack_flush_wait_s
                self._ack_flush_wait_s = 0.0
                if self._ack_runner is not None:
                    background_s = self._ack_runner.pop_exec_seconds()
                    log_metrics["perf/durable_ack_background_time_s"] = (
                        background_s / max(1, perf.steps)
                    )
                log_metrics.update(
                    {
                        "perf/optimizer_steps_per_hour": (
                            perf.steps * 3600.0 / perf_elapsed_s
                        ),
                        "perf/optimizer_step_time_s": (
                            perf_elapsed_s / max(1, perf.steps)
                        ),
                        "perf/data_wait_time_s": (
                            perf.data_wait_s / max(1, perf.steps)
                        ),
                        "perf/train_compute_time_s": (
                            perf.train_compute_s / max(1, perf.steps)
                        ),
                        "perf/durable_ack_time_s": (
                            perf.durable_ack_s / max(1, perf.steps)
                        ),
                        "perf/global_samples_per_second": (
                            perf.samples * self.data_parallel_size / perf_elapsed_s
                        ),
                    }
                )
                # Loader wait attribution: producer bucket = capture or
                # dispatch cannot keep up, fetch bucket = transfer is slow.
                perf_snapshot = getattr(data, "perf_counters_snapshot", None)
                if callable(perf_snapshot):
                    loader = perf_snapshot(reset=True)
                    steps = max(1, perf.steps)
                    samples = max(1.0, loader["fetch_samples"])
                    log_metrics.update(
                        {
                            "perf/data_wait_producer_s": (
                                loader["wait_producer_s"] / steps
                            ),
                            "perf/data_wait_fetch_s": (loader["wait_fetch_s"] / steps),
                            "perf/fetch_seconds_per_sample": (
                                loader["fetch_s"] / samples
                            ),
                            "perf/fetch_delivered_gib_per_s": (
                                loader["fetch_bytes"]
                                / (1 << 30)
                                / max(1e-12, perf_elapsed_s)
                            ),
                        }
                    )
                self.logger(log_metrics, self.global_step)
                self._perf = perf = _PerfWindow()
            eval_metrics: Optional[Dict[str, Any]] = None
            if eval_enabled and self.global_step % self.eval_interval == 0:
                eval_metrics = self.evaluate_configured()
                if eval_metrics:
                    if self.logger:
                        self.logger(eval_metrics, self.global_step)
                    self._last_eval_metrics = dict(eval_metrics)
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
                self._checkpoint_manager().update_best(self.global_step, eval_metrics)
            if progress is not None:
                progress.update(1)
            if self.max_steps is not None and self.global_step >= self.max_steps:
                return self.global_step
        return self.global_step

    def _windows(self, data: Iterable[TrainBatch]) -> Iterator[Iterator[TrainBatch]]:
        """Cut the training stream into optimizer windows, lazily.

        A window spans epoch boundaries, so accumulation does not restart at
        an epoch. Each window has its first micro-batch in hand, which is how
        the end of the stream at a boundary differs from its end inside a
        window; the rest is pulled as the frame consumes it. The run loop
        keeps no reference to a micro-batch it has handed over.
        """
        stream = self._batches(data)
        first: List[TrainBatch] = []
        for batch in stream:
            first.append(batch)
            del batch
            yield itertools.chain(
                _take(first), itertools.islice(stream, self.accumulation_steps - 1)
            )

    def _batches(self, data: Iterable[TrainBatch]) -> Iterator[TrainBatch]:
        """Yield the training stream epoch after epoch, keeping its position.

        The position, the pending acknowledgements and the data wait are
        updated as each batch is pulled, whoever pulls it.
        """
        resumed = time.perf_counter()
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
                data_wait_started = time.perf_counter()
                try:
                    batch = next(_it)
                except StopIteration:
                    break
                self._perf.data_wait_s += time.perf_counter() - data_wait_started
                self._perf.samples += len(batch.sample_ids)
                self._epoch_batch += 1
                self._epoch_samples += len(batch.sample_ids)
                self.micro_step += 1
                if self.ack_fn is not None:
                    self._pending_ack.extend(batch.sample_ids)
                self._step_profiler.before_micro_step(self.global_step)
                self._perf.stream_s += time.perf_counter() - resumed
                # Suspended here until the next batch is asked for, which may
                # be a whole replay later: hold no reference meanwhile.
                handed = [batch]
                del batch
                yield handed.pop()
                resumed = time.perf_counter()
            self._epoch_batch = 0
            self._epoch_samples = 0
            # Persist the *next* epoch after a naturally exhausted pass.  A
            # checkpoint taken after fit() returns must describe completed
            # work, not epoch ``N`` at batch zero (which would replay that
            # entire epoch on resume).
            self.epoch = epoch + 1
        self._perf.stream_s += time.perf_counter() - resumed

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
        self._flush_durable_ack()
        if self.eval_data_factory is None:
            return self.evaluate(None)
        data = self.eval_data_factory()
        if data is None or not hasattr(data, "__enter__"):
            return self.evaluate(data)
        with data as entered:
            return self.evaluate(data if entered is None else entered)

    def evaluate(self, data: Optional[Iterable[TrainBatch]]) -> Dict[str, Any]:
        """Full-pass eval via :class:`Evaluator`.

        Returns rank-identical ``eval/*`` metrics, or ``{}`` when zero batches
        were processed globally. ``data=None`` (an empty local shard) still
        joins the evaluator's collectives.
        """
        from specforge.eval import Evaluator

        self._flush_durable_ack()
        # Use the train path's live context so schedule-dependent losses (for
        # example Domino's lambda_base) are not evaluated as if at step zero.
        ctx = StepContext(global_step=self.global_step, total_steps=self.total_steps)
        return Evaluator().run(self.runner, data, ctx)

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
        # After a failed fit() the data position is ahead of the applied
        # updates; saving it would replace a good checkpoint of this step.
        self._refuse_after_failure("be saved")
        # A checkpoint must never be ahead of its durable ack: resume requires
        # the ledger marker to equal the checkpoint step.
        self._flush_durable_ack()
        # Every rank participates: gathering a sharded model is collective and
        # every rank saves its own state. A rank that fails once the gather is
        # over, copying its optimizer state for one, reports it through the
        # agreement; raising here would leave the others waiting in it. A
        # failure inside the gather propagates: the ranks are apart already.
        state = failure = None
        try:
            state = self.runner.checkpoint_state()
        except RankFailure as error:
            failure = error
        # What rank 0 records for resume has to hold for every rank; a
        # difference raises here, before anything is written.
        try:
            self.runner.agree(
                None if failure is None else str(failure),
                value=None if state is None else ("loss kernels", state.kernels),
            )
        except Exception:
            if failure is not None:
                # The rank that failed raises what it failed with.
                raise failure.__cause__ or failure
            raise
        mgr = self._checkpoint_manager()
        shared = None
        if mgr.is_rank0():
            shared = {
                "draft_state_dict": state.draft,
                "global_step": step,
                "epoch": self.epoch,
                "epoch_batch": self._epoch_batch,
                "epoch_samples": self._epoch_samples,
                "strategy": self.strategy_name,
                "run_id": self.run_id,
                "world_size": (
                    torch.distributed.get_world_size()
                    if torch.distributed.is_initialized()
                    else 1
                ),
                **self.checkpoint_extra,
            }
            if "canonical_loss_implementations" in shared:
                # Write the exact kernel snapshot that was agreed across ranks above.
                shared["canonical_loss_implementations"] = dict(state.kernels)
            if state.replicated is not None:
                # Saved once in the shared payload; the rank files still keep
                # each rank's own state.
                shared["replicated_optimizer_state"] = state.replicated
        ckpt_dir = mgr.save(shared, step, rank_state=state.rank)
        self.last_checkpoint_step = step
        return Checkpoint(
            checkpoint_uri=f"file://{os.path.abspath(ckpt_dir)}",
            global_step=step,
            epoch=self.epoch,
            strategy=self.strategy_name,
        )


__all__ = ["TrainerController", "Checkpoint"]
