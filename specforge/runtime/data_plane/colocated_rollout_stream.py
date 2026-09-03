# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Bounded pull-through rollout for colocated online training."""

from __future__ import annotations

import itertools
import time
from typing import Dict, Iterable, Iterator, List, Optional

from specforge.runtime.contracts import SampleRef


class LocalRolloutStream:
    """Capture one target batch on the trainer thread whenever the loader asks.

    ``prompts`` is this rank's lazy, deterministic prompt stream (see
    ``specforge.training.prompt_plan``). When ``get`` finds the sample queue
    empty it ingests exactly one target batch (``target_batch_size`` prompts)
    into the rank-private controller and runs the worker once; the worker
    commits this rank's ``local_batch_size`` samples. Nothing is captured ahead
    of demand, so the feature store never stages more than one local batch, and
    the trainer releases the previous batch before the next capture starts.

    The stream implements the loader's ``queue`` contract (``get``/``ack``/
    ``fail``/``depth``/``in_flight``) and is not prefetch-safe: capture runs
    device work on the calling thread. Cross-process retry and durability stay
    with the disaggregated runtime.
    """

    loader_prefetch_safe = False

    def __init__(
        self,
        *,
        controller,
        worker,
        feature_store,
        prompts: Iterable[dict],
        local_batch_size: int,
        target_batch_size: int,
    ) -> None:
        if local_batch_size < 1:
            raise ValueError("local_batch_size must be >= 1")
        if target_batch_size < local_batch_size or target_batch_size % local_batch_size:
            raise ValueError(
                "target_batch_size must be a positive multiple of local_batch_size"
            )
        if not callable(getattr(feature_store, "abort_all", None)):
            raise TypeError(
                "local rollout requires a private feature store with abort_all()"
            )

        self.controller = controller
        self.worker = worker
        self.feature_store = feature_store
        self.local_batch_size = int(local_batch_size)
        self.target_batch_size = int(target_batch_size)
        self._prompts: Iterator[dict] = iter(prompts)
        self._queue = controller.sample_queue
        self._started = False
        self._closed = False
        # Cumulative counters; perf_metrics() reports interval deltas for the
        # capture counters and running peaks for staging.
        self.produced_count = 0
        self.capture_calls = 0
        self.capture_time_s = 0.0
        self.peak_staged_samples = 0
        self.peak_staged_bytes = 0
        self._reported = {
            "produced_count": 0,
            "capture_calls": 0,
            "capture_time_s": 0.0,
        }

    # -- lifecycle -------------------------------------------------------------
    def __enter__(self) -> "LocalRolloutStream":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.close(reason="training_failed" if exc_type else "training_finished")
        return False

    def close(self, *, reason: str = "closed") -> None:
        if self._closed:
            return
        self._closed = True
        self.worker.stop(reason=reason)
        self.feature_store.abort_all(reason=reason)

    def _start(self) -> None:
        if not self._started:
            self._started = True
            self.worker.start()

    # -- capture ---------------------------------------------------------------
    def _capture_next_batch(self) -> bool:
        """Ingest and capture one target batch; False once prompts are exhausted."""
        prompts = list(itertools.islice(self._prompts, self.target_batch_size))
        if not prompts:
            return False
        if len(prompts) != self.target_batch_size:
            raise RuntimeError(
                f"prompt stream ended with {len(prompts)} prompts; the plan must "
                f"be a multiple of the {self.target_batch_size}-prompt target batch"
            )
        self.controller.ingest_prompts(prompts)
        started = time.perf_counter()
        try:
            refs = self.worker.run_once(max_tasks=len(prompts))
        finally:
            self.capture_calls += 1
            self.capture_time_s += time.perf_counter() - started
        self._raise_if_failed()
        if len(refs) != self.local_batch_size:
            raise RuntimeError(
                f"capture committed {len(refs)} local sample(s) for a "
                f"{len(prompts)}-prompt target batch; expected {self.local_batch_size}"
            )
        self.produced_count += len(refs)
        self._record_staging()
        return True

    def _raise_if_failed(self) -> None:
        failed = int(self.controller.status().get("prompts_failed", 0))
        if failed:
            raise RuntimeError(
                f"local rollout has {failed} terminally failed prompt(s); "
                "refusing partial training"
            )

    def _record_staging(self) -> None:
        health = self.feature_store.health()
        staged = int(health.get("resident_samples", 0))
        self.peak_staged_samples = max(self.peak_staged_samples, staged)
        self.peak_staged_bytes = max(
            self.peak_staged_bytes, int(health.get("resident_bytes", 0))
        )
        if staged > self.local_batch_size:
            raise RuntimeError(
                f"local rollout staged {staged} samples; the bound is one local "
                f"batch of {self.local_batch_size}"
            )

    # -- queue contract --------------------------------------------------------
    def get(
        self,
        max_refs: int,
        timeout_s: Optional[float] = None,
        **_unused,
    ) -> List[SampleRef]:
        del timeout_s  # capture is synchronous; there is nothing to wait for
        if max_refs < 1 or self._closed:
            return []
        if max_refs > self.local_batch_size:
            raise ValueError(
                f"requested {max_refs} refs from a stream that stages at most "
                f"{self.local_batch_size} local samples"
            )
        self._start()
        refs: List[SampleRef] = []
        try:
            while len(refs) < max_refs:
                leased = self._queue.get(max_refs - len(refs), timeout_s=0.0)
                if leased:
                    refs.extend(leased)
                elif not self._capture_next_batch():
                    break
            return refs
        except BaseException:
            if refs:
                self._queue.fail(
                    refs, reason="local_rollout_get_failed", retryable=True
                )
            raise

    def ack(self, refs: List[SampleRef]) -> None:
        self._queue.ack(refs)

    def fail(self, refs: List[SampleRef], reason: str, retryable: bool) -> None:
        self._queue.fail(refs, reason=reason, retryable=retryable)
        if not retryable:
            for ref in refs:
                self.feature_store.abort(ref.sample_id, reason=reason)

    def depth(self) -> int:
        return self._queue.depth()

    def in_flight(self) -> int:
        return self._queue.in_flight()

    # -- observability ---------------------------------------------------------
    def perf_metrics(self) -> Dict[str, float]:
        """Interval capture counters plus running peaks of staged features."""
        calls = self.capture_calls - self._reported["capture_calls"]
        seconds = self.capture_time_s - self._reported["capture_time_s"]
        produced = self.produced_count - self._reported["produced_count"]
        self._reported.update(
            capture_calls=self.capture_calls,
            capture_time_s=self.capture_time_s,
            produced_count=self.produced_count,
        )
        return {
            "colocated_capture_calls": float(calls),
            "colocated_capture_time_s": seconds,
            "colocated_capture_samples_per_second": produced / max(seconds, 1e-12),
            "colocated_peak_staged_samples": float(self.peak_staged_samples),
            "colocated_peak_staged_gib": self.peak_staged_bytes / float(1 << 30),
        }


__all__ = ["LocalRolloutStream"]
