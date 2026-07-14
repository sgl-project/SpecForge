# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Bounded pull-through rollout for colocated online training.

The local target and trainer share one process (and usually one GPU).  Producing
the complete prompt pool before training therefore retains every sample's target
features in ``LocalFeatureStore`` at once.  ``LocalRolloutStream`` presents the
normal loader-facing queue contract, but pumps rollout synchronously only when
the loader asks for its next batch.  At most ``max_resident_samples`` freshly
captured samples are resident at a time.

This is intentionally local-only.  Cross-process producer/consumer completion,
retry and backpressure belong to the disaggregated streaming channel.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from specforge.runtime.contracts import SampleRef


class LocalRolloutStream:
    """A bounded ``SampleRefQueue`` facade that produces refs on demand.

    Rollout runs in the caller of :meth:`get`, so a producer exception is the
    loader/trainer exception rather than an out-of-band thread failure.  The
    stream is a context manager: leaving the training lifecycle stops every
    worker and evicts any produced-but-unconsumed local features.
    """

    # FeatureDataLoader's generic prefetch thread must not move local target
    # inference onto a background CUDA thread.  Besides making failure ordering
    # less clear, a prefetched rollout could continue after trainer early-stop.
    loader_prefetch_safe = False

    def __init__(
        self,
        *,
        controller,
        workers: Sequence,
        feature_store,
        max_resident_samples: int,
        max_stalled_rounds: int = 3,
    ) -> None:
        if not workers:
            raise ValueError("local rollout requires at least one worker")
        if max_resident_samples < 1:
            raise ValueError("max_resident_samples must be >= 1")
        if max_stalled_rounds < 1:
            raise ValueError("max_stalled_rounds must be >= 1")
        if not callable(getattr(feature_store, "abort_all", None)):
            raise TypeError(
                "local rollout requires a private feature store with abort_all()"
            )

        self.controller = controller
        self.workers = tuple(workers)
        self.feature_store = feature_store
        self.max_resident_samples = int(max_resident_samples)
        self.max_stalled_rounds = int(max_stalled_rounds)
        self._queue = controller.sample_queue

        self._started = False
        self._closed = False
        self._next_worker = 0
        self.produced_count = 0
        self.peak_resident_samples = 0

    def __enter__(self) -> "LocalRolloutStream":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        reason = "training_failed" if exc_type is not None else "training_finished"
        self.close(reason=reason)
        return False

    def _start(self) -> None:
        if self._started or self._closed:
            return
        self._started = True
        for worker in self.workers:
            worker.start()

    def _record_produced(self, refs: List[SampleRef]) -> None:
        resident = int(self.feature_store.health().get("resident_samples", 0))
        self.produced_count += len(refs)
        self.peak_resident_samples = max(self.peak_resident_samples, resident)
        if resident > self.max_resident_samples:
            raise RuntimeError(
                "local rollout exceeded its resident-sample bound: "
                f"{resident} > {self.max_resident_samples}"
            )

    def _raise_if_failed(self, status: dict) -> None:
        failed = int(status.get("prompts_failed", 0))
        if failed:
            raise RuntimeError(
                "local rollout ended with "
                f"{failed} terminally failed prompt(s); refusing partial training"
            )

    def _drained(self, status: dict) -> bool:
        return not status.get("prompts_pending") and not status.get("prompts_leased")

    def _pump_once(self, max_tasks: int) -> List[SampleRef]:
        resident = int(self.feature_store.health().get("resident_samples", 0))
        self.peak_resident_samples = max(self.peak_resident_samples, resident)
        capacity = self.max_resident_samples - resident
        if capacity <= 0:
            raise RuntimeError(
                "local rollout cannot make progress: its feature store is at "
                f"the {self.max_resident_samples}-sample bound but the ref queue "
                "cannot fill the requested batch"
            )

        worker = self.workers[self._next_worker]
        self._next_worker = (self._next_worker + 1) % len(self.workers)
        refs = worker.run_once(max_tasks=min(max_tasks, capacity))
        self._record_produced(refs)
        return refs

    def get(
        self,
        max_refs: int,
        timeout_s: Optional[float] = None,
        **_unused,
    ) -> List[SampleRef]:
        """Lease a full batch, producing only enough refs to satisfy it.

        ``timeout_s`` is accepted for the queue protocol.  Local model inference
        itself is synchronous and cannot be meaningfully cancelled by that
        timeout, so completion is governed by rollout success or prompt-pool EOF.
        """
        del timeout_s
        if max_refs < 1:
            return []
        if max_refs > self.max_resident_samples:
            raise ValueError(
                f"requested {max_refs} refs from a local rollout stream bounded "
                f"to {self.max_resident_samples} resident samples"
            )
        if self._closed:
            return []

        self._start()
        refs: List[SampleRef] = []
        stalled_worker_calls = 0
        max_stalled_calls = self.max_stalled_rounds * len(self.workers)
        try:
            while len(refs) < max_refs:
                leased = self._queue.get(max_refs - len(refs), timeout_s=0.0)
                if leased:
                    refs.extend(leased)
                    stalled_worker_calls = 0
                    continue

                status = self.controller.status()
                self._raise_if_failed(status)
                if self._drained(status):
                    return refs

                produced = self._pump_once(max_refs - len(refs))
                if produced:
                    stalled_worker_calls = 0
                    continue
                stalled_worker_calls += 1
                if stalled_worker_calls >= max_stalled_calls:
                    status = self.controller.status()
                    self._raise_if_failed(status)
                    raise RuntimeError(
                        "local rollout made no progress while prompts remain: "
                        f"pending={status.get('prompts_pending', 0)} "
                        f"leased={status.get('prompts_leased', 0)}"
                    )
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
        if retryable:
            return
        for ref in refs:
            self.feature_store.abort(ref.sample_id, reason=reason)

    def depth(self) -> int:
        return self._queue.depth()

    def in_flight(self) -> int:
        return self._queue.in_flight()

    def close(self, *, reason: str = "closed") -> None:
        """Stop production and evict every unacknowledged local feature."""
        if self._closed:
            return
        self._closed = True

        for worker in self.workers:
            worker.stop(reason=reason)
        self.feature_store.abort_all(reason=reason)


__all__ = ["LocalRolloutStream"]
