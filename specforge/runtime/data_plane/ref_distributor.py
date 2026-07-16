# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""RefDistributor: the centralized dispatcher for DP online consumers.

One distributor per run (it lives on trainer DP rank 0). It is the ONLY reader
of the producer's ref channel and the only holder of consume book-keeping:

    source channel -> commit (ONE ledger, dedup) -> optimizer-step windows -> per-rank inbox

Each rank reads a private inbox (a plain consume-once :class:`StreamingRefQueue`)
and holds no channel offset, no partition math, no ledger — the design goal is a
single book-keeping authority, not N. Online consumers may release one complete
DP micro-batch round at a time while retaining the larger optimizer-window
boundary for durable acknowledgements. An end-of-stream after a partially
released optimizer window fails loudly so those unacknowledged refs can be
replayed from the last durable checkpoint.

The consumed counter on the source channel (the producer's backpressure signal)
mirrors optimizer-durable work: each rank acknowledges its OWN inbox only after
the gathered SQLite ack/marker succeeds, and the distributor forwards the sum
of the inbox sidecars to the source counter. The consumer publishes the global
dispatch quantum before capture begins, and the producer rejects a watermark
smaller than that first window. Dispatch itself does not count as consumption.

If the distributor dies it drops a ``.failed`` sentinel (with the traceback)
into every inbox; :class:`InboxChannel` readers raise on it at the next poll —
a loud, immediate all-rank failure instead of a silent hang, and distinct from
the clean ``.closed`` end-of-stream.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import traceback
from collections import Counter
from typing import Iterable, List, Optional

from specforge.runtime.contracts import SampleRef
from specforge.runtime.data_plane.streaming_ref_channel import StreamingRefChannel

logger = logging.getLogger(__name__)

_FAILED_SUFFIX = ".failed"
_INBOX_SUFFIXES = (
    "",
    ".closed",
    ".consumed_count",
    ".consumed_count.tmp",
    _FAILED_SUFFIX,
)


class InboxChannel(StreamingRefChannel):
    """Reader view of a distributor inbox: fails loudly if the distributor died.

    ``StreamingRefQueue.get`` consults ``is_closed()`` every poll cycle, so a
    ``.failed`` sentinel converts "distributor thread died" from an unbounded
    hang into an immediate RuntimeError on every rank.
    """

    def is_closed(self) -> bool:
        failed = self.path + _FAILED_SUFFIX
        if os.path.exists(failed):
            with open(failed) as f:
                raise RuntimeError(f"ref-distributor died:\n{f.read()}")
        return super().is_closed()

    def failure(self) -> Optional[str]:
        failure = super().failure()
        if failure is None:
            return None
        return f"ref-distributor died:\n{failure}"


class RefDistributor:
    """Single-authority push dispatcher: source channel -> per-rank inboxes."""

    def __init__(
        self,
        source: StreamingRefChannel,
        controller,  # DataFlowController (rank 0's, with the run's durable store)
        inbox_dir: str,
        dp_size: int,
        *,
        feature_store,
        refs_per_rank_step: int,
        refs_per_rank_batch: Optional[int] = None,
        skip_ids: Optional[Iterable[str]] = None,
        requeued_ids: Optional[Iterable[str]] = None,
        worker_id: str = "ref-distributor",
        poll_s: float = 0.05,
        idle_timeout_s: Optional[float] = None,
        clock=time.monotonic,
        sleep=time.sleep,
    ) -> None:
        if dp_size < 1:
            raise ValueError(f"dp_size must be >= 1, got {dp_size}")
        if refs_per_rank_step < 1:
            raise ValueError(
                f"refs_per_rank_step must be >= 1, got {refs_per_rank_step}"
            )
        if refs_per_rank_batch is None:
            refs_per_rank_batch = refs_per_rank_step
        if refs_per_rank_batch < 1:
            raise ValueError(
                f"refs_per_rank_batch must be >= 1, got {refs_per_rank_batch}"
            )
        if refs_per_rank_step % refs_per_rank_batch:
            raise ValueError(
                "refs_per_rank_step must be divisible by refs_per_rank_batch: "
                f"{refs_per_rank_step} % {refs_per_rank_batch} != 0"
            )
        self.source = source
        self.controller = controller
        self.dp_size = dp_size
        self.feature_store = feature_store
        self.refs_per_rank_step = refs_per_rank_step
        self.refs_per_rank_batch = refs_per_rank_batch
        self.dispatch_quantum = dp_size * refs_per_rank_step
        self.dispatch_round_quantum = dp_size * refs_per_rank_batch
        self.worker_id = worker_id
        self.poll_s = poll_s
        self.idle_timeout_s = idle_timeout_s
        self._clock = clock
        self._sleep = sleep
        self._skip = set(skip_ids or ())
        self._requeued = set(requeued_ids or ())
        # Inboxes are EPHEMERAL (the ledger is the durable state): recreate them
        # fresh so a restarted run cannot replay a previous attempt's dispatch.
        # Rank 0 broadcasts setup success before any rank opens a reader.
        os.makedirs(inbox_dir, exist_ok=True)
        for rank in range(dp_size):
            base = self.inbox_path(inbox_dir, rank)
            for suffix in _INBOX_SUFFIXES:
                try:
                    os.remove(base + suffix)
                except FileNotFoundError:
                    pass
        self._inboxes = [
            StreamingRefChannel(self.inbox_path(inbox_dir, rank))
            for rank in range(dp_size)
        ]
        # Continue the producer-visible counter instead of rewinding it after a
        # consumer restart. A crash can occur after SQLite commit/feature abort
        # but before the rank-local inbox ack; repair that narrow window from
        # the durable released prefix before the producer applies backpressure.
        consumed = self.source.seed_consumed()
        if len(self._skip) > consumed:
            self.source.mark_consumed(len(self._skip) - consumed)
        self._inbox_consumed = 0  # last forwarded sum of the inbox sidecars
        self._window: List[SampleRef] = []
        self._window_dispatched = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.finished = False
        self.error: Optional[BaseException] = None
        self.stats = {
            "dispatched": 0,
            "skipped": 0,
            "duplicates": 0,
            "dropped": 0,
        }

    @staticmethod
    def inbox_path(inbox_dir: str, dp_rank: int) -> str:
        return os.path.join(inbox_dir, f"inbox-rank{dp_rank}.jsonl")

    def _forward_consumed(self) -> bool:
        """Mirror optimizer-durable inbox acks onto source backpressure."""
        consumed = sum(inbox.consumed_remote() for inbox in self._inboxes)
        delta = consumed - self._inbox_consumed
        if delta <= 0:
            return False
        self._inbox_consumed = consumed
        self.source.mark_consumed(delta)
        return True

    def pump(self) -> bool:
        """One non-blocking cycle: ingest + dispatch + counter. True on progress.

        Sets :attr:`finished` (and closes the inboxes) once the source is
        closed-and-drained and no full window remains.
        """
        if self.finished:
            return False
        source_failure = self.source.failure()
        if source_failure is not None:
            raise RuntimeError(f"source producer failed: {source_failure}")
        progress = False

        raw = self.source.poll()
        source_drained = False
        if not raw and self.source.is_closed():
            # The producer can append its final ref and close after the first
            # empty poll. Once close is observed no later publication is valid,
            # so this post-close poll is the authoritative drained check.
            raw = self.source.poll()
            source_drained = not raw
        if raw:
            progress = True
            candidates = []
            for ref in raw:
                if ref.sample_id in self._skip:
                    self._skip.discard(ref.sample_id)
                    self.stats["skipped"] += 1
                    continue
                candidates.append(ref)

            fresh = self.controller.commit_samples(self.worker_id, candidates)
            fresh_ids = Counter(ref.sample_id for ref in fresh)
            for ref in candidates:
                if fresh_ids[ref.sample_id]:
                    fresh_ids[ref.sample_id] -= 1
                    continue
                # Duplicate publication within an attempt is idempotent.
                self.stats["duplicates"] += 1
                if ref.sample_id in self._requeued:
                    # Reconciliation already put this unacked sample into the
                    # transient queue. Its inbox ack will settle the original
                    # publication exactly once.
                    self._requeued.discard(ref.sample_id)
                else:
                    # A same-attempt duplicate never reaches an inbox, so
                    # settle that extra publication immediately.
                    self.source.mark_consumed(1)

        # Release complete DP micro-batch rounds while tracking the surrounding
        # optimizer window. Round-robin assignment keeps every rank in lockstep;
        # durable acknowledgement still happens only after the full window.
        queue = self.controller.sample_queue
        while True:
            need = self.dispatch_round_quantum - len(self._window)
            if need:
                self._window.extend(queue.get(need, timeout_s=0.0))
            if len(self._window) < self.dispatch_round_quantum:
                break
            rank_batches: List[List[SampleRef]] = [[] for _ in range(self.dp_size)]
            for index, ref in enumerate(self._window):
                rank_batches[index % self.dp_size].append(ref)
            for inbox, refs in zip(self._inboxes, rank_batches):
                inbox.publish_batch(refs)
            self.stats["dispatched"] += self.dispatch_round_quantum
            self._window_dispatched += self.dispatch_round_quantum
            self._window = []
            if self._window_dispatched == self.dispatch_quantum:
                self._window_dispatched = 0
            progress = True

        if self._forward_consumed():
            progress = True

        if source_drained and queue.depth() == 0:
            self._finish()
        return progress

    def _finish(self) -> None:
        partial_dispatched = self._window_dispatched
        if self._window:
            self.stats["dropped"] = len(self._window)
            reason = (
                f"end-of-stream has {len(self._window)} refs after the last full "
                f"optimizer window (required global quantum={self.dispatch_quantum}: "
                f"dp_size={self.dp_size} * refs_per_rank_step="
                f"{self.refs_per_rank_step})"
            )
            self.controller.sample_queue.fail(
                self._window, reason=reason, retryable=False
            )
            cleanup_errors = []
            for ref in self._window:
                try:
                    adopt = getattr(self.feature_store, "adopt", None)
                    if callable(adopt):
                        adopt(ref)
                    self.feature_store.abort(ref.sample_id, reason=reason)
                except Exception as exc:  # best effort before the loud failure
                    cleanup_errors.append(f"{ref.sample_id}: {exc}")
            self.source.mark_consumed(len(self._window))
            self._window = []
            if cleanup_errors:
                reason += f"; cleanup errors={cleanup_errors}"
                raise RuntimeError(reason)
            logger.warning("ref-distributor: %s; tail settled without dispatch", reason)
        if partial_dispatched:
            raise RuntimeError(
                "end-of-stream after dispatching a partial optimizer window: "
                f"dispatched={partial_dispatched}/{self.dispatch_quantum} refs. "
                "The refs remain unacknowledged for replay from the last durable "
                "optimizer checkpoint."
            )
        for inbox in self._inboxes:
            inbox.close()
        self.finished = True
        logger.info("ref-distributor: finished %s", self.stats)

    def run(self) -> None:
        """Blocking loop; ends when the source is closed-and-drained or stop().

        On error the inboxes are NOT closed (a clean close would read as a
        normal end-of-stream and let ranks finish "successfully" on partial
        data) — ``_run_guarded`` drops a ``.failed`` sentinel instead, which
        every rank's :class:`InboxChannel` raises on at its next poll.
        """
        last_progress = self._clock()
        while not self._stop.is_set() and not self.finished:
            progress = self.pump()
            now = self._clock()
            if progress:
                last_progress = now
                continue
            if (
                self.idle_timeout_s is not None
                and now - last_progress > self.idle_timeout_s
            ):
                raise TimeoutError(
                    f"ref-distributor: no refs for {self.idle_timeout_s:.0f}s and "
                    f"the source channel is still open (producer dead?)"
                )
            self._sleep(self.poll_s)

    def _fail_inboxes(self, exc: BaseException) -> None:
        """Poison every inbox so all ranks fail loudly (never a silent hang —
        and never a clean close that would masquerade as end-of-stream)."""
        text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        for inbox in self._inboxes:
            try:
                inbox.fail(text)
            except OSError:  # best effort; ranks still have idle_timeout_s
                logger.exception("ref-distributor: could not poison %s", inbox.path)

    def _run_guarded(self) -> None:
        try:
            self.run()
        except BaseException as exc:  # noqa: BLE001 - surfaced via self.error
            self.error = exc
            logger.error("ref-distributor: died; poisoning inboxes: %s", exc)
            logger.debug("ref-distributor traceback", exc_info=True)
            self._fail_inboxes(exc)

    def start(self) -> "RefDistributor":
        self._thread = threading.Thread(
            target=self._run_guarded, name="ref-distributor", daemon=True
        )
        self._thread.start()
        return self

    def stop(self, join_timeout_s: float = 10.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=join_timeout_s)


__all__ = ["InboxChannel", "RefDistributor"]
