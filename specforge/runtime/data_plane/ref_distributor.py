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
single book-keeping authority, not N. Refs are released one complete DP
micro-batch round at a time, but an optimizer window is OPENED only once the
whole window is already committed locally: a released round obligates every
rank to a full accumulation window, and durable acknowledgements only advance
in whole windows, so a window that could not complete would strand refs that no
resume can ever acknowledge. An end-of-stream therefore always lands on a
window boundary; the sub-window leftover (fewer refs than one global window)
is settled without dispatch — drop_last-style, matching the floor semantics of
``resolve_online_total_steps`` — and the stream closes cleanly. Note the
cold-start consequence: ranks receive their first micro-batch only after the
whole first window is captured, so rank-side idle timeouts must cover
full-window capture latency, not a single round.

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
                    # Released samples stay skip-listed for the whole replay:
                    # the prior attempt settled EVERY publication it polled
                    # (ack for the fresh one, immediate settle for duplicates)
                    # into the persisted consumed sidecar this attempt seeded
                    # from. Settling a re-polled duplicate again would advance
                    # consumed past the producer's published accounting and
                    # crash its reconciliation; a duplicate the prior attempt
                    # never polled merely under-counts, which only tightens
                    # backpressure and is safe.
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
            if self._window_dispatched == 0:
                # Open a new optimizer window only when the WHOLE window is
                # already committed locally. Dispatching the first round
                # obligates every rank to a full accumulation window, and acks
                # advance only in whole windows — so a window opened without
                # its full quantum secured could end mid-accumulation at
                # end-of-stream, stranding refs no resume can ever settle.
                # Once secured, the window completes within this same loop.
                if len(self._window) + queue.depth() < self.dispatch_quantum:
                    break
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

        if source_drained:
            self._finish()
        return progress

    def _finish(self) -> None:
        # The dispatch loop only opens an optimizer window once its whole
        # quantum is committed locally and then completes it within the same
        # loop, so end-of-stream always lands on a window boundary: every
        # dispatched ref belongs to a completed window whose acks the ranks
        # can durably deliver. Guard that invariant loudly — settling refs of
        # a dispatched window here would corrupt the ack accounting.
        if self._window_dispatched:
            raise AssertionError(
                "internal invariant violated: a dispatched optimizer window "
                "must complete within the pump cycle that opened it "
                f"(dispatched={self._window_dispatched}/{self.dispatch_quantum})"
            )
        leftover = list(self._window)
        self._window = []
        queue = self.controller.sample_queue
        while True:
            batch = queue.get(self.dispatch_quantum, timeout_s=0.0)
            if not batch:
                break
            leftover.extend(batch)
        if leftover:
            self.stats["dropped"] = len(leftover)
            reason = (
                f"end-of-stream leaves {len(leftover)} refs, fewer than one "
                f"optimizer window (required global quantum="
                f"{self.dispatch_quantum}: dp_size={self.dp_size} * "
                f"refs_per_rank_step={self.refs_per_rank_step}); settled "
                "without dispatch so every rank finishes on a completed window"
            )
            queue.fail(leftover, reason=reason, retryable=False)
            cleanup_errors = []
            for ref in leftover:
                try:
                    adopt = getattr(self.feature_store, "adopt", None)
                    if callable(adopt):
                        adopt(ref)
                    self.feature_store.abort(ref.sample_id, reason=reason)
                except Exception as exc:  # best effort before the loud failure
                    cleanup_errors.append(f"{ref.sample_id}: {exc}")
            self.source.mark_consumed(len(leftover))
            if cleanup_errors:
                reason += f"; cleanup errors={cleanup_errors}"
                raise RuntimeError(reason)
            logger.warning("ref-distributor: %s", reason)
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
