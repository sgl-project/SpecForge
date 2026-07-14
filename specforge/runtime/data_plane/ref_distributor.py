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
single book-keeping authority, not N. Refs are dispatched only in complete
``dp_size * refs_per_rank_step`` windows. Each rank receives exactly
``refs_per_rank_step = batch_size * accumulation_steps`` refs, so the stream
cannot end on a partial batch or an unreduced FSDP accumulation step. An
unaligned end-of-stream fails the attempt after cleaning the undispatched refs.

The consumed counter on the source channel (the producer's backpressure signal)
mirrors what the ranks actually consumed: each rank's loader acks its OWN inbox
per micro-batch, and the distributor forwards the sum of the inbox sidecars to
the source counter. The consumer publishes the global dispatch quantum before
capture begins, and the producer rejects a watermark smaller than that first
window. After dispatch starts, per-micro-batch acks release capacity
incrementally. Dispatch itself does not count as consumption.

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
from typing import List, Optional

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
        self.source = source
        self.controller = controller
        self.dp_size = dp_size
        self.feature_store = feature_store
        self.refs_per_rank_step = refs_per_rank_step
        self.dispatch_quantum = dp_size * refs_per_rank_step
        self.worker_id = worker_id
        self.poll_s = poll_s
        self.idle_timeout_s = idle_timeout_s
        self._clock = clock
        self._sleep = sleep
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
        self._inbox_consumed = 0  # last forwarded sum of the inbox sidecars
        self._window: List[SampleRef] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.finished = False
        self.error: Optional[BaseException] = None
        self.stats = {"dispatched": 0, "duplicates": 0, "dropped": 0}
        # PROFILE_DISTRIB=<secs> -> periodic [dist] line: polled/dispatched
        # rates, per-ref commit + count cost, inbox-publish cost, rank lag.
        self._prof_s = float(os.environ.get("PROFILE_DISTRIB", "0"))
        self._pd = {
            "polled": 0,
            "disp0": 0,
            "commit": 0.0,
            "cnt": 0.0,
            "inboxpub": 0.0,
            "t0": time.monotonic(),
        }

    @staticmethod
    def inbox_path(inbox_dir: str, dp_rank: int) -> str:
        return os.path.join(inbox_dir, f"inbox-rank{dp_rank}.jsonl")

    def _forward_consumed(self) -> bool:
        """Mirror the ranks' inbox acks onto the source counter (backpressure).

        After the first complete optimizer window is dispatched, per-micro-batch
        granularity lets the producer reuse capacity without waiting for the
        whole optimizer step to finish.
        """
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
        if raw:
            progress = True
            self._pd["polled"] += len(raw)
            for ref in raw:
                _t = time.monotonic()
                before = self.controller.store.committed_count()
                self._pd["cnt"] += time.monotonic() - _t
                _t = time.monotonic()
                self.controller.commit_samples(self.worker_id, [ref])
                self._pd["commit"] += time.monotonic() - _t
                _t = time.monotonic()
                if self.controller.store.committed_count() == before:
                    # Duplicate publication within an attempt is idempotent.
                    self.stats["duplicates"] += 1
                    # It will never reach an inbox, so settle its source-channel
                    # publication immediately instead of pinning backpressure.
                    self.source.mark_consumed(1)
                self._pd["cnt"] += time.monotonic() - _t

        # Dispatch complete optimizer-step windows. Round-robin assignment gives
        # every rank batch_size * accumulation_steps refs in the same order.
        queue = self.controller.sample_queue
        while True:
            need = self.dispatch_quantum - len(self._window)
            if need:
                self._window.extend(queue.get(need, timeout_s=0.0))
            if len(self._window) < self.dispatch_quantum:
                break
            _t = time.monotonic()
            for index, ref in enumerate(self._window):
                self._inboxes[index % self.dp_size].publish(ref)
            self._pd["inboxpub"] += time.monotonic() - _t
            self.stats["dispatched"] += self.dispatch_quantum
            self._window = []
            progress = True

        if self._forward_consumed():
            progress = True

        if self._prof_s:
            now = time.monotonic()
            win = now - self._pd["t0"]
            if win >= self._prof_s:
                pd = self._pd
                n = max(1, pd["polled"])
                disp = self.stats["dispatched"] - pd["disp0"]
                lag = [
                    inbox._published - inbox.consumed_remote()
                    for inbox in self._inboxes
                ]
                print(
                    f"[dist] win_s={win:.1f} polled={pd['polled']} disp={disp} "
                    f"q={queue.depth()} dup={self.stats['duplicates']} "
                    f"cnt_ms={1000*pd['cnt']/n:.2f} commit_ms={1000*pd['commit']/n:.2f} "
                    f"inboxpub_ms={1000*pd['inboxpub']/n:.2f} lag={lag} (per-ref avg)",
                    flush=True,
                )
                self._pd = {
                    "polled": 0,
                    "disp0": self.stats["dispatched"],
                    "commit": 0.0,
                    "cnt": 0.0,
                    "inboxpub": 0.0,
                    "t0": now,
                }

        if not raw and self.source.is_closed() and queue.depth() == 0:
            self._finish()
        return progress

    def _finish(self) -> None:
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
                    self.feature_store.abort(ref.sample_id, reason=reason)
                except Exception as exc:  # best effort before the loud failure
                    cleanup_errors.append(f"{ref.sample_id}: {exc}")
            self.source.mark_consumed(len(self._window))
            self._window = []
            if cleanup_errors:
                reason += f"; cleanup errors={cleanup_errors}"
            raise RuntimeError(reason)
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
