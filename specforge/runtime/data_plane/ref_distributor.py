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

    source channel -> commit (ONE ledger, dedup) -> aligned round-robin -> per-rank inbox

Each rank reads a private inbox (a plain consume-once :class:`StreamingRefQueue`)
and holds no channel offset, no partition math, no ledger — the design goal is a
single book-keeping authority, not N. Refs are dispatched in windows of exactly
``dp_size`` (one ref per rank per window) so every rank sees the SAME count:
the lockstep invariant FSDP collectives require. Anything less than a full
window at end-of-stream is dropped (logged; it stays committed-unacked in the
ledger, so a restart re-trains it).

The consumed counter on the source channel (the producer's backpressure signal)
is advanced by exactly two events, both through this object: restart-skips of
already-released refs, and the authority's durable ack
(``DPAckController.ack_sink -> note_acked``). Dispatch itself does NOT count —
in-flight must keep covering everything not yet trained.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Iterable, List, Optional

from specforge.runtime.contracts import SampleRef
from specforge.runtime.data_plane.streaming_ref_channel import StreamingRefChannel

logger = logging.getLogger(__name__)

_INBOX_SUFFIXES = ("", ".closed", ".consumed_count", ".consumed_count.tmp")


class RefDistributor:
    """Single-authority push dispatcher: source channel -> per-rank inboxes."""

    def __init__(
        self,
        source: StreamingRefChannel,
        controller,  # DataFlowController (rank 0's, with the run's durable store)
        inbox_dir: str,
        dp_size: int,
        *,
        skip_ids: Optional[Iterable[str]] = None,
        worker_id: str = "ref-distributor",
        poll_s: float = 0.05,
        idle_timeout_s: Optional[float] = None,
        clock=time.monotonic,
        sleep=time.sleep,
    ) -> None:
        if dp_size < 1:
            raise ValueError(f"dp_size must be >= 1, got {dp_size}")
        self.source = source
        self.controller = controller
        self.dp_size = dp_size
        self.worker_id = worker_id
        self.poll_s = poll_s
        self.idle_timeout_s = idle_timeout_s
        self._clock = clock
        self._sleep = sleep
        self._skip = set(skip_ids) if skip_ids else None
        # Inboxes are EPHEMERAL (the ledger is the durable state): recreate them
        # fresh so a restarted run cannot replay a previous attempt's dispatch.
        # The launcher barriers ranks AFTER construction, before readers open.
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
        # A restarted consumer must not rewind the producer's counter.
        self.source.seed_consumed()
        # mark_consumed is not thread-safe; acks (trainer thread) and skips
        # (this thread) serialize here.
        self._consume_lock = threading.Lock()
        self._window: List[SampleRef] = []  # leased, awaiting a full dp_size window
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.finished = False
        self.error: Optional[BaseException] = None
        self.stats = {"dispatched": 0, "skipped": 0, "duplicates": 0, "dropped": 0}

    @staticmethod
    def inbox_path(inbox_dir: str, dp_rank: int) -> str:
        return os.path.join(inbox_dir, f"inbox-rank{dp_rank}.jsonl")

    def note_acked(self, n: int) -> None:
        """Advance the producer's backpressure counter for n durably-acked refs."""
        if n:
            with self._consume_lock:
                self.source.mark_consumed(n)

    def pump(self) -> bool:
        """One non-blocking cycle: ingest + dispatch. Returns True on progress.

        Sets :attr:`finished` (and closes the inboxes) once the source is
        closed-and-drained and no full window remains.
        """
        if self.finished:
            return False
        progress = False

        raw = self.source.poll()
        if raw:
            progress = True
            skipped = 0
            for ref in raw:
                if self._skip and ref.sample_id in self._skip:
                    self._skip.discard(ref.sample_id)
                    skipped += 1
                    if not self._skip:
                        self._skip = None
                    continue
                before = self.controller.store.committed_count()
                self.controller.commit_samples(self.worker_id, [ref])
                if self.controller.store.committed_count() == before:
                    # duplicate publication (e.g. producer restart); a
                    # reconcile-requeued ref also lands here and trains later.
                    self.stats["duplicates"] += 1
            if skipped:
                self.stats["skipped"] += skipped
                self.note_acked(skipped)  # already trained: keep in-flight exact

        # Dispatch every full window: one ref per rank, round-robin — per-rank
        # counts stay exactly equal, which is what keeps DP ranks in lockstep.
        queue = self.controller.sample_queue
        while True:
            need = self.dp_size - len(self._window)
            if need:
                self._window.extend(queue.get(need, timeout_s=0.0))
            if len(self._window) < self.dp_size:
                break
            for rank, ref in enumerate(self._window):
                self._inboxes[rank].publish(ref)
            self.stats["dispatched"] += self.dp_size
            self._window = []
            progress = True

        if not raw and self.source.is_closed() and queue.depth() == 0:
            self._finish()
        return progress

    def _finish(self) -> None:
        if self._window:
            # Committed-but-unacked in the ledger: a restart re-trains them.
            self.stats["dropped"] = len(self._window)
            logger.warning(
                "ref-distributor: dropping %d end-of-stream refs (< dp_size=%d "
                "window); they stay committed-unacked and replay on restart",
                len(self._window),
                self.dp_size,
            )
            self._window = []
        for inbox in self._inboxes:
            inbox.close()
        self.finished = True
        logger.info("ref-distributor: finished %s", self.stats)

    def run(self) -> None:
        """Blocking loop; ends when the source is closed-and-drained or stop().

        On error the inboxes are deliberately NOT closed: a clean close would
        read as a normal end-of-stream and let ranks finish "successfully" on
        partial data — instead they surface a loud idle TimeoutError.
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

    def _run_guarded(self) -> None:
        try:
            self.run()
        except BaseException as exc:  # noqa: BLE001 - surfaced via self.error
            self.error = exc
            logger.exception("ref-distributor: died; DP ranks will idle-timeout")

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


__all__ = ["RefDistributor"]
