# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""StreamingRefChannel: a cross-process, append-only ``SampleRef`` stream.

The offline disaggregation path hands the consumer a *static* ref manifest
(:func:`disagg_ingest.write_ref_manifest`) written once before training. The
*online* disaggregated path needs the opposite: the rollout producer commits
refs continuously while the trainer consumes them, on a different node. This is
that control-plane channel.

* **Tensor-free.** Only ``SampleRef`` metadata travels here (asserted on
  publish); the feature tensors go through the ``FeatureStore`` (Mooncake), so a
  shared *data* mount is never required -- only this small control file.
* **Append-only JSONL.** ``publish()`` appends one ref per line and fsyncs;
  ``poll()`` tail-reads complete lines from the last offset, buffering a partial
  trailing line so a reader never parses a half-written record.
* **Consume-once friendly.** The reader marks how many refs it has consumed
  (``mark_consumed``) in a sidecar counter the writer reads back, so the producer
  can apply backpressure (``in_flight_remote``) without any shared in-process
  state -- the same split-process model as the rest of disaggregation.
* **Explicit close.** ``close()`` drops an EOF sentinel so the reader's
  :meth:`stream` terminates once the file is drained instead of polling forever.

This is the framework, intentionally filesystem-backed (works over any shared
control mount). A networked control plane (Redis/the durable MetadataStore)
slots in behind the same publish/poll API later.
"""

from __future__ import annotations

import json
import os
import time
from typing import Iterator, List, Optional, Set

from specforge.runtime.contracts import SampleRef, assert_no_tensors
from specforge.runtime.data_plane.disagg_ingest import _ref_from_dict, _ref_to_dict

_CLOSED_SUFFIX = ".closed"
_CONSUMED_SUFFIX = ".consumed_count"


class StreamingRefChannel:
    """An append-only, tensor-free ``SampleRef`` stream over a shared file.

    One instance per process. The *producer* calls :meth:`publish`/:meth:`close`
    (+ reads :meth:`in_flight_remote` for backpressure); the *consumer* calls
    :meth:`poll`/:meth:`stream` (+ :meth:`mark_consumed`). Both point at the same
    ``path`` on a shared control mount.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        # producer-side
        self._published = 0
        # consumer-side
        self._read_offset = 0
        self._buf = ""
        self._consumed = 0

    # -- producer ----------------------------------------------------------
    def publish(self, ref: SampleRef) -> None:
        """Append one tensor-free ref. Durable (fsync) so a remote reader sees it."""
        assert_no_tensors([ref])
        line = json.dumps(_ref_to_dict(ref), separators=(",", ":")) + "\n"
        with open(self.path, "a") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        self._published += 1

    def publish_many(self, refs: List[SampleRef]) -> None:
        for r in refs:
            self.publish(r)

    def close(self) -> None:
        """Drop the EOF sentinel so the reader's stream() terminates when drained."""
        open(self.path + _CLOSED_SUFFIX, "w").close()

    @property
    def published(self) -> int:
        return self._published

    def consumed_remote(self) -> int:
        """How many refs the consumer reports having consumed (cross-process)."""
        try:
            with open(self.path + _CONSUMED_SUFFIX) as f:
                return int(f.read() or "0")
        except (FileNotFoundError, ValueError):
            return 0

    def in_flight_remote(self) -> int:
        """published - consumed: the producer's backpressure signal."""
        return self._published - self.consumed_remote()

    # -- consumer ----------------------------------------------------------
    def is_closed(self) -> bool:
        """True once the producer has dropped the EOF sentinel."""
        return os.path.exists(self.path + _CLOSED_SUFFIX)

    _is_closed = is_closed  # backwards-compatible alias

    def poll(self, max_n: Optional[int] = None) -> List[SampleRef]:
        """Return refs appended since the last poll (complete lines only).

        Non-blocking: returns whatever is available now (possibly empty). A
        partially written trailing line is buffered until its newline arrives, so
        a ref is never parsed half-written.
        """
        try:
            with open(self.path, "r") as f:
                f.seek(self._read_offset)
                chunk = f.read()
                self._read_offset = f.tell()
        except FileNotFoundError:
            chunk = ""
        if chunk:
            self._buf += chunk
        # parse the buffer even when no new bytes arrived -- a previous max_n call
        # may have left complete lines buffered.
        out: List[SampleRef] = []
        while "\n" in self._buf:
            if max_n is not None and len(out) >= max_n:
                break
            line, self._buf = self._buf.split("\n", 1)
            line = line.strip()
            if line:
                out.append(_ref_from_dict(json.loads(line)))
        return out

    def mark_consumed(self, n: int) -> None:
        """Record n more consumed refs in the sidecar the producer reads back."""
        self._consumed += n
        tmp = self.path + _CONSUMED_SUFFIX + ".tmp"
        with open(tmp, "w") as f:
            f.write(str(self._consumed))
        os.replace(tmp, self.path + _CONSUMED_SUFFIX)  # atomic counter publish

    def seed_consumed(self) -> int:
        """Adopt the sidecar's persisted count as this instance's baseline.

        ``mark_consumed`` publishes this instance's cumulative in-memory count,
        so a restarted consumer (fresh instance, ``_consumed=0``) would rewind
        the sidecar and inflate the producer's ``in_flight_remote`` forever.
        Call once on restart, before the first ``mark_consumed``.
        """
        self._consumed = self.consumed_remote()
        return self._consumed

    def stream(
        self,
        *,
        poll_s: float = 0.25,
        idle_timeout_s: Optional[float] = None,
        clock=time.monotonic,
        sleep=time.sleep,
    ) -> Iterator[SampleRef]:
        """Yield refs until the channel is closed AND drained.

        Blocks (polling every ``poll_s``) while the producer is still live.
        ``idle_timeout_s`` (if set) raises ``TimeoutError`` after that long with
        no new ref and no close sentinel -- a liveness guard against a dead
        producer.
        """
        last_progress = clock()
        while True:
            batch = self.poll()
            if batch:
                last_progress = clock()
                for ref in batch:
                    yield ref
                continue
            if self._is_closed():
                # one final drain after close, then stop
                tail = self.poll()
                for ref in tail:
                    yield ref
                if not tail:
                    return
                continue
            if idle_timeout_s is not None and clock() - last_progress > idle_timeout_s:
                raise TimeoutError(
                    f"StreamingRefChannel {self.path}: no refs for "
                    f"{idle_timeout_s:.0f}s and not closed (producer dead?)"
                )
            sleep(poll_s)


class StreamingRefQueue:
    """Adapts a :class:`StreamingRefChannel` to the ``SampleRefQueue`` protocol
    (``get``/``ack``/``fail``) the ``FeatureDataLoader`` consumes in queue mode.

    ``get(n)`` BLOCKS (polling) until ``n`` refs are buffered or the channel is
    closed-and-drained, so the trainer streams the whole online run and only sees
    an empty batch (-> loop ends) once the producer has closed. ``ack`` advances
    the channel's consumed counter (the producer's backpressure signal); the
    feature-store free already happened in the loader's materialize (get ->
    release) for a consume-once store.

    ``skip_ids`` is the restart contract (O1.1): the channel JSONL is append-only
    and a restarted consumer re-reads it from offset 0, so without a skip set it
    would re-train every already-durably-acked sample. The launcher derives this
    set from the shared durable :class:`MetadataStore` (the samples
    ``reconcile_on_restart`` reports as released) and passes it here; matching refs
    are dropped on read and counted as consumed so the producer's backpressure
    signal (``in_flight_remote``) stays exact across the restart.
    """

    def __init__(
        self,
        channel: StreamingRefChannel,
        *,
        poll_s: float = 0.1,
        idle_timeout_s: Optional[float] = None,
        skip_ids: Optional[Set[str]] = None,
        clock=time.monotonic,
        sleep=time.sleep,
    ) -> None:
        self.channel = channel
        self.poll_s = poll_s
        self.idle_timeout_s = idle_timeout_s
        self._skip_ids = set(skip_ids) if skip_ids else None
        self._clock = clock
        self._sleep = sleep
        self._buf: List[SampleRef] = []

    def _poll(self) -> "tuple[List[SampleRef], int]":
        """Poll the channel, dropping already-trained refs (restart skip).

        Returns ``(fresh_refs, raw_count)``. ``raw_count`` is the number of refs
        the channel actually yielded (skipped or not), so the caller can tell
        "made progress" from "nothing new" even when every polled ref was skipped.
        """
        raw = self.channel.poll()
        if not self._skip_ids or not raw:
            return raw, len(raw)
        fresh: List[SampleRef] = []
        skipped = 0
        for ref in raw:
            if ref.sample_id in self._skip_ids:
                self._skip_ids.discard(ref.sample_id)
                skipped += 1
            else:
                fresh.append(ref)
        if skipped:
            # already durably trained on a prior run: count them consumed so the
            # producer's in_flight_remote backpressure stays exact across restart.
            self.channel.mark_consumed(skipped)
            if not self._skip_ids:
                # The restart skip-ids are the already-trained FRONT prefix, so once
                # they are all drained drop back to the zero-overhead fast path
                # instead of hashing every ref for the rest of a long online run.
                self._skip_ids = None
        return fresh, len(raw)

    def get(self, n: int, timeout_s: float = 0.0) -> List[SampleRef]:
        last_progress = self._clock()
        while len(self._buf) < n:
            new, raw = self._poll()
            if raw:
                # progress on the channel (even if all refs were skipped): buffer
                # what survived the skip filter and poll again before sleeping.
                last_progress = self._clock()
                if new:
                    self._buf.extend(new)
                continue
            if self.channel.is_closed():
                drain, _ = self._poll()
                self._buf.extend(drain)  # final drain
                break
            if (
                self.idle_timeout_s is not None
                and self._clock() - last_progress > self.idle_timeout_s
            ):
                raise TimeoutError(
                    f"StreamingRefQueue {self.channel.path}: idle "
                    f"{self.idle_timeout_s:.0f}s with the channel still open"
                )
            self._sleep(self.poll_s)
        take = min(n, len(self._buf))
        out, self._buf = self._buf[:take], self._buf[take:]
        return out

    def ack(self, refs: List[SampleRef]) -> None:
        self.channel.mark_consumed(len(refs))  # backpressure: producer reads this

    def fail(self, refs: List[SampleRef], reason: str, retryable: bool) -> None:
        if retryable:
            self._buf[:0] = refs  # re-buffer at the front for the next get()
        else:
            self.channel.mark_consumed(len(refs))

    def depth(self) -> int:
        return len(self._buf)


__all__ = ["StreamingRefChannel", "StreamingRefQueue"]
