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
from typing import Iterator, List, Optional

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
    def _is_closed(self) -> bool:
        return os.path.exists(self.path + _CLOSED_SUFFIX)

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


__all__ = ["StreamingRefChannel"]
