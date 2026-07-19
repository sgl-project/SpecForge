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
* **Explicit outcome.** ``close()`` drops an EOF sentinel only for a successful
  producer. ``fail()`` publishes the producer exception separately, so a remote
  trainer cannot mistake a truncated rollout for a normal end of input.
* **Peer stop.** The consumer publishes done/failed sentinels. A producer that is
  blocked on its in-flight watermark can then stop without process supervision.

This is the framework, intentionally filesystem-backed (works over any shared
control mount). A networked control plane (Redis/the durable MetadataStore)
slots in behind the same publish/poll API later.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence

from specforge.runtime.contracts import SampleRef, assert_no_tensors
from specforge.runtime.data_plane.ref_serialization import ref_from_dict, ref_to_dict

_CLOSED_SUFFIX = ".closed"
_CONSUMED_SUFFIX = ".consumed_count"
_FAILED_SUFFIX = ".failed"
_CONSUMER_DONE_SUFFIX = ".consumer_done"
_CONSUMER_FAILED_SUFFIX = ".consumer_failed"
_CONSUMER_QUANTUM_SUFFIX = ".consumer_quantum"


@dataclass
class RefPublishTransaction:
    """Track the ownership-transferred prefix of one publication batch.

    JSONL append is not atomic across a batch. A complete line can also become
    visible before its fsync reports failure. The channel increments its
    publication counter once the full line has been accepted; the transaction
    observes that progress even when ``publish`` raises. Cleanup can therefore
    preserve every possibly visible ref and abort only the untouched suffix.
    """

    channel: "StreamingRefChannel"
    refs: tuple[SampleRef, ...]
    published_count: int = 0

    @property
    def published_refs(self) -> tuple[SampleRef, ...]:
        return self.refs[: self.published_count]

    @property
    def unpublished_refs(self) -> tuple[SampleRef, ...]:
        return self.refs[self.published_count :]

    def commit(self) -> None:
        """Publish the remaining suffix, retaining progress if one append fails."""

        for ref in self.unpublished_refs:
            before = self.channel.published
            try:
                self.channel.publish(ref)
            except BaseException as publish_exc:
                transferred = self.channel.published - before
                if transferred not in (0, 1):
                    raise RuntimeError(
                        "reference channel reported invalid publication progress: "
                        f"{before} -> {self.channel.published}"
                    ) from publish_exc
                self.published_count += transferred
                raise
            else:
                self.published_count += 1


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
        # ``mark_consumed`` is called from both the trainer thread (batch acks)
        # and the prefetch worker (failure settlement), so the increment and
        # the sidecar publish must be one atomic section.
        self._consumed_lock = threading.Lock()

    # -- producer ----------------------------------------------------------
    def publish(self, ref: SampleRef) -> None:
        """Append one tensor-free ref. Durable (fsync) so a remote reader sees it."""
        assert_no_tensors([ref])
        payload = (json.dumps(ref_to_dict(ref), separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        fd = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            offset = 0
            while offset < len(payload):
                written = os.write(fd, payload[offset:])
                if written <= 0:
                    raise OSError("reference channel append made no progress")
                offset += written
            # Ownership transfers only after the complete newline-terminated
            # record has reached the kernel. fsync may still report a durability
            # failure after the record is visible, so progress must be observable
            # to the surrounding transaction before that call.
            self._published += 1
            os.fsync(fd)
        finally:
            os.close(fd)

    def publish_batch(self, refs: Sequence[SampleRef]) -> None:
        """Append a ref batch with one open/write cycle and durability barrier.

        Complete JSONL records count as published as soon as their bytes reach
        the kernel, including when a later write or the final fsync fails. This
        matches :meth:`publish` ownership semantics while allowing ephemeral DP
        inboxes to amortize one fsync across an optimizer dispatch window.
        """

        refs = tuple(refs)
        if not refs:
            return
        assert_no_tensors(refs)
        records = [
            (json.dumps(ref_to_dict(ref), separators=(",", ":")) + "\n").encode("utf-8")
            for ref in refs
        ]
        payload = b"".join(records)
        record_ends = []
        end = 0
        for record in records:
            end += len(record)
            record_ends.append(end)

        fd = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            offset = 0
            published_in_batch = 0
            while offset < len(payload):
                written = os.write(fd, payload[offset:])
                if written <= 0:
                    raise OSError("reference channel batch append made no progress")
                offset += written
                while (
                    published_in_batch < len(record_ends)
                    and record_ends[published_in_batch] <= offset
                ):
                    self._published += 1
                    published_in_batch += 1
            os.fsync(fd)
        finally:
            os.close(fd)

    def begin_publish(self, refs: Sequence[SampleRef]) -> RefPublishTransaction:
        """Create an observable batch publication boundary without writing yet."""

        return RefPublishTransaction(self, tuple(refs))

    def publish_many(self, refs: List[SampleRef]) -> None:
        self.begin_publish(refs).commit()

    def close(self) -> None:
        """Drop the EOF sentinel so the reader's stream() terminates when drained."""
        self._write_sidecar(_CLOSED_SUFFIX, "")

    def fail(self, reason: str) -> None:
        """Publish a terminal producer failure for the remote consumer."""
        self._write_sidecar(_FAILED_SUFFIX, reason or "producer failed")

    def failure(self) -> Optional[str]:
        """Return the producer failure reason, if one has been published."""
        return self._read_sidecar(_FAILED_SUFFIX)

    def mark_consumer_done(self) -> None:
        """Tell a split producer that training completed successfully."""
        self._write_sidecar(_CONSUMER_DONE_SUFFIX, "")

    def mark_consumer_failed(self, reason: str) -> None:
        """Tell a split producer that training failed and it must stop."""
        self._write_sidecar(_CONSUMER_FAILED_SUFFIX, reason or "consumer failed")

    def consumer_failure(self) -> Optional[str]:
        """Return the consumer failure reason, if one has been published."""
        return self._read_sidecar(_CONSUMER_FAILED_SUFFIX)

    def consumer_stopped(self) -> bool:
        """Whether the remote consumer completed or failed."""
        return os.path.exists(self.path + _CONSUMER_DONE_SUFFIX) or os.path.exists(
            self.path + _CONSUMER_FAILED_SUFFIX
        )

    def publish_consumer_quantum(
        self,
        refs_per_optimizer_step: int,
        *,
        allow_existing: bool = False,
    ) -> None:
        """Claim and publish the consumer's global optimizer-step ref quantum.

        Rank 0 calls this exactly once for a fresh attempt. The producer waits
        for the value before capture so its in-flight watermark cannot deadlock
        below the number of refs needed for one lockstep optimizer step.
        """
        quantum = int(refs_per_optimizer_step)
        if quantum < 1:
            raise ValueError(f"consumer quantum must be >= 1, got {quantum}")
        path = self.path + _CONSUMER_QUANTUM_SUFFIX
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError as exc:
            if allow_existing:
                existing = self.consumer_quantum()
                if existing == quantum:
                    return
                raise ValueError(
                    f"consumer quantum changed across resume for {self.path!r}: "
                    f"existing={existing}, requested={quantum}"
                ) from exc
            raise ValueError(
                f"consumer quantum already exists for {self.path!r}; every online "
                "attempt requires a fresh reference channel"
            ) from exc
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(str(quantum))
            stream.flush()
            os.fsync(stream.fileno())

    def consumer_quantum(self) -> Optional[int]:
        """Return the consumer's global optimizer-step ref quantum, if ready."""
        path = self.path + _CONSUMER_QUANTUM_SUFFIX
        try:
            with open(path, encoding="utf-8") as stream:
                value = int(stream.read())
        except FileNotFoundError:
            return None
        except ValueError as exc:
            raise RuntimeError(f"invalid consumer quantum at {path}") from exc
        if value < 1:
            raise RuntimeError(f"invalid consumer quantum {value} at {path}")
        return value

    def _write_sidecar(self, suffix: str, value: str) -> None:
        tmp = f"{self.path}{suffix}.tmp.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, self.path + suffix)

    def _read_sidecar(self, suffix: str) -> Optional[str]:
        try:
            with open(self.path + suffix, encoding="utf-8") as stream:
                return stream.read().strip() or "unknown remote failure"
        except FileNotFoundError:
            return None

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
                out.append(ref_from_dict(json.loads(line)))
        return out

    def mark_consumed(self, n: int) -> None:
        """Record n more consumed refs in the sidecar the producer reads back."""
        with self._consumed_lock:
            self._consumed += n
            # Thread-unique tmp name: a concurrent writer must never observe
            # (or os.replace away) another writer's half-written counter file.
            tmp = (
                f"{self.path}{_CONSUMED_SUFFIX}"
                f".tmp.{os.getpid()}.{threading.get_ident()}"
            )
            with open(tmp, "w") as f:
                f.write(str(self._consumed))
            os.replace(tmp, self.path + _CONSUMED_SUFFIX)  # atomic counter publish

    def seed_consumed(self) -> int:
        """Continue an existing consumed counter after a consumer restart."""
        with self._consumed_lock:
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
            failure = self.failure()
            if failure is not None:
                raise RuntimeError(
                    f"StreamingRefChannel {self.path}: producer failed: {failure}"
                )
            if self.is_closed():
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
    online disaggregated consumers use a retaining store, so loader release ends
    only the read lease. The durable optimizer-boundary controller owns the
    later physical feature deletion.

    """

    # A loader that stops after prefetching must put its never-yielded refs back
    # without advancing the durable consumed counter. The retaining disagg
    # feature store keeps those refs materializable across that retry.
    loader_close_retryable = True

    def __init__(
        self,
        channel: StreamingRefChannel,
        *,
        poll_s: float = 0.1,
        idle_timeout_s: Optional[float] = None,
        clock=time.monotonic,
        sleep=time.sleep,
    ) -> None:
        self.channel = channel
        self.poll_s = poll_s
        self.idle_timeout_s = idle_timeout_s
        self._clock = clock
        self._sleep = sleep
        self._buf: List[SampleRef] = []
        # ``get`` may run on the loader-prefetch thread while the optimizer
        # boundary acknowledges an earlier prefix on the training thread.
        # Track that prefix explicitly so a durable ack can advance channel
        # backpressure only for the exact refs it covered.
        self._inflight: List[SampleRef] = []
        self._inflight_lock = threading.Lock()
        self._wait_log_interval_s = float(
            os.environ.get("SPECFORGE_STREAM_WAIT_LOG_INTERVAL", 30.0)
        )
        self._last_wait_log = 0.0

    def get(self, n: int, timeout_s: float = 0.0) -> List[SampleRef]:
        del timeout_s  # the durable stream blocks until closed, failed, or timed out
        return self._get(n)

    def get_interruptible(
        self,
        n: int,
        *,
        stop_event: threading.Event,
    ) -> List[SampleRef]:
        """Blocking ``get`` variant that a loader-owned worker can stop.

        Normal :meth:`get` retains its closed-and-drained blocking contract.
        This variant checks the loader event between channel polls; refs already
        buffered locally remain unconsumed and therefore recoverable when the
        loader stops.
        """
        return self._get(n, stop_event=stop_event)

    def _get(
        self,
        n: int,
        *,
        stop_event: Optional[threading.Event] = None,
    ) -> List[SampleRef]:
        last_progress = self._clock()
        while len(self._buf) < n:
            if stop_event is not None and stop_event.is_set():
                return []
            new = self.channel.poll()
            if new:
                last_progress = self._clock()
                self._buf.extend(new)
                continue
            failure = self.channel.failure()
            if failure is not None:
                raise RuntimeError(
                    f"StreamingRefChannel {self.channel.path}: producer failed: "
                    f"{failure}"
                )
            if self.channel.is_closed():
                drain = self.channel.poll()
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
            now = self._clock()
            if (
                self._wait_log_interval_s > 0
                and now - self._last_wait_log >= self._wait_log_interval_s
            ):
                print(
                    "[stream-ref-queue] "
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} waiting "
                    f"path={self.channel.path} need={n} buffered={len(self._buf)} "
                    f"closed={self.channel.is_closed()} "
                    f"since_progress={now - last_progress:.1f}s",
                    flush=True,
                )
                self._last_wait_log = now
            if stop_event is None:
                self._sleep(self.poll_s)
            else:
                stop_event.wait(self.poll_s)
        if stop_event is not None and stop_event.is_set():
            return []
        take = min(n, len(self._buf))
        out, self._buf = self._buf[:take], self._buf[take:]
        with self._inflight_lock:
            self._inflight.extend(out)
        return out

    def ack(self, refs: List[SampleRef]) -> None:
        self.ack_ids([ref.sample_id for ref in refs])

    def ack_ids(self, sample_ids: List[str]) -> None:
        """Acknowledge the exact leased prefix after its durable train ack.

        The channel exposes a count rather than per-id acknowledgements, so
        accepting an arbitrary subset would make restart accounting ambiguous.
        Training consumes refs in lease order; enforce that invariant loudly.
        """
        if not sample_ids:
            return
        with self._inflight_lock:
            actual = [ref.sample_id for ref in self._inflight[: len(sample_ids)]]
            if actual != list(sample_ids):
                raise RuntimeError(
                    "stream acknowledgement is not the leased prefix: "
                    f"expected={actual}, got={list(sample_ids)}"
                )
            del self._inflight[: len(sample_ids)]
        self.channel.mark_consumed(len(sample_ids))

    def fail(self, refs: List[SampleRef], reason: str, retryable: bool) -> None:
        failed_ids = {ref.sample_id for ref in refs}
        with self._inflight_lock:
            self._inflight = [
                ref for ref in self._inflight if ref.sample_id not in failed_ids
            ]
        if retryable:
            self._buf[:0] = refs  # re-buffer at the front for the next get()
        else:
            self.channel.mark_consumed(len(refs))

    def depth(self) -> int:
        return len(self._buf)

    def in_flight_ids(self) -> List[str]:
        with self._inflight_lock:
            return [ref.sample_id for ref in self._inflight]


__all__ = ["RefPublishTransaction", "StreamingRefChannel", "StreamingRefQueue"]
