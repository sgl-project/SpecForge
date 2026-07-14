# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""FeatureDataLoader: ``SampleRef`` + ``FeatureStore`` -> ``TrainBatch``.

Leases refs (consume-once queue or re-iterable ref list), applies the injected
per-sample transform and collate, and yields ``TrainBatch``es — no model
knowledge. clone-on-fetch (default) clones tensors out of the store and releases
the handle immediately, so prefetch can never race a release.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, Iterator, List, Optional

import torch

from specforge.runtime.contracts import SampleRef, TrainBatch
from specforge.runtime.data_plane.feature_store import FeatureStore
from specforge.runtime.data_plane.sample_ref_queue import SampleRefQueue

PerSampleTransform = Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
CollateFn = Callable[[List[Dict[str, torch.Tensor]]], Dict[str, Any]]


def _default_collate(features: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
    """Trivial stack collate (used only when no collate_fn is injected)."""
    keys = features[0].keys()
    return {k: torch.stack([f[k] for f in features], dim=0) for k in keys}


class FeatureDataLoader:
    def __init__(
        self,
        store: FeatureStore,
        queue: Optional[SampleRefQueue] = None,
        *,
        refs: Optional[List[SampleRef]] = None,
        batch_size: int = 1,
        collate_fn: Optional[CollateFn] = None,
        per_sample_transform: Optional[PerSampleTransform] = None,
        device: "torch.device | str" = "cpu",
        clone_on_fetch: bool = True,
        drop_last: bool = True,
        strategy: str = "eagle3",
        ack: bool = True,
        gc_interval_s: Optional[float] = 15.0,
        num_workers: int = 0,
    ) -> None:
        if (queue is None) == (refs is None):
            raise ValueError(
                "provide exactly one of `queue` (stream) or `refs` (re-iterable)"
            )
        self.store = store
        self.queue = queue
        self._refs = list(refs) if refs is not None else None
        self.batch_size = batch_size
        self.collate_fn = collate_fn or _default_collate
        self.per_sample_transform = per_sample_transform
        self.device = device
        # CLONE_ON_FETCH=0 skips the defensive clone; safe for the mooncake
        # zero-copy path, whose get() already allocates a fresh tensor.
        if os.environ.get("CLONE_ON_FETCH", "1") == "0":
            clone_on_fetch = False
        self.clone_on_fetch = clone_on_fetch
        self.drop_last = drop_last
        self.strategy = strategy
        self.ack = ack
        if num_workers < 0:
            raise ValueError("num_workers must be >= 0")
        self.num_workers = int(num_workers)
        self._seek_batches = 0
        # Remote stores defer a physical free while the get() read-lease is live
        # (Mooncake remove -> -706): release() parks it and gc() must retry.
        # Pump gc on a cadence LONGER than the lease TTL — the store gives up
        # after max_release_attempts, so rapid retries would leak the object.
        self.gc_interval_s = gc_interval_s if hasattr(store, "gc") else None
        self._last_gc = time.monotonic()

    def _maybe_gc(self) -> None:
        if self.gc_interval_s is None:
            return
        now = time.monotonic()
        if now - self._last_gc >= self.gc_interval_s:
            self._last_gc = now
            self.store.gc()

    def _validate_refs(self, refs: List[SampleRef]) -> None:
        strategies = {ref.strategy for ref in refs}
        if strategies != {self.strategy}:
            raise ValueError(
                f"loader strategy={self.strategy!r} received refs with "
                f"strategies={sorted(strategies)}"
            )

        schema_versions = {ref.schema_version for ref in refs}
        if len(schema_versions) != 1:
            raise ValueError(
                f"mixed schema versions in batch: {sorted(schema_versions)}"
            )

        target_reprs = {ref.metadata.get("target_repr") for ref in refs}
        if len(target_reprs) != 1:
            raise ValueError(
                f"mixed target_repr values in batch: {sorted(map(repr, target_reprs))}"
            )

        spec_sets = [set(ref.feature_specs) for ref in refs if ref.feature_specs]
        if spec_sets and any(spec_set != spec_sets[0] for spec_set in spec_sets[1:]):
            raise ValueError(f"mixed feature spec names in batch: {spec_sets}")

        if not spec_sets:
            return
        first_specs = next(ref.feature_specs for ref in refs if ref.feature_specs)
        for ref in refs:
            if not ref.feature_specs:
                continue
            for name, spec in ref.feature_specs.items():
                expected = first_specs[name]
                if spec.dtype != expected.dtype or len(spec.shape) != len(
                    expected.shape
                ):
                    raise ValueError(
                        f"incompatible feature spec for sample {ref.sample_id}, "
                        f"feature {name!r}: {spec} vs {expected}"
                    )

    # PROFILE_LOADER=N -> every N batches print one [loader rK] line splitting
    # queue-wait / store-get / clone / release / gc time per batch.
    _prof_loader = int(os.environ.get("PROFILE_LOADER", "0"))

    def _lp(self) -> dict:
        s = getattr(self, "_lp_state", None)
        if s is None:
            s = {
                "b": 0,
                "wait": 0.0,
                "get": 0.0,
                "clone": 0.0,
                "rel": 0.0,
                "gc": 0.0,
                "bytes": 0,
                "t0": time.monotonic(),
            }
            self._lp_state = s
        return s

    def _materialize(self, ref: SampleRef) -> Dict[str, torch.Tensor]:
        _prof = self._prof_loader
        _t = time.monotonic()
        tensors, handle = self.store.get(ref, device=self.device)
        if _prof:
            s = self._lp()
            s["get"] += time.monotonic() - _t
            s["bytes"] += sum(t.numel() * t.element_size() for t in tensors.values())
            _t = time.monotonic()
        if self.clone_on_fetch:
            tensors = {k: v.clone() for k, v in tensors.items()}
        if _prof:
            self._lp()["clone"] += time.monotonic() - _t
            _t = time.monotonic()
        self.store.release(handle, reason="loaded")
        if _prof:
            self._lp()["rel"] += time.monotonic() - _t
            _t = time.monotonic()
        self._maybe_gc()
        if _prof:
            self._lp()["gc"] += time.monotonic() - _t
        if self.per_sample_transform is not None:
            tensors = self.per_sample_transform(tensors)
        return tensors

    def _make_batch(self, refs: List[SampleRef]) -> TrainBatch:
        self._validate_refs(refs)
        per_sample = [self._materialize(r) for r in refs]
        batch_tensors = self.collate_fn(per_sample)
        non_tensors = [
            name
            for name, value in batch_tensors.items()
            if not isinstance(value, torch.Tensor)
        ]
        if non_tensors:
            raise TypeError(f"collate_fn returned non-tensors for {non_tensors}")
        return TrainBatch(
            sample_ids=[r.sample_id for r in refs],
            strategy=self.strategy,
            tensors=batch_tensors,
            metadata={
                "target_repr": refs[0].metadata.get("target_repr"),
                "ttt_length": refs[0].metadata.get("ttt_length"),
            },
        )

    def __iter__(self) -> Iterator[TrainBatch]:
        if self._refs is not None:
            yield from self._iter_refs()
        else:
            yield from self._iter_queue()

    def _iter_refs(self) -> Iterator[TrainBatch]:
        # Acking (the durable marker) is the trainer's job here, not the loader's.
        skip, self._seek_batches = self._seek_batches, 0
        chunks = []
        for start in range(skip * self.batch_size, len(self._refs), self.batch_size):
            chunk = self._refs[start : start + self.batch_size]
            if self.drop_last and len(chunk) < self.batch_size:
                break
            chunks.append(chunk)
        if self.num_workers > 0:
            yield from self._iter_refs_prefetch(chunks)
            return
        for chunk in chunks:
            yield self._make_batch(chunk)

    def _iter_refs_prefetch(
        self, chunks: List[List[SampleRef]]
    ) -> Iterator[TrainBatch]:
        """Materialize fixed offline batches concurrently while preserving order."""
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor

        chunk_iter = iter(chunks)
        pending = deque()
        with ThreadPoolExecutor(
            max_workers=self.num_workers,
            thread_name_prefix="feature-loader",
        ) as executor:
            for _ in range(self.num_workers):
                try:
                    chunk = next(chunk_iter)
                except StopIteration:
                    break
                pending.append(executor.submit(self._make_batch, chunk))
            while pending:
                yield pending.popleft().result()
                try:
                    chunk = next(chunk_iter)
                except StopIteration:
                    continue
                pending.append(executor.submit(self._make_batch, chunk))

    def seek(self, num_batches: int) -> None:
        """Skip the first ``num_batches`` of the NEXT iteration (refs mode; one-shot).

        Raises on a queue stream (which has no resumable position) and when the
        skip exceeds the available batches (a silently empty epoch is banned).
        """
        if self._refs is None:
            raise ValueError(
                "seek() applies to refs mode only; a queue stream is consume-once"
            )
        skip = max(0, int(num_batches))
        if self.drop_last:
            available = len(self._refs) // self.batch_size
        else:
            available = -(-len(self._refs) // self.batch_size)
        if skip > available:
            raise ValueError(
                f"seek({num_batches}) skips past the end of the data: only "
                f"{available} batches available ({len(self._refs)} refs at "
                f"batch_size={self.batch_size}, drop_last={self.drop_last}); "
                f"the resume position does not fit this dataset"
            )
        self._seek_batches = skip

    def _iter_queue(self) -> Iterator[TrainBatch]:
        # LOADER_PREFETCH=N (>0) materializes up to N batches ahead on a
        # background thread so the training step never pays fetch latency
        # inline. Ack still happens on the consuming thread AFTER the trainer
        # has taken the batch (same in-flight semantics as the sync path).
        depth = self.num_workers or int(os.environ.get("LOADER_PREFETCH", "0"))
        if not getattr(self.queue, "loader_prefetch_safe", True):
            # A pull-through local rollout runs target inference inside get().
            # Keep that CUDA work on the training thread so early-stop and
            # producer failures share one deterministic lifecycle.
            depth = 0
        if depth > 0:
            yield from self._iter_queue_prefetch(depth)
            return
        _prof = self._prof_loader
        while True:
            _t = time.monotonic()
            refs = self.queue.get(self.batch_size, timeout_s=0.0)
            if _prof:
                s = self._lp()
                s["wait"] += time.monotonic() - _t
            if not refs:
                return
            if self.drop_last and len(refs) < self.batch_size:
                reason = (
                    "online stream ended with an incomplete batch: "
                    f"received {len(refs)} refs but batch_size={self.batch_size}"
                )
                self.queue.fail(refs, reason=reason, retryable=False)
                raise RuntimeError(reason)
            try:
                batch = self._make_batch(refs)
            except Exception as exc:
                self.queue.fail(refs, reason=f"materialize:{exc}", retryable=False)
                raise
            if _prof:
                s = self._lp()
                s["b"] += 1
                if s["b"] >= _prof:
                    win = time.monotonic() - s["t0"]
                    rank = os.environ.get("RANK", "?")
                    print(
                        f"[loader r{rank}] b={s['b']} win_s={win:.2f} "
                        f"wait_ms={1000*s['wait']/s['b']:.1f} "
                        f"get_ms={1000*s['get']/s['b']:.1f} "
                        f"clone_ms={1000*s['clone']/s['b']:.1f} "
                        f"rel_ms={1000*s['rel']/s['b']:.1f} "
                        f"gc_ms={1000*s['gc']/s['b']:.1f} "
                        f"MB={s['bytes']/s['b']/1e6:.1f} (avg/batch)",
                        flush=True,
                    )
                    self._lp_state = None
            yield batch
            if self.ack:
                self.queue.ack(refs)

    def _iter_queue_prefetch(self, depth: int) -> Iterator[TrainBatch]:
        import queue as _queue
        import threading

        _prof = self._prof_loader
        buf: "_queue.Queue" = _queue.Queue(maxsize=depth)
        _EOS = object()

        def _worker() -> None:
            try:
                while True:
                    refs = self.queue.get(self.batch_size, timeout_s=0.0)
                    if not refs:
                        buf.put(_EOS)
                        return
                    if self.drop_last and len(refs) < self.batch_size:
                        reason = (
                            "online stream ended with an incomplete batch: "
                            f"received {len(refs)} refs but "
                            f"batch_size={self.batch_size}"
                        )
                        self.queue.fail(refs, reason=reason, retryable=False)
                        buf.put(RuntimeError(reason))
                        return
                    try:
                        batch = self._make_batch(refs)
                    except Exception as exc:
                        self.queue.fail(
                            refs, reason=f"materialize:{exc}", retryable=False
                        )
                        buf.put(exc)
                        return
                    buf.put((batch, refs))
            except BaseException as exc:  # loud failure, never a silent hang
                buf.put(exc)

        threading.Thread(target=_worker, name="loader-prefetch", daemon=True).start()
        while True:
            _t = time.monotonic()
            item = buf.get()
            if _prof:
                s = self._lp()
                s["wait"] += time.monotonic() - _t
            if item is _EOS:
                return
            if isinstance(item, BaseException):
                raise item
            batch, refs = item
            if _prof:
                s = self._lp()
                s["b"] += 1
                if s["b"] >= _prof:
                    win = time.monotonic() - s["t0"]
                    rank = os.environ.get("RANK", "?")
                    print(
                        f"[loader r{rank}] b={s['b']} win_s={win:.2f} "
                        f"wait_ms={1000*s['wait']/s['b']:.1f} "
                        f"get_ms={1000*s['get']/s['b']:.1f} "
                        f"clone_ms={1000*s['clone']/s['b']:.1f} "
                        f"rel_ms={1000*s['rel']/s['b']:.1f} "
                        f"gc_ms={1000*s['gc']/s['b']:.1f} "
                        f"MB={s['bytes']/s['b']/1e6:.1f} (avg/batch, prefetch)",
                        flush=True,
                    )
                    self._lp_state = None
            yield batch
            if self.ack:
                self.queue.ack(refs)

    def set_epoch(self, epoch: int) -> None:
        # hook for per-epoch shuffling of the offline ref set (no-op for now)
        self._epoch = epoch

    def close(self) -> None:
        pass


__all__ = ["FeatureDataLoader", "PerSampleTransform", "CollateFn"]
