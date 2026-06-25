# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""FeatureDataLoader: ``SampleRef`` + ``FeatureStore`` -> ``TrainBatch``.

The loader is the one place the online/offline difference is erased: it leases
refs from a queue, fetches their tensors from the store, normalizes each sample
(an injectable ``per_sample_transform``), collates a batch (an injectable
``collate_fn``), and emits a ``TrainBatch``. Because both transform and collate
are injected, the loader carries no model knowledge and is unit-testable on CPU;
the offline-EAGLE3 run injects the existing ``OfflineEagle3Dataset.process_data``
and ``DataCollatorWithPadding`` so the result is bit-identical to today's path.

clone-on-fetch is the default: the loader clones tensors out of the store and
releases the store handle immediately, so prefetch can never race a release.
"""

from __future__ import annotations

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
    ) -> None:
        # Two iteration modes, reflecting that online and offline differ in
        # *iteration* even though they converge at SampleRef:
        #   - queue: a consume-once stream (online rollout produces over time)
        #   - refs:  a fixed set, re-iterable across epochs (offline manifest)
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
        self.clone_on_fetch = clone_on_fetch
        self.drop_last = drop_last
        self.strategy = strategy
        self.ack = ack

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

    def _materialize(self, ref: SampleRef) -> Dict[str, torch.Tensor]:
        tensors, handle = self.store.get(ref, device=self.device)
        if self.clone_on_fetch:
            tensors = {k: v.clone() for k, v in tensors.items()}
        self.store.release(handle, reason="loaded")
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
        # Offline: a fixed ref set, re-iterable every epoch. Acking (durable
        # marker) is the trainer's job via its ack callback, not the loader's.
        for start in range(0, len(self._refs), self.batch_size):
            chunk = self._refs[start : start + self.batch_size]
            if self.drop_last and len(chunk) < self.batch_size:
                break
            yield self._make_batch(chunk)

    def _iter_queue(self) -> Iterator[TrainBatch]:
        while True:
            refs = self.queue.get(self.batch_size, timeout_s=0.0)
            if not refs:
                return
            if self.drop_last and len(refs) < self.batch_size:
                # Incomplete trailing batch: fail-retryable so it is not lost,
                # then stop (mirrors DataLoader(drop_last=True) per epoch pass).
                self.queue.fail(refs, reason="drop_last", retryable=True)
                return
            try:
                batch = self._make_batch(refs)
            except Exception as exc:
                self.queue.fail(refs, reason=f"materialize:{exc}", retryable=False)
                raise
            yield batch
            if self.ack:
                self.queue.ack(refs)

    def set_epoch(self, epoch: int) -> None:
        # hook for per-epoch shuffling of the offline ref set (no-op for now)
        self._epoch = epoch

    def close(self) -> None:
        pass


__all__ = ["FeatureDataLoader", "PerSampleTransform", "CollateFn"]
