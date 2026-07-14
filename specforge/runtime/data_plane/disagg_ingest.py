# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Disaggregated offline producer: ingest features into a shared FeatureStore.

This is the *producer* half of an offline disaggregated run. It reads SpecForge
offline feature files (``.ckpt`` / ``.ckpt.gz``) and ``put()``s each one into a
:class:`SharedDirFeatureStore` that lives on a filesystem both pools share. The
tensors land in the store (shared mount); the returned ``SampleRef``s are
tensor-free metadata, which is all the consumer (trainer) needs to fetch them.

The ref set is the only thing that must cross the producer→consumer boundary, so
it is serialized as a small JSON **manifest** — the disaggregated analogue of the
in-process ``OfflineManifestReader.read()`` list. The control-plane invariant
(metadata only, never tensors) is asserted before the manifest is written.
"""

from __future__ import annotations

import json
import os
from typing import Callable, List, Optional

from specforge.runtime.contracts import SCHEMA_VERSION, SampleRef, assert_no_tensors
from specforge.runtime.data_plane.feature_store import FeatureStore, load_feature_file
from specforge.runtime.data_plane.offline_reader import list_feature_files
from specforge.runtime.data_plane.ref_serialization import ref_from_dict, ref_to_dict


def ingest_offline_features(
    store: FeatureStore,
    hidden_states_path: str,
    *,
    strategy: str = "eagle3",
    run_id: str = "disagg",
    ttt_length: int = 7,
    max_len: int = 2048,
    limit: Optional[int] = None,
    on_ref: Optional[Callable[[SampleRef], None]] = None,
) -> List[SampleRef]:
    """Load offline ``.ckpt`` features and ``put()`` them into ``store``.

    The strategy registry owns the raw feature contract. EAGLE3 stores its
    final/auxiliary target states, while DFlash and Domino store their shared
    ``hidden_states`` capture.  Tensors are copied verbatim; the consumer uses
    the same strategy transform as a colocated run, so both topologies receive
    byte-identical normalized batches. ``on_ref`` runs immediately after every
    successful ``put()`` so a lifecycle owner can retain the refs needed to
    clean up a partially ingested attempt.
    """
    from specforge.training.strategies.registry import resolve_strategy

    spec = resolve_strategy(strategy)
    if spec.make_offline_reader is None:
        raise NotImplementedError(
            f"offline ingestion for strategy {strategy!r} is not wired; its "
            "StrategySpec needs make_offline_reader"
        )
    reader = spec.make_offline_reader(
        hidden_states_path,
        run_id=run_id,
        ttt_length=ttt_length,
        max_len=max_len,
    )
    feature_keys = tuple(reader.feature_keys)
    paths = list_feature_files(hidden_states_path)
    if limit is not None:
        paths = paths[:limit]
    refs: List[SampleRef] = []
    for index, path in enumerate(paths):
        raw = load_feature_file(path)
        missing = [key for key in feature_keys if key not in raw]
        if missing:
            raise KeyError(f"{path} missing required offline feature keys {missing}")
        tensors = {key: raw[key] for key in feature_keys}
        input_ids = raw["input_ids"]
        ref = store.put(
            tensors,
            sample_id=f"{run_id}:{index:08d}",
            metadata={
                "run_id": run_id,
                "strategy": strategy,
                "format": f"offline_{strategy}",
                "target_repr": reader.target_repr,
                "ttt_length": ttt_length,
                "max_len": max_len,
                "num_tokens": int(input_ids.numel()),
                "file_index": index,
            },
        )
        refs.append(ref)
        if on_ref is not None:
            on_ref(ref)
    return refs


def write_ref_manifest(refs: List[SampleRef], path: str) -> None:
    """Atomically write the tensor-free ref manifest the consumer reads.

    Asserts the no-tensor invariant first (a manifest is control-plane state),
    then publishes with a single rename so a reader never sees a partial file.
    """
    assert_no_tensors(refs)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "refs": [ref_to_dict(r) for r in refs],
    }
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


def read_ref_manifest(path: str) -> List[SampleRef]:
    """Read a ref manifest written by :func:`write_ref_manifest`."""
    with open(path) as f:
        payload = json.load(f)
    return [ref_from_dict(d) for d in payload["refs"]]


__all__ = ["ingest_offline_features", "write_ref_manifest", "read_ref_manifest"]
