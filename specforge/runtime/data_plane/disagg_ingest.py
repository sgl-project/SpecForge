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
offline EAGLE3 feature files (the ``.ckpt`` produced by
``scripts/prepare_hidden_states.py``) and ``put()``s each one into a
:class:`SharedDirFeatureStore` that lives on a filesystem both pools share. The
tensors land in the store (shared mount); the returned ``SampleRef``s are
tensor-free metadata, which is all the consumer (trainer) needs to fetch them.

The ref set is the only thing that must cross the producer→consumer boundary, so
it is serialized as a small JSON **manifest** — the disaggregated analogue of the
in-process ``OfflineManifestReader.read()`` list. The control-plane invariant
(metadata only, never tensors) is asserted before the manifest is written.
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import List, Optional

from specforge.runtime.contracts import (
    SCHEMA_VERSION,
    FeatureSpec,
    SampleRef,
    assert_no_tensors,
)
from specforge.runtime.data_plane.feature_store import FeatureStore, load_feature_file
from specforge.runtime.data_plane.offline_reader import (
    _OFFLINE_EAGLE3_KEYS,
    list_feature_files,
)


def ingest_offline_features(
    store: FeatureStore,
    hidden_states_path: str,
    *,
    run_id: str = "disagg",
    ttt_length: int = 7,
    max_len: int = 2048,
    limit: Optional[int] = None,
) -> List[SampleRef]:
    """Load offline ``.ckpt`` features and ``put()`` them into ``store``.

    Returns the ``disagg://`` ``SampleRef``s, in the same deterministic file
    order as ``OfflineManifestReader``. The raw EAGLE3 keys
    (``input_ids``/``loss_mask``/``hidden_state``/``aux_hidden_state``) are stored
    verbatim — the consumer applies the identical ``OfflineEagle3Dataset``
    aux->input / hidden->target swap at load time, so an offline run and a
    disaggregated run see byte-identical tensors.
    """
    paths = list_feature_files(hidden_states_path)
    if limit is not None:
        paths = paths[:limit]
    refs: List[SampleRef] = []
    for index, path in enumerate(paths):
        raw = load_feature_file(path)
        missing = [k for k in _OFFLINE_EAGLE3_KEYS if k not in raw]
        if missing:
            raise KeyError(f"{path} missing required offline feature keys {missing}")
        tensors = {k: raw[k] for k in _OFFLINE_EAGLE3_KEYS}
        input_ids = raw["input_ids"]
        ref = store.put(
            tensors,
            sample_id=f"{run_id}:{index:08d}",
            metadata={
                "run_id": run_id,
                "strategy": "eagle3",
                "format": "offline_eagle3",
                "target_repr": "hidden_state",
                "ttt_length": ttt_length,
                "max_len": max_len,
                "num_tokens": int(input_ids.numel()),
                "file_index": index,
            },
        )
        refs.append(ref)
    return refs


def _ref_to_dict(ref: SampleRef) -> dict:
    return dataclasses.asdict(ref)  # nested FeatureSpec -> dict; shape tuple -> list


def _ref_from_dict(d: dict) -> SampleRef:
    specs = {
        name: FeatureSpec(**{**spec, "shape": tuple(spec["shape"])})
        for name, spec in d["feature_specs"].items()
    }
    return SampleRef(**{**d, "feature_specs": specs})


def write_ref_manifest(refs: List[SampleRef], path: str) -> None:
    """Atomically write the tensor-free ref manifest the consumer reads.

    Asserts the no-tensor invariant first (a manifest is control-plane state),
    then publishes with a single rename so a reader never sees a partial file.
    """
    assert_no_tensors(refs)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "refs": [_ref_to_dict(r) for r in refs],
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
    return [_ref_from_dict(d) for d in payload["refs"]]


__all__ = ["ingest_offline_features", "write_ref_manifest", "read_ref_manifest"]
