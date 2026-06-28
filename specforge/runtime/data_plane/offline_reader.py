# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""OfflineManifestReader: turn precomputed feature files into ``SampleRef``s.

The reader walks a directory of SpecForge offline feature files (the ``.ckpt`` /
``.ckpt.gz`` produced by ``scripts/prepare_hidden_states.py``) and emits one
metadata-only ``SampleRef`` per file, referencing the file in place via a
``file://`` URI (read-only existing-file mode — no tensor copy, no tensor
through the controller). The actual per-sample normalization (the
``OfflineEagle3Dataset.process_data`` swap: stored ``aux_hidden_state`` becomes
the draft input ``hidden_state`` and stored ``hidden_state`` becomes the
``target``) is the FeatureDataLoader's job, keeping this reader independent of
the model code.
"""

from __future__ import annotations

import os
from typing import Dict, Iterator, List, Optional, Tuple

from specforge.runtime.contracts import SCHEMA_VERSION, FeatureSpec, SampleRef
from specforge.runtime.data_plane.feature_store import (
    load_feature_file,
    spec_from_tensor,
)

_FEATURE_SUFFIXES = (".ckpt", ".ckpt.gz")
# Raw keys present in a SpecForge offline EAGLE3 feature file.
_OFFLINE_EAGLE3_KEYS = ("input_ids", "loss_mask", "hidden_state", "aux_hidden_state")


def _inspect_feature_file(
    path: str, feature_keys: Tuple[str, ...]
) -> Tuple[Dict[str, FeatureSpec], int, int]:
    raw = load_feature_file(path)
    missing = [key for key in feature_keys if key not in raw]
    if missing:
        raise KeyError(f"{path} missing required offline feature keys {missing}")

    specs: Dict[str, FeatureSpec] = {}
    estimated_bytes = 0
    for key in feature_keys:
        value = raw[key]
        if not hasattr(value, "shape") or not hasattr(value, "dtype"):
            raise TypeError(f"{path} feature {key!r} is not a tensor: {type(value)!r}")
        specs[key] = spec_from_tensor(key, value)
        estimated_bytes += int(value.numel() * value.element_size())

    input_ids = raw.get("input_ids")
    num_tokens = int(input_ids.numel()) if input_ids is not None else 0
    return specs, num_tokens, estimated_bytes


def list_feature_files(path: str) -> List[str]:
    """Deterministically (sorted) list feature files under ``path``."""
    if os.path.isfile(path):
        return [os.path.abspath(path)]
    files: List[str] = []
    for root, _dirs, names in os.walk(path):
        for name in names:
            if name.endswith(_FEATURE_SUFFIXES):
                files.append(os.path.abspath(os.path.join(root, name)))
    files.sort()  # deterministic, stable cross-rank ordering
    return files


class OfflineManifestReader:
    """Reads a directory of offline feature files into ``SampleRef`` records."""

    def __init__(
        self,
        hidden_states_path: str,
        *,
        run_id: str = "offline",
        strategy: str = "eagle3",
        target_model_version: str = "unknown",
        tokenizer_version: str = "unknown",
        feature_keys: tuple = _OFFLINE_EAGLE3_KEYS,
        ttt_length: int = 7,
        max_len: int = 2048,
        target_repr: str = "hidden_state",
        validate_files: bool = True,
    ) -> None:
        self.hidden_states_path = hidden_states_path
        self.run_id = run_id
        self.strategy = strategy
        self.target_model_version = target_model_version
        self.tokenizer_version = tokenizer_version
        self.feature_keys = tuple(feature_keys)
        self.ttt_length = ttt_length
        self.max_len = max_len
        self.target_repr = target_repr
        self.validate_files = validate_files

    def _ref_for(self, index: int, path: str) -> SampleRef:
        sample_id = f"{self.run_id}:{index:08d}"
        specs: Dict[str, FeatureSpec] = {}
        num_tokens = 0
        estimated_bytes = 0
        if self.validate_files:
            specs, num_tokens, estimated_bytes = _inspect_feature_file(
                path, self.feature_keys
            )
        return SampleRef(
            sample_id=sample_id,
            run_id=self.run_id,
            source_task_id=None,
            feature_store_uri=f"file://{path}",
            feature_keys={k: k for k in self.feature_keys},
            feature_specs=specs,
            strategy=self.strategy,
            schema_version=SCHEMA_VERSION,
            target_model_version=self.target_model_version,
            tokenizer_version=self.tokenizer_version,
            num_tokens=num_tokens,
            estimated_bytes=estimated_bytes,
            metadata={
                "format": "offline_eagle3",
                "target_repr": self.target_repr,
                "schema_version": SCHEMA_VERSION,
                "ttt_length": self.ttt_length,
                "max_len": self.max_len,
                "file_index": index,
            },
        )

    def __iter__(self) -> Iterator[SampleRef]:
        for index, path in enumerate(list_feature_files(self.hidden_states_path)):
            yield self._ref_for(index, path)

    def read(self, limit: Optional[int] = None) -> List[SampleRef]:
        refs: List[SampleRef] = []
        for i, ref in enumerate(self):
            if limit is not None and i >= limit:
                break
            refs.append(ref)
        return refs


__all__ = ["OfflineManifestReader", "list_feature_files"]
