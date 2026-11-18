# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Stable, scalable identities for frozen model inputs used on resume.

Fresh runs record a versioned identity from weight-shard metadata and the
contents of small JSON descriptors.  They never stream every target shard just
to construct checkpoint metadata.  Rank 0 resolves that identity and broadcasts
it, so a distributed launch also avoids multiplying filesystem metadata work.

Resume requires the current identity format.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Callable, Mapping, Optional

MODEL_SOURCE_IDENTITY_FORMAT_FIELD = "source_identity_format"
MODEL_SOURCE_IDENTITY_FORMAT = "stat-v2"

_TRACKED_SUFFIXES = (
    ".bin",
    ".json",
    ".pt",
    ".pth",
    ".safetensors",
)


def _sha256_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _tracked_files(directory: str) -> list[tuple[str, str, os.stat_result]]:
    files = []
    for entry in os.scandir(directory):
        if not entry.is_file(follow_symlinks=True) or not entry.name.endswith(
            _TRACKED_SUFFIXES
        ):
            continue
        files.append((entry.name, entry.path, entry.stat(follow_symlinks=True)))
    return sorted(files, key=lambda item: item[0])


def _metadata_file_identity(
    path: str,
    name: str,
    stat: os.stat_result,
) -> tuple[Any, ...]:
    # Model/config index JSON is small and semantically important.  Hashing it
    # keeps exact descriptor provenance without reading multi-GB weight shards.
    if path.endswith(".json"):
        return ("sha256", name, stat.st_size, _sha256_file(path))
    # Binary shards use the rsync-style (name, size, mtime) quick check only.
    # ctime/device/inode are deliberately excluded: they change across
    # NFS/FUSE/overlay remounts and pod reschedules even when the bytes are
    # identical, so embedding them rejects exactly the resumes this contract
    # exists to serve. Content-level identity is still pinned by the sha256'd
    # JSON descriptors above (config + safetensors index cover shard layout).
    return (
        "stat",
        name,
        stat.st_size,
        stat.st_mtime_ns,
    )


def model_source_identity(path: Optional[str]) -> Any:
    """Return the current cheap identity for one local or referenced source."""

    if not path:
        return None
    expanded = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(expanded):
        return ("reference", path)
    if os.path.isfile(expanded):
        return (
            "file-stat-v2",
            expanded,
            _metadata_file_identity(expanded, "", os.stat(expanded)),
        )

    files = tuple(
        _metadata_file_identity(file_path, name, stat)
        for name, file_path, stat in _tracked_files(expanded)
    )
    return ("directory-stat-v2", expanded, files)


def _rank_zero_value(factory: Callable[[], Any], *, operation: str) -> Any:
    """Run filesystem work on rank 0 and broadcast either its value or error."""

    import torch.distributed as dist

    distributed = (
        dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    )
    if not distributed:
        return factory()

    payload = [None]
    if dist.get_rank() == 0:
        try:
            payload[0] = ("ok", factory())
        except Exception as exc:
            payload[0] = ("error", type(exc).__name__, str(exc))
    dist.broadcast_object_list(payload, src=0)
    result = payload[0]
    if not isinstance(result, tuple) or not result:
        raise RuntimeError(f"rank-0 {operation} returned a malformed broadcast")
    if result[0] == "error":
        raise RuntimeError(f"rank-0 {operation} failed with {result[1]}: {result[2]}")
    if result[0] != "ok" or len(result) != 2:
        raise RuntimeError(f"rank-0 {operation} returned a malformed broadcast")
    return result[1]


def _compute_model_resume_provenance(
    cfg: Any,
    draft_config: Any,
    target_config: Any,
    *,
    capture_layers: Optional[list[int]],
) -> dict[str, Any]:
    draft_source = cfg.draft if isinstance(cfg.draft, str) else None
    return {
        MODEL_SOURCE_IDENTITY_FORMAT_FIELD: MODEL_SOURCE_IDENTITY_FORMAT,
        "target_model": model_source_identity(cfg.model.target_model_path),
        "target_revision": (
            target_config.get("_commit_hash")
            if isinstance(target_config, dict)
            else getattr(target_config, "_commit_hash", None)
        ),
        "draft_config": model_source_identity(draft_source),
        "draft_revision": getattr(draft_config, "_commit_hash", None),
        "vocab_mapping": model_source_identity(cfg.model.vocab_mapping_path or None),
        "embedding_key": cfg.model.embedding_key,
        "lm_head_key": cfg.model.lm_head_key,
        "load_target_embedding": cfg.model.load_target_embedding,
        "capture_layers": (
            tuple(int(layer_id) for layer_id in capture_layers)
            if capture_layers is not None
            else None
        ),
        "input_modality": cfg.model.input_modality,
        "torch_dtype": cfg.model.torch_dtype,
    }


def model_resume_provenance(
    cfg: Any,
    draft_config: Any,
    target_config: Any,
    *,
    capture_layers: Optional[list[int]],
) -> dict[str, Any]:
    """Resolve one rank-agreed current provenance mapping."""

    return _rank_zero_value(
        lambda: _compute_model_resume_provenance(
            cfg,
            draft_config,
            target_config,
            capture_layers=capture_layers,
        ),
        operation="model provenance resolution",
    )


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, (Mapping, tuple, list)):
        raise ValueError(
            f"{name} model provenance is malformed: {type(value).__name__}"
        )
    try:
        return dict(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} model provenance is malformed") from exc


def model_provenance_for_resume_comparison(current: Any, saved: Any) -> tuple[Any, Any]:
    """Validate current-format provenance before comparing resume state."""
    for name, value in (("current", current), ("checkpoint", saved)):
        identity_format = _mapping(value, name=name).get(
            MODEL_SOURCE_IDENTITY_FORMAT_FIELD
        )
        if identity_format != MODEL_SOURCE_IDENTITY_FORMAT:
            raise ValueError(
                f"{name} model provenance uses unsupported identity format "
                f"{identity_format!r}; expected {MODEL_SOURCE_IDENTITY_FORMAT!r}"
            )
    return current, saved


__all__ = [
    "MODEL_SOURCE_IDENTITY_FORMAT",
    "MODEL_SOURCE_IDENTITY_FORMAT_FIELD",
    "model_provenance_for_resume_comparison",
    "model_resume_provenance",
    "model_source_identity",
]
