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

Checkpoints written before the metadata format contain full SHA-256 identities.
Those identities remain resumable: validation reconstructs the legacy value on
rank 0 only and caches it by a size/timestamp/file-identity signature. A
successful resume then writes the current format into its next checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Callable, Mapping, Optional

logger = logging.getLogger(__name__)

MODEL_SOURCE_IDENTITY_FORMAT_FIELD = "source_identity_format"
MODEL_SOURCE_IDENTITY_FORMAT = "stat-v1"

_LEGACY_HASH_CACHE_VERSION = 1
_LEGACY_HASH_CACHE_DIRECTORY = ".specforge-provenance-cache"
_MODEL_SOURCE_FIELDS = ("target_model", "draft_config", "vocab_mapping")
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
    return (
        "stat",
        name,
        stat.st_size,
        stat.st_mtime_ns,
        # ctime changes for both in-place writes and same-path replacements,
        # including copies that deliberately preserve mtime. This keeps the
        # cheap resume identity from accepting different same-size shard bytes;
        # device/inode also covers filesystems with coarse ctime resolution.
        stat.st_ctime_ns,
        stat.st_dev,
        stat.st_ino,
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
            "file-stat-v1",
            expanded,
            _metadata_file_identity(expanded, "", os.stat(expanded)),
        )

    files = tuple(
        _metadata_file_identity(file_path, name, stat)
        for name, file_path, stat in _tracked_files(expanded)
    )
    return ("directory-stat-v1", expanded, files)


def _legacy_model_source_identity_uncached(path: Optional[str]) -> Any:
    """Reproduce the pre-stat-v1 identity byte-for-byte."""

    def file_identity(file_path: str, name: str) -> tuple[str, int, str]:
        return (name, os.path.getsize(file_path), _sha256_file(file_path))

    if not path:
        return None
    expanded = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(expanded):
        return ("reference", path)
    if os.path.isfile(expanded):
        return ("file", expanded, file_identity(expanded, ""))

    files = [
        file_identity(file_path, name)
        for name, file_path, _stat in _tracked_files(expanded)
    ]
    return ("directory", expanded, tuple(sorted(files)))


def _source_stat_signature(path: Optional[str]) -> Any:
    if not path:
        return None
    expanded = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(expanded):
        return ("reference", path)

    def record(name: str, stat: os.stat_result) -> tuple[Any, ...]:
        return (
            name,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
            stat.st_dev,
            stat.st_ino,
        )

    if os.path.isfile(expanded):
        return ("file", expanded, record("", os.stat(expanded)))
    return (
        "directory",
        expanded,
        tuple(record(name, stat) for name, _path, stat in _tracked_files(expanded)),
    )


def _freeze_json(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, dict):
        return {key: _freeze_json(item) for key, item in value.items()}
    return value


def _legacy_cache_path(cache_root: str, signature: Any) -> str:
    encoded = json.dumps(signature, sort_keys=True, separators=(",", ":")).encode()
    key = hashlib.sha256(encoded).hexdigest()
    return os.path.join(cache_root, _LEGACY_HASH_CACHE_DIRECTORY, f"{key}.json")


def _read_legacy_cache(path: str, signature: Any) -> Any:
    try:
        with open(path, encoding="utf-8") as stream:
            payload = json.load(stream)
    except (FileNotFoundError, OSError, ValueError):
        return None
    if (
        payload.get("version") != _LEGACY_HASH_CACHE_VERSION
        or _freeze_json(payload.get("signature")) != signature
        or "identity" not in payload
    ):
        return None
    return _freeze_json(payload["identity"])


def _write_legacy_cache(path: str, signature: Any, identity: Any) -> None:
    temporary = f"{path}.{os.getpid()}.tmp"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(
                {
                    "version": _LEGACY_HASH_CACHE_VERSION,
                    "signature": signature,
                    "identity": identity,
                },
                stream,
                sort_keys=True,
            )
        os.replace(temporary, path)
    except OSError as exc:
        logger.debug("could not persist model-provenance cache %s: %s", path, exc)
        try:
            os.remove(temporary)
        except OSError:
            pass


def legacy_model_source_identity(path: Optional[str], *, cache_root: str) -> Any:
    """Return the old full-content identity, reusing a safe stat-keyed cache."""

    signature = _source_stat_signature(path)
    cache_path = _legacy_cache_path(cache_root, signature)
    cached = _read_legacy_cache(cache_path, signature)
    if cached is not None:
        return cached
    identity = _legacy_model_source_identity_uncached(path)
    _write_legacy_cache(cache_path, signature, identity)
    return identity


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
    draft_source = cfg.model.draft_model_config or getattr(
        draft_config,
        "_name_or_path",
        None,
    )
    return {
        MODEL_SOURCE_IDENTITY_FORMAT_FIELD: MODEL_SOURCE_IDENTITY_FORMAT,
        "target_model": model_source_identity(cfg.model.target_model_path),
        "target_revision": getattr(target_config, "_commit_hash", None),
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


def _legacy_source_from_current(identity: Any, *, cache_root: str) -> Any:
    if identity is None:
        return None
    if not isinstance(identity, (tuple, list)) or not identity:
        raise ValueError(f"current model source identity is malformed: {identity!r}")
    kind = identity[0]
    if kind == "reference":
        return tuple(identity)
    if kind not in {"file-stat-v1", "directory-stat-v1"} or len(identity) < 2:
        raise ValueError(f"unsupported current model source identity: {identity!r}")
    return legacy_model_source_identity(identity[1], cache_root=cache_root)


def model_provenance_for_resume_comparison(
    current: Any,
    saved: Any,
    *,
    cache_root: str,
) -> Any:
    """Adapt current provenance to an old checkpoint's comparison format.

    Current checkpoints compare their metadata identities directly.  A saved
    mapping with no format field is the legacy full-content contract; reconstruct
    that exact value lazily so existing checkpoints remain resumable.
    """

    current_mapping = _mapping(current, name="current")
    saved_mapping = _mapping(saved, name="saved")
    current_format = current_mapping.get(MODEL_SOURCE_IDENTITY_FORMAT_FIELD)
    saved_format = saved_mapping.get(MODEL_SOURCE_IDENTITY_FORMAT_FIELD)

    if current_format is None:
        return current
    if current_format != MODEL_SOURCE_IDENTITY_FORMAT:
        raise ValueError(
            f"unsupported current model provenance format {current_format!r}"
        )
    if saved_format == current_format:
        return current
    if saved_format is not None:
        raise ValueError(
            "checkpoint model provenance uses unsupported identity format "
            f"{saved_format!r}; expected {MODEL_SOURCE_IDENTITY_FORMAT!r}"
        )

    def legacy_value() -> tuple[tuple[str, Any], ...]:
        migrated = dict(current_mapping)
        migrated.pop(MODEL_SOURCE_IDENTITY_FORMAT_FIELD)
        for field in _MODEL_SOURCE_FIELDS:
            migrated[field] = _legacy_source_from_current(
                migrated.get(field),
                cache_root=cache_root,
            )
        return tuple((key, migrated[key]) for key in sorted(migrated))

    return _rank_zero_value(
        legacy_value,
        operation="legacy model provenance hashing",
    )


__all__ = [
    "MODEL_SOURCE_IDENTITY_FORMAT",
    "MODEL_SOURCE_IDENTITY_FORMAT_FIELD",
    "legacy_model_source_identity",
    "model_provenance_for_resume_comparison",
    "model_resume_provenance",
    "model_source_identity",
]
