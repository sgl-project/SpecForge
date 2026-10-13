# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Per-sample costs of offline feature sets for cost-aware scheduling.

Refs with ``num_tokens``/``valid_anchors`` metadata are used as-is; ``file://``
refs are scanned once and cached under ``<cache_dir>/cost_aware_index.json``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

from specforge.runtime.data_plane.window_plan import (
    Cost,
    OfflineWindowPlan,
    SchedulingPolicy,
    count_valid_anchors,
    ref_length,
    ref_valid_anchors,
)

logger = logging.getLogger(__name__)

INDEX_FILENAME = "cost_aware_index.json"
_INDEX_VERSION = 1
_SCAN_WORKERS = 8
_FILE_PREFIX = "file://"

#: ``(length, valid_anchors)`` per feature file; anchors are None without a mask.
FileEntry = Tuple[int, Optional[int]]


def _file_path(ref) -> Optional[str]:
    uri = getattr(ref, "feature_store_uri", "") or ""
    return uri[len(_FILE_PREFIX) :] if uri.startswith(_FILE_PREFIX) else None


def index_fingerprint(paths: Sequence[str], max_len: int) -> str:
    """Cache key: file list, sizes, mtimes, ``max_len`` (delete the cache to force a rescan)."""
    digest = hashlib.sha1()
    digest.update(f"version={_INDEX_VERSION};max_len={int(max_len)}\n".encode())
    for path in paths:
        stat = os.stat(path)
        digest.update(f"{path}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def costs_digest(costs: Sequence[Cost]) -> str:
    """Digest of the cost vector in ref order; recorded for resume checks."""
    digest = hashlib.sha1()
    digest.update(f"n={len(costs)}\n".encode())
    for cost in costs:
        digest.update((",".join(repr(float(value)) for value in cost) + "\n").encode())
    return digest.hexdigest()


def scan_feature_file(path: str, max_len: int) -> FileEntry:
    """Length and valid-anchor count of one offline feature file."""
    from specforge.runtime.data_plane.feature_store import load_feature_file

    raw = load_feature_file(path)
    input_ids = raw["input_ids"]
    length = min(int(input_ids.numel()), int(max_len))
    loss_mask = raw.get("loss_mask")
    if loss_mask is None or not hasattr(loss_mask, "reshape"):
        return length, None
    return length, count_valid_anchors(loss_mask.reshape(-1)[:length])


def scan_feature_files(
    paths: Sequence[str], max_len: int, *, workers: int = _SCAN_WORKERS
) -> Dict[str, FileEntry]:
    """Scan ``paths`` in parallel threads (I/O bound: mmap + two small tensors)."""
    entries: Dict[str, FileEntry] = {}
    total = len(paths)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        scanned = pool.map(lambda path: scan_feature_file(path, max_len), paths)
        for count, (path, entry) in enumerate(zip(paths, scanned), 1):
            entries[path] = entry
            if count % 1000 == 0 or count == total:
                logger.info(
                    "cost-aware scheduling: indexed %d/%d feature files",
                    count,
                    total,
                )
    return entries


def load_or_build_index(
    paths: Sequence[str], max_len: int, cache_dir: Optional[str]
) -> Tuple[Dict[str, FileEntry], str]:
    """Return ``{path: (length, valid_anchors)}`` and the index fingerprint."""
    fingerprint = index_fingerprint(paths, max_len)
    cache_path = os.path.join(cache_dir, INDEX_FILENAME) if cache_dir else None
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path) as handle:
                payload = json.load(handle)
            if payload.get("fingerprint") == fingerprint:
                samples = payload["samples"]
                return (
                    {
                        path: (
                            int(samples[path][0]),
                            None if samples[path][1] is None else int(samples[path][1]),
                        )
                        for path in paths
                    },
                    fingerprint,
                )
            logger.info(
                "cost-aware scheduling: %s belongs to another feature set; rescanning",
                cache_path,
            )
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            logger.warning(
                "cost-aware scheduling: ignoring unreadable %s (%s)", cache_path, exc
            )
    logger.info(
        "cost-aware scheduling: indexing %d feature files (one-time scan)",
        len(paths),
    )
    entries = scan_feature_files(paths, max_len)
    if cache_path:
        payload = {
            "version": _INDEX_VERSION,
            "fingerprint": fingerprint,
            "max_len": int(max_len),
            "samples": {path: list(entries[path]) for path in paths},
        }
        try:
            os.makedirs(cache_dir, exist_ok=True)
            tmp = f"{cache_path}.{os.getpid()}.tmp"
            with open(tmp, "w") as handle:
                json.dump(payload, handle)
            os.replace(
                tmp, cache_path
            )  # atomic: concurrent ranks write the same content
        except OSError as exc:
            logger.warning(
                "cost-aware scheduling: could not cache the index at %s (%s)",
                cache_path,
                exc,
            )
    return entries, fingerprint


def offline_sample_costs(
    refs: Sequence,
    policy: SchedulingPolicy,
    *,
    max_len: int,
    cache_dir: Optional[str] = None,
) -> Tuple[List[Cost], SchedulingPolicy, Optional[str]]:
    """Costs aligned with ``refs`` (lengths clamped to ``max_len``), the policy applied
    (anchor keys fall back to ``length`` without loss masks) and the index fingerprint.
    """
    lengths = [ref_length(ref) for ref in refs]
    anchors = [ref_valid_anchors(ref) for ref in refs]
    files = [_file_path(ref) for ref in refs]
    missing = [
        index
        for index in range(len(refs))
        if lengths[index] is None
        or (policy.uses_anchors and anchors[index] is None and files[index])
    ]
    fingerprint = None
    if missing:
        paths = []
        for index in missing:
            if files[index] is None:
                raise ValueError(
                    "cost-aware scheduling: sample "
                    f"{getattr(refs[index], 'sample_id', index)!r} records neither "
                    "num_tokens metadata nor a file:// feature path"
                )
            paths.append(files[index])
        entries, fingerprint = load_or_build_index(paths, max_len, cache_dir)
        for index, path in zip(missing, paths):
            length, valid = entries[path]
            if lengths[index] is None:
                lengths[index] = length
            if anchors[index] is None:
                anchors[index] = valid
    clamped = [min(int(length), int(max_len)) for length in lengths]
    applied = policy
    if policy.uses_anchors and any(value is None for value in anchors):
        unknown = sum(1 for value in anchors if value is None)
        logger.warning(
            "cost-aware scheduling: %d/%d samples carry no valid-anchor count "
            "(no loss mask); scheduling by length instead of %r",
            unknown,
            len(refs),
            policy.key,
        )
        applied = policy.with_key("length")
    costs = [
        applied.sample_cost(length, valid) for length, valid in zip(clamped, anchors)
    ]
    if any(cost is None for cost in costs):
        raise ValueError("cost-aware scheduling: a sample has no usable length")
    return costs, applied, fingerprint


def offline_window_plan(
    refs: Sequence,
    policy: Optional[SchedulingPolicy],
    *,
    batch_size: int,
    accumulation_steps: int,
    max_len: int,
    cache_dir: Optional[str] = None,
) -> Tuple[Optional[OfflineWindowPlan], Optional[dict]]:
    """Build the sampler plan and its checkpoint contract (both None if disabled)."""
    if policy is None:
        return None, None
    costs, applied, _ = offline_sample_costs(
        refs, policy, max_len=max_len, cache_dir=cache_dir
    )
    window_rounds = int(applied.window_rounds) or int(accumulation_steps)
    contract = {
        **applied.contract(),
        "window_rounds": window_rounds,
        "max_len": int(max_len),
        "costs_digest": costs_digest(costs),
    }
    logger.info(
        "cost-aware scheduling: key=%s window=%d rounds over %d samples",
        applied.key,
        window_rounds,
        len(refs),
    )
    return OfflineWindowPlan(costs, int(batch_size), window_rounds), contract


__all__ = [
    "INDEX_FILENAME",
    "costs_digest",
    "index_fingerprint",
    "load_or_build_index",
    "offline_sample_costs",
    "offline_window_plan",
    "scan_feature_file",
    "scan_feature_files",
]
