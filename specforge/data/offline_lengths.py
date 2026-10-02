"""Opt-in, persistent token-length metadata for offline batching.

Legacy feature directories do not contain lengths. Only rank zero indexes
missing lengths, once per file revision; training still loads tensors lazily.
The cache lives in the run output, never in the read-only feature directory.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import replace

from specforge.data.length_bucketing import sample_length

logger = logging.getLogger(__name__)
_INDEX_HEARTBEAT_SECONDS = 30.0


def _index_lengths(refs, cache_dir, progress=None):
    from specforge.runtime.data_plane.feature_store import load_feature_file

    cache_path = os.path.join(cache_dir, "offline-token-lengths.json")
    try:
        with open(cache_path) as stream:
            saved = json.load(stream)
        entries = saved.get("entries", {}) if saved.get("version") == 1 else {}
        if not isinstance(entries, dict):
            entries = {}
    except (OSError, ValueError, TypeError, AttributeError):
        entries = {}
    lengths, updated = [], {}
    missing = sum(ref.num_tokens <= 0 for ref in refs)
    if missing:
        logger.info(
            "Resolving %d offline token lengths; uncached .ckpt.gz files require "
            "one decompression pass. Cache: %s",
            missing,
            cache_path,
        )
    inspected = 0
    for ref in refs:
        if progress is not None:
            progress()
        known_length = sample_length(ref)
        if known_length is not None:
            lengths.append(known_length)
            continue
        if not ref.feature_store_uri.startswith("file://"):
            raise ValueError(
                "offline length bucketing requires token-length metadata for "
                f"non-file sample {ref.sample_id!r}"
            )
        path = os.path.abspath(ref.feature_store_uri[len("file://") :])
        stat = os.stat(path)
        signature = [stat.st_size, stat.st_mtime_ns, stat.st_ino]
        key = ref.feature_keys.get("input_ids", "input_ids").split("/")[-1]
        entry = entries.get(path)
        if (
            isinstance(entry, dict)
            and entry.get("signature") == signature
            and entry.get("input_ids_key") == key
            and type(entry.get("length")) is int
            and entry["length"] > 0
        ):
            length = entry["length"]
        else:
            raw = load_feature_file(path)
            ids = raw[key]
            if ids.ndim not in (1, 2) or (ids.ndim == 2 and ids.shape[0] != 1):
                raise ValueError(f"{path}: expected one input_ids sequence")
            length = int(ids.shape[-1])
            if length <= 0:
                raise ValueError(f"{path}: input_ids must not be empty")
            del ids, raw
            inspected += 1
            if inspected % 100 == 0:
                logger.info("Indexed %d offline feature files", inspected)
        lengths.append(length)
        updated[path] = {
            "signature": signature,
            "input_ids_key": key,
            "length": length,
        }

    if updated:
        os.makedirs(cache_dir, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=cache_dir, prefix=".offline-lengths-", delete=False
            ) as stream:
                temporary = stream.name
                json.dump({"version": 1, "entries": updated}, stream)
            os.replace(temporary, cache_path)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)
    logger.info("Offline length metadata ready (%d newly inspected files)", inspected)
    return lengths


def ensure_offline_lengths(refs, *, cache_dir):
    """Return refs with token lengths, broadcasting rank-zero results/errors."""
    import torch.distributed as dist

    refs = list(refs)
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    if distributed:
        # Every rank must agree before any rank can enter trainer collectives.
        # Equal counts alone cannot detect a differently ordered manifest.
        digest = hashlib.sha256()
        for ref in refs:
            digest.update(
                json.dumps(
                    [
                        ref.sample_id,
                        ref.feature_store_uri,
                        sample_length(ref),
                        ref.feature_keys.get("input_ids", "input_ids"),
                    ],
                    separators=(",", ":"),
                ).encode()
            )
        identity = (len(refs), digest.hexdigest())
        identities = [None] * dist.get_world_size()
        dist.all_gather_object(identities, identity)
        if any(item != identity for item in identities):
            raise ValueError("offline feature lists differ between training ranks")
    payload = [None]
    if rank == 0:
        last_progress = time.monotonic()

        def heartbeat():
            nonlocal last_progress
            if (
                distributed
                and time.monotonic() - last_progress >= _INDEX_HEARTBEAT_SECONDS
            ):
                # A first pass over compressed datasets can exceed the process
                # group's timeout. Peers consume these small messages while
                # rank zero keeps indexing instead of timing out in one wait.
                dist.broadcast_object_list([{"progress": True}], src=0)
                last_progress = time.monotonic()

        try:
            payload[0] = {"lengths": _index_lengths(refs, cache_dir, heartbeat)}
        except Exception as exc:
            if not distributed:
                raise
            payload[0] = {"error": f"{type(exc).__name__}: {exc}"}
    if distributed:
        while True:
            dist.broadcast_object_list(payload, src=0)
            if "progress" not in payload[0]:
                break
    result = payload[0]
    if "error" in result:
        raise RuntimeError(f"offline length indexing failed: {result['error']}")
    lengths = result["lengths"]
    return [replace(ref, num_tokens=length) for ref, length in zip(refs, lengths)]
