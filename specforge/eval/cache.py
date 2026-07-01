# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""EvalCache: the on-disk cache for materialized eval feature sets (E1).

Repeat evals over an unchanged *(eval-data, target, revision, tokenizer,
template, aux-layers, seqlen)* tuple must not recompute target hidden states.
The cache maps an :class:`~specforge.eval.evaluator.EvalConfig` identity to a
directory of materialized eval features — whatever the producer writes there
(offline feature files a ``spec.make_offline_reader`` reads back into
``SampleRef``s for a ``FeatureDataLoader``).

``get_or_produce`` is the whole consumption surface: a hit returns the cached
directory; a miss runs the producer into a staging directory that is renamed
into place atomically — the rename IS the completeness marker, so a crashed
producer leaves only a stale ``*.tmp-*`` staging dir, never a half-visible
entry. A ``eval_cache_meta.json`` with the plain identity fields is written
beside the features for humans debugging a cache.

Deliberately **separate** from the tokenization cache under ``specforge/data``
— different key fields, different lifecycle; do not merge (plan §4.4). The
cache is process-local: multi-rank callers should produce on rank 0 and
barrier before reading (it does no cross-rank coordination itself).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from typing import Callable, Optional

from specforge.eval.evaluator import EvalConfig


class EvalCache:
    def __init__(self, cache_dir: str) -> None:
        self.root = os.path.join(cache_dir, "eval_cache")

    # -- identity ------------------------------------------------------------
    def key(self, config: EvalConfig) -> str:
        """Digest of the eval identity — flipping ANY identity field changes it,
        and execution knobs (micro-batch size) deliberately do not."""
        content = "|".join(config.identity_fields())
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def dir_for(self, config: EvalConfig) -> str:
        return os.path.join(self.root, self.key(config))

    # -- lookup / produce ------------------------------------------------------
    def try_get(self, config: EvalConfig) -> Optional[str]:
        """The cached feature directory for this identity, or ``None`` on a miss."""
        path = self.dir_for(config)
        return path if os.path.isdir(path) else None

    def get_or_produce(
        self, config: EvalConfig, produce: Callable[[str], None]
    ) -> str:
        """Return the eval feature directory, producing it once on a miss.

        ``produce(staging_dir)`` materializes the eval set (e.g. runs the
        target capture over the eval prompts) into an empty staging directory;
        on return the staging is atomically renamed into place. If a concurrent
        producer won the race, its entry is used and this one's staging is
        discarded — either way exactly one complete entry exists afterwards.
        """
        hit = self.try_get(config)
        if hit is not None:
            return hit
        final = self.dir_for(config)
        os.makedirs(self.root, exist_ok=True)
        staging = f"{final}.tmp-{uuid.uuid4().hex[:8]}"
        os.makedirs(staging)
        try:
            produce(staging)
            with open(os.path.join(staging, "eval_cache_meta.json"), "w") as fh:
                json.dump(
                    {"key": self.key(config), "identity": config.identity_fields()},
                    fh,
                    indent=2,
                )
            try:
                os.rename(staging, final)
            except OSError:
                # Lost a produce race: another process published first — theirs
                # is complete, use it. Anything else is a real failure.
                if not os.path.isdir(final):
                    raise
        finally:
            if os.path.isdir(staging):
                shutil.rmtree(staging, ignore_errors=True)
        return final

    def invalidate(self, config: EvalConfig) -> bool:
        """Drop the entry for this identity; ``True`` if one existed."""
        path = self.try_get(config)
        if path is None:
            return False
        shutil.rmtree(path)
        return True


__all__ = ["EvalCache"]
