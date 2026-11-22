"""Calibrate visual KL using the released MMFlash image recipe."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

import numpy as np

SCALE = 10_000


def fingerprint_input_ids(ids: Sequence[int]) -> str:
    """BLAKE2b-64 over int32 ids, matching the authors' sidecar on A100 hosts."""
    return hashlib.blake2b(
        np.asarray(ids, dtype="<i4").tobytes(), digest_size=8
    ).hexdigest()


class VisualScoreTable:
    def __init__(self, entries):
        self._entries = entries

    @classmethod
    def load(cls, path):
        raw = {}
        with open(path, encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                key = str(row["id"])
                if key in raw:
                    raise ValueError(f"duplicate score id {key!r}")
                kl = np.asarray(row["kl"], dtype=np.float32)
                entropy = np.asarray(row["entropy"], dtype=np.float32)
                n = int(row["n_loss"])
                if n <= 0 or kl.shape != (n,) or entropy.shape != (n,):
                    raise ValueError(f"invalid score vector lengths for {key!r}")
                if not np.isfinite(kl).all() or not np.isfinite(entropy).all():
                    raise ValueError(f"non-finite scores for {key!r}")
                raw[key] = (row["n_tokens"], row["fp"], row["mask_fp"], kl, entropy)
        if not raw:
            raise ValueError("the scoring corpus contains no eligible image rows")
        # The released recipe uses 1001 quantiles of the full training corpus.
        # right insertion / 1000 (including ties) reproduces the author transform.
        all_kl = np.concatenate([entry[3] for entry in raw.values()])
        edges = np.quantile(all_kl, np.linspace(0, 1, 1001))
        entries = {}
        for key, (length, fp, mask_fp, kl, entropy) in raw.items():
            rank = (
                (np.searchsorted(edges, kl, side="right") / 1000)
                .clip(0, 1)
                .astype(np.float32)
            )
            g = rank * np.exp(-np.maximum(entropy, 0))
            entries[key] = (length, fp, mask_fp, np.rint(g * SCALE).astype(np.int16))
        return cls(entries)

    def channel(self, key: str, ids: Sequence[int], mask: Sequence[int]) -> list[int]:
        if key not in self._entries:
            raise ValueError(f"missing scores for image row {key!r}")
        length, fp, mask_fp, compact = self._entries[key]
        positions = np.flatnonzero(mask)
        if (
            len(ids) != length
            or len(mask) != length
            or len(positions) != len(compact)
            or fingerprint_input_ids(ids) != fp
            or fingerprint_input_ids(mask) != mask_fp
        ):
            raise ValueError(f"visual score tokens/mask do not align for row {key!r}")
        channel = np.zeros(length, dtype=np.int16)
        channel[positions] = compact
        return channel.tolist()
