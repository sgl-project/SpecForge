# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Dataset vocabulary mappings shared by online and offline EAGLE training."""

from __future__ import annotations

from collections import Counter
from typing import Optional


def count_effective_feature_tokens(
    hidden_states_path: str,
    *,
    max_length: Optional[int] = None,
    target_vocab_size: Optional[int] = None,
) -> Counter:
    """Count loss-bearing tokens directly from prepared offline features.

    Offline feature files already contain the exact ``input_ids`` and
    ``loss_mask`` used for training. Reading those tensors avoids requiring the
    original raw conversation dataset solely to rebuild a vocabulary map.
    """
    from specforge.runtime.data_plane.feature_store import load_feature_file
    from specforge.runtime.data_plane.offline_reader import list_feature_files

    paths = list_feature_files(hidden_states_path)
    if not paths:
        raise ValueError(f"no offline feature files found under {hidden_states_path!r}")

    counts: Counter = Counter()
    for path in paths:
        raw = load_feature_file(path)
        missing = [name for name in ("input_ids", "loss_mask") if name not in raw]
        if missing:
            raise KeyError(
                f"{path} cannot derive an EAGLE vocab mapping; missing {missing}"
            )
        input_ids = raw["input_ids"].reshape(-1)
        loss_mask = raw["loss_mask"].reshape(-1)
        if input_ids.numel() != loss_mask.numel():
            raise ValueError(
                f"{path} has {input_ids.numel()} input ids but "
                f"{loss_mask.numel()} loss-mask entries"
            )
        if max_length is not None:
            input_ids = input_ids[:max_length]
            loss_mask = loss_mask[:max_length]
        selected = input_ids[loss_mask.to(dtype=bool)]
        if selected.numel() == 0:
            continue
        token_ids, frequencies = selected.unique(return_counts=True)
        for token_id, frequency in zip(token_ids.tolist(), frequencies.tolist()):
            token_id = int(token_id)
            if token_id < 0:
                raise ValueError(f"{path} contains negative token id {token_id}")
            if target_vocab_size is not None and token_id >= target_vocab_size:
                raise ValueError(
                    f"{path} contains token id {token_id}, outside target vocab "
                    f"size {target_vocab_size}"
                )
            counts[token_id] += int(frequency)
    return counts


__all__ = ["count_effective_feature_tokens"]
