# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Config helpers shared by the EAGLE-4 draft and its attention backends."""

from __future__ import annotations

from typing import Any, Optional

EAGLE4_ARCH = "LlamaForCausalLMEagle4"


def draft_uses_rope(config: Any) -> bool:
    """RoPE is off if any of the three position settings disables it."""

    return not (
        getattr(config, "use_rope", True) is False
        or getattr(config, "position_embedding_type", None) == "none"
        or getattr(config, "partial_rotary_factor", None) == 0
    )


def draft_sliding_window(config: Any) -> Optional[int]:
    """Context window of an EAGLE-4 draft, or None.

    Enablement follows the serving loader: ``layer_types`` wins, then
    ``use_sliding_window``, then a bare ``sliding_window``. Other architectures
    keep full attention even when a leftover ``sliding_window`` is present.
    """

    architectures = getattr(config, "architectures", None) or ()
    if EAGLE4_ARCH not in architectures:
        return None
    window = getattr(config, "sliding_window", None)
    layer_types = getattr(config, "layer_types", None)
    enabled = getattr(config, "use_sliding_window", None)
    if layer_types is not None:
        enabled = "sliding_attention" in layer_types
    elif enabled is None:
        enabled = window is not None
    if enabled and not window:
        raise ValueError(f"Sliding window is enabled but sliding_window={window!r}")
    if not enabled:
        return None
    window = int(window)
    if window < 1:
        raise ValueError(f"sliding_window must be a positive integer, got {window!r}")
    return window


def parse_efficient_lm_head_config(config: Any) -> Optional[dict]:
    """Return a validated ``efficient_lm_head_config`` dict, or None."""

    head = getattr(config, "efficient_lm_head_config", None)
    if not head:
        return None
    if not isinstance(head, dict):
        raise ValueError(
            f"efficient_lm_head_config must be a dict, got {type(head).__name__}"
        )
    head_type = head.get("type")
    if head_type not in ("low_rank", "weighted_low_rank"):
        raise ValueError(
            "efficient_lm_head_config.type must be 'low_rank' or "
            f"'weighted_low_rank', got {head_type!r}"
        )
    rank = head.get("rank")
    num_candidates = head.get("num_candidates", head.get("top_k"))
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
        raise ValueError(f"efficient_lm_head_config.rank must be >= 1, got {rank!r}")
    if (
        not isinstance(num_candidates, int)
        or isinstance(num_candidates, bool)
        or num_candidates < 1
    ):
        raise ValueError(
            "efficient_lm_head_config.num_candidates must be >= 1, got "
            f"{num_candidates!r}"
        )
    parsed = {
        "type": head_type,
        "rank": rank,
        "num_candidates": num_candidates,
    }
    if "smooth_factor" in head:
        parsed["smooth_factor"] = head["smooth_factor"]
    return parsed


__all__ = [
    "EAGLE4_ARCH",
    "draft_sliding_window",
    "draft_uses_rope",
    "parse_efficient_lm_head_config",
]
