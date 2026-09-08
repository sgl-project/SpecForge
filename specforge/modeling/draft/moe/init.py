# coding=utf-8
"""Warm-start plans: which target experts seed which draft experts.

A drafter whose MoE matches the target's expert shape can inherit expert
weights instead of training them from scratch. This module holds the
implementation-independent part: choosing the mapping. Applying a plan needs
the target checkpoint's (possibly quantized) weights and the experts backend's
native layout, and is provided alongside each target-family preset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .config import MoEConfig


@dataclass(frozen=True)
class WarmStartPlan:
    """``target_expert_ids[i]`` is the target expert that seeds draft expert ``i``."""

    target_expert_ids: Tuple[int, ...]
    copy_shared_expert: bool = True
    copy_gate_rows: bool = True

    @property
    def n_draft_experts(self) -> int:
        return len(self.target_expert_ids)


def select_target_experts(
    n_target: int, n_draft: int, strategy: str = "strided"
) -> Tuple[int, ...]:
    """Pick ``n_draft`` distinct target experts.

    ``"strided"`` spreads picks evenly over the target's expert ids (the
    default: no assumption about which experts matter for the draft's data);
    ``"first"`` takes the leading ids.
    """
    if n_draft <= 0 or n_target <= 0:
        raise ValueError("expert counts must be positive")
    if n_draft > n_target:
        raise ValueError(
            f"cannot seed {n_draft} draft experts from {n_target} target experts"
        )
    if strategy == "first":
        return tuple(range(n_draft))
    if strategy == "strided":
        return tuple((i * n_target) // n_draft for i in range(n_draft))
    raise ValueError(f"unknown warm-start selection strategy {strategy!r}")


def plan_warm_start(
    cfg: MoEConfig, n_target_experts: int, strategy: str = "strided"
) -> WarmStartPlan:
    return WarmStartPlan(
        target_expert_ids=select_target_experts(
            n_target_experts, cfg.n_routed_experts, strategy
        ),
        copy_shared_expert=bool(cfg.n_shared_experts),
    )
