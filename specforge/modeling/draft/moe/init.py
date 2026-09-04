# coding=utf-8
"""Warm-start plans: which target experts seed which draft experts.

A drafter whose MoE matches the target's expert shape can inherit expert
weights instead of training them from scratch. This module holds the
mapping (:func:`plan_warm_start`) and applying it to one ``MoELayer`` from a
target layer's *dequantized* tensors in official naming
(:func:`apply_warm_start`). Reading and dequantizing the target checkpoint is
target-specific and lives with the target's tooling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Tuple

import torch

from .config import MoEConfig
from .layer import MoELayer
from .state_dict import from_checkpoint_state_dict


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


_EXPERT_WEIGHTS = ("w1", "w2", "w3")


def apply_warm_start(
    layer: MoELayer, plan: WarmStartPlan, source: Mapping[str, torch.Tensor]
) -> List[str]:
    """Seed ``layer`` from one target MoE layer.

    ``source`` holds the target layer's tensors in official naming, relative
    to the layer: ``experts.{j}.w{1,2,3}.weight``, ``gate.weight`` ``[E_t, H]``,
    optionally ``gate.bias`` ``[E_t]`` and ``shared_experts.w{1,2,3}.weight``.
    Returns the (module-native) keys that were loaded.
    """
    if plan.n_draft_experts != layer.cfg.n_routed_experts:
        raise ValueError(
            f"plan seeds {plan.n_draft_experts} experts but the layer has "
            f"{layer.cfg.n_routed_experts}"
        )
    official = {}
    for i, j in enumerate(plan.target_expert_ids):
        for w in _EXPERT_WEIGHTS:
            official[f"experts.{i}.{w}.weight"] = source[f"experts.{j}.{w}.weight"]
    if plan.copy_gate_rows:
        rows = torch.as_tensor(plan.target_expert_ids, dtype=torch.long)
        official["gate.weight"] = source["gate.weight"][rows]
        if "gate.bias" in source:
            official["gate.bias"] = source["gate.bias"][rows]
    if plan.copy_shared_expert and layer.shared_experts is not None:
        for w in _EXPERT_WEIGHTS:
            official[f"shared_experts.{w}.weight"] = source[
                f"shared_experts.{w}.weight"
            ]
    native = from_checkpoint_state_dict(official)
    result = layer.load_state_dict(native, strict=False)
    if result.unexpected_keys:
        raise KeyError(f"warm start produced unexpected keys: {result.unexpected_keys}")
    return sorted(native)
