# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""DFlashAdapter: the DFlash schema pinned onto PolicyFeatureAdapter.

DFlash's store-ready dict is ``{input_ids, hidden_states, loss_mask}`` — note
``hidden_states`` is the concatenated target capture layers, and there is NO
``target`` distribution / no vocab projection (DFlash trains on hard real-token
labels), so ``DFLASH_FEATURE_SCHEMA`` sets ``target_feature=None`` and skips the
``t2d`` projection step entirely.

``verify_feature_contract`` (run by the RolloutWorker before any store write)
keys its eagle3-specific aux/target checks on the feature names
``"hidden_state"`` / ``"target"``, which DFlash does not emit, so those checks
self-skip; the recorded-aux-layer check is skipped too because the schema does
not emit ``__aux_layer_ids__`` and the RolloutWorker reads it via
``feats.pop("__aux_layer_ids__", None)``.

Kept as a named class for back-compat; the shared runtime conversion lives in
:class:`~specforge.inference.adapters.policy.PolicyFeatureAdapter`.
"""

from __future__ import annotations

from typing import Optional

import torch

from specforge.inference.adapters.policy import (
    DFLASH_FEATURE_SCHEMA,
    PolicyFeatureAdapter,
)


class DFlashAdapter(PolicyFeatureAdapter):
    """DFlash ``FeatureSource`` over a ``TargetEngine`` (via its generic ``capture()``)."""

    SUPPORTED_FEATURE_NAMES = DFLASH_FEATURE_SCHEMA.names

    def __init__(
        self,
        target_model,
        *,
        device: str = "cuda",
        t2d: Optional[torch.Tensor] = None,  # unused (DFlash has no vocab map); kept
    ) -> None:  # for a uniform make_adapter(target_model, *, device, t2d) signature
        super().__init__(
            target_model,
            schema=DFLASH_FEATURE_SCHEMA,
            device=device,
            t2d=t2d,
            # DFlash engines' capture() does not take shard_returns; never pass it.
            shard_returns=None,
        )


__all__ = ["DFlashAdapter"]
