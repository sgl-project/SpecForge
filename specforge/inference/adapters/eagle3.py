# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""SGLangAdapter: the EAGLE3 schema pinned onto PolicyFeatureAdapter.

The runtime conversion (length grouping, batched ``TargetEngine.capture``,
per-sample slicing, target->draft projection, order preservation) lives once in
:class:`~specforge.inference.adapters.policy.PolicyFeatureAdapter`; this class
only pins ``EAGLE3_FEATURE_SCHEMA`` — the store-ready dict is ``{input_ids,
attention_mask, loss_mask, hidden_state, target}`` plus the out-of-band
``__aux_layer_ids__`` record that ``verify_feature_contract`` checks.

Kept as a named class for back-compat (it predates the schema merge and is the
default adapter name throughout the runtime docs/tests).
"""

from __future__ import annotations

from typing import Optional

import torch

from specforge.inference.adapters.policy import (
    EAGLE3_FEATURE_SCHEMA,
    PolicyFeatureAdapter,
)


class SGLangAdapter(PolicyFeatureAdapter):
    """EAGLE3 ``FeatureSource`` over a ``TargetEngine`` (via its generic ``capture()``)."""

    SUPPORTED_FEATURE_NAMES = EAGLE3_FEATURE_SCHEMA.names

    def __init__(
        self,
        target_model,
        *,
        device: str = "cuda",
        t2d: Optional[torch.Tensor] = None,
        shard_returns: bool = False,
    ) -> None:
        super().__init__(
            target_model,
            schema=EAGLE3_FEATURE_SCHEMA,
            device=device,
            t2d=t2d,
            shard_returns=shard_returns,
        )


__all__ = ["SGLangAdapter"]
