# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Sparse MoE FFN for DFlash-family drafts.

One configurable layer, not one block per target family. A draft JSON selects a
``moe_preset`` (the routing recipe of a target family) and may override any
architecture knob for ablations; the expensive parts (dispatch, checkpoint
naming, FSDP layout, deferred balance updates) are shared.

Modules:

- :mod:`.config`      ``MoEConfig`` + the preset registry; resolves a draft JSON
"""

from .config import (
    MOE_PRESETS,
    MoEConfig,
    available_moe_presets,
    is_moe_config,
    register_moe_preset,
    resolve_moe_config,
)

__all__ = [
    "MOE_PRESETS",
    "MoEConfig",
    "available_moe_presets",
    "is_moe_config",
    "register_moe_preset",
    "resolve_moe_config",
]
