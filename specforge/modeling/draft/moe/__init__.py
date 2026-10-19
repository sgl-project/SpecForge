# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Sparse MoE FFN for draft models.

One configurable layer, not one block per target family. A draft JSON names a
``moe_preset`` (a target family's routing recipe) and may override any
architecture knob for ablations; dispatch, checkpoint naming, FSDP layout and
deferred balance updates are shared. See ``DESIGN.md``.

Modules:

- :mod:`.base`        ``Registry``, ``RoutingResult``, ``TrainingState``
- :mod:`.config`      ``MoEConfig`` + presets; resolves a draft JSON
- :mod:`.router`      score functions and routers (with router centering)
- :mod:`.balance`     balancing policies (none, noaux_tc, aux_loss)
- :mod:`.experts`     routed experts
- :mod:`.shared`      the shared expert
- :mod:`.layer`       ``MoELayer``, ``build_ffn`` and the model-level hooks
- :mod:`.checkpoint`  checkpoint layouts, the export fold
- :mod:`.families`    per-family presets and layouts
"""

from .balance import BalanceController, register_balance_controller
from .base import RoutingResult, TrainingState
from .checkpoint import (
    exported_config_updates,
    from_checkpoint_state_dict,
    register_layout,
    to_checkpoint_state_dict,
    to_serving_state_dict,
)
from .config import MoEConfig, is_moe_config, register_moe_preset, resolve_moe_config
from .experts import RoutedExperts, register_experts_backend
from .layer import (
    MoELayer,
    apply_pending_balance_updates,
    build_ffn,
    collect_moe_aux_loss,
    collect_moe_metrics,
    iter_moe_layers,
)
from .router import Router, register_router, register_score_function
from .shared import SharedExpert, register_shared_expert

# Registers every family's preset and layout.
from . import families  # isort: skip

__all__ = [
    "BalanceController",
    "MoEConfig",
    "MoELayer",
    "RoutedExperts",
    "Router",
    "RoutingResult",
    "SharedExpert",
    "TrainingState",
    "apply_pending_balance_updates",
    "build_ffn",
    "collect_moe_aux_loss",
    "collect_moe_metrics",
    "exported_config_updates",
    "families",
    "from_checkpoint_state_dict",
    "is_moe_config",
    "iter_moe_layers",
    "register_balance_controller",
    "register_experts_backend",
    "register_layout",
    "register_moe_preset",
    "register_router",
    "register_score_function",
    "register_shared_expert",
    "resolve_moe_config",
    "to_checkpoint_state_dict",
    "to_serving_state_dict",
]
