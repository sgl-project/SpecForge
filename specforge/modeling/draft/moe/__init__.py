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
naming, FSDP layout, deferred balance updates) are shared. See ``DESIGN.md``.

Modules:

- :mod:`.config`      ``MoEConfig`` + the preset registry; resolves a draft JSON
- :mod:`.router`      ``Router`` contract (scores -> top-k) + score functions
- :mod:`.balance`     ``BalanceController`` contract: balancing as a policy
- :mod:`.experts`     ``RoutedExperts`` contract: expert weights + dispatch
- :mod:`.shared`      ``SharedExpert`` contract
- :mod:`.layer`       ``MoELayer`` composition; ``build_ffn`` is the dense/MoE switch
- :mod:`.hooks`       model-level plumbing: balance updates, aux loss, metrics
- :mod:`.state_dict`  module layout <-> official checkpoint naming boundary
- :mod:`.init`        warm-start plans from a target model's experts

Implementations register into the registries from their own modules; this
package defines contracts and holds no routing math itself.
"""

from .balance import (
    BALANCE_CONTROLLERS,
    BalanceController,
    build_balance_controller,
    register_balance_controller,
)
from .config import (
    MOE_PRESETS,
    MoEConfig,
    available_moe_presets,
    is_moe_config,
    register_moe_preset,
    resolve_moe_config,
)
from .experts import (
    EXPERTS_BACKENDS,
    RoutedExperts,
    build_routed_experts,
    register_experts_backend,
)
from .hooks import (
    apply_pending_balance_updates,
    collect_moe_aux_loss,
    collect_moe_metrics,
    iter_moe_layers,
)
from .init import WarmStartPlan, plan_warm_start, select_target_experts
from .layer import MoELayer, build_ffn
from .router import (
    ROUTERS,
    SCORE_FUNCTIONS,
    Router,
    RoutingResult,
    build_router,
    get_score_function,
    register_router,
    register_score_function,
)
from .shared import (
    SHARED_EXPERTS,
    SharedExpert,
    build_shared_expert,
    register_shared_expert,
)
from .state_dict import (
    from_checkpoint_state_dict,
    register_state_dict_converter,
    to_checkpoint_state_dict,
)

__all__ = [
    "BALANCE_CONTROLLERS",
    "BalanceController",
    "EXPERTS_BACKENDS",
    "MOE_PRESETS",
    "MoEConfig",
    "MoELayer",
    "ROUTERS",
    "RoutedExperts",
    "Router",
    "RoutingResult",
    "SCORE_FUNCTIONS",
    "SHARED_EXPERTS",
    "SharedExpert",
    "WarmStartPlan",
    "apply_pending_balance_updates",
    "available_moe_presets",
    "build_balance_controller",
    "build_ffn",
    "build_routed_experts",
    "build_router",
    "build_shared_expert",
    "collect_moe_aux_loss",
    "collect_moe_metrics",
    "from_checkpoint_state_dict",
    "get_score_function",
    "is_moe_config",
    "iter_moe_layers",
    "plan_warm_start",
    "register_balance_controller",
    "register_experts_backend",
    "register_moe_preset",
    "register_router",
    "register_score_function",
    "register_shared_expert",
    "register_state_dict_converter",
    "resolve_moe_config",
    "select_target_experts",
    "to_checkpoint_state_dict",
]
