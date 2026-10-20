# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Sparse MoE FFN for draft models, configured by ``moe_preset``; see ``DESIGN.md``."""

from .balance import (
    BALANCE_CONTROLLERS,
    BalanceController,
    build_balance_controller,
    register_balance_controller,
)
from .base import RoutingResult, TrainingState
from .config import (
    MOE_PRESETS,
    MoEConfig,
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
    "TrainingState",
    "WarmStartPlan",
    "apply_pending_balance_updates",
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
