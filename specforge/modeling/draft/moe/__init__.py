# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Sparse MoE FFN for draft models, configured by ``moe_preset``; see ``DESIGN.md``."""

from .balance import BalanceController, register_balance_controller
from .base import RoutingResult, TrainingState
from .checkpoint import (
    exported_config_updates,
    from_checkpoint_state_dict,
    register_layout,
    serving_architecture,
    serving_config,
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
    "serving_architecture",
    "serving_config",
    "to_checkpoint_state_dict",
    "to_serving_state_dict",
]
