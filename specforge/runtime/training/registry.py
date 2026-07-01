# coding=utf-8
"""E0 import shim — moved to ``specforge.training.strategies.registry``."""

from specforge.training.strategies.registry import (  # noqa: F401
    StrategySpec,
    available_strategies,
    concat_collate,
    register_strategy,
    resolve_strategy,
)

__all__ = [
    "StrategySpec",
    "concat_collate",
    "register_strategy",
    "resolve_strategy",
    "available_strategies",
]
