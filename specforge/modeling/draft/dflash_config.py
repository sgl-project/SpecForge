"""Compatibility exports; new code imports draft.config or its algorithm module."""

from .config.dflash import (
    DFlashConfig,
    build_target_layer_ids,
    resolve_dflash_attention_layout,
    resolve_dflash_attention_mode,
    resolve_dflash_causal_block,
    validate_dflash_attention_config,
    validate_dflash_mla_config,
)
from .config.dflash2 import DFlash2Config
from .config.domino import DominoConfig
from .config.dspark import DSparkConfig

__all__ = [
    "resolve_dflash_causal_block",
    "resolve_dflash_attention_layout",
    "resolve_dflash_attention_mode",
    "validate_dflash_mla_config",
    "validate_dflash_attention_config",
    "build_target_layer_ids",
    "DFlashConfig",
    "DFlash2Config",
    "DominoConfig",
    "DSparkConfig",
]
