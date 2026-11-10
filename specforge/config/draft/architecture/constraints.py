"""Architecture cross-node constraints, called by the shared validation stage."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..model import DraftConfig


def validate_target_context(config: DraftConfig) -> None:
    """Constraints shared by architectures that inject full-vocabulary context."""
    if config.draft_vocab_size != config.vocab_size:
        raise ValueError(
            "target-context protocols require draft_vocab_size == vocab_size"
        )
    if config.spec_config.target_features.mask_token_id >= config.vocab_size:
        raise ValueError(
            "spec_config.target_features.mask_token_id must be below vocab_size"
        )
    # bound tables are the target's own, at the target's width
    if (
        config.target_bindings
        and config.spec_config.target_features.hidden_size != config.hidden_size
    ):
        raise ValueError(
            "target_bindings require spec_config.target_features.hidden_size "
            "== hidden_size"
        )
    for name, attention in config.attention_configs.items():
        if "target_context_kv" not in attention.capabilities():
            raise ValueError(
                f"attention_configs.{name}: architecture requires target_context_kv"
            )


def validate_dflash2(config: DraftConfig) -> None:
    validate_target_context(config)
    if config.spec_config.selector_top_k > config.draft_vocab_size:
        raise ValueError("spec_config.selector_top_k must not exceed draft_vocab_size")
    for name, layer in config.layer_configs.items():
        if layer.sublayer_conv is None:
            raise ValueError(f"layer_configs.{name}: DFlash2 requires sublayer_conv")
        if (
            "block_sublayer_pair"
            not in config.conv_configs[layer.sublayer_conv].capabilities()
        ):
            raise ValueError(
                f"layer_configs.{name}: convolution must support block_sublayer_pair"
            )
