"""FFN construction with injectable dense kernels and explicit variant checks."""

from typing import Callable

from torch import nn
from transformers import PretrainedConfig

from specforge.config.draft.components import component_config


def build_ffn(
    config: PretrainedConfig, *, dense_factory: Callable[[PretrainedConfig], nn.Module]
) -> nn.Module:
    """Build a token-wise FFN preserving the final hidden dimension.

    Future MoE variants own their routing/shared-expert semantics and must
    explicitly integrate any auxiliary training outputs with the caller.
    """
    choice = component_config(config, "ffn_config", "dense")
    if choice.type != "dense" or choice.variant is not None:
        raise ValueError(f"FFN {choice!r} is not implemented")
    return dense_factory(config)
