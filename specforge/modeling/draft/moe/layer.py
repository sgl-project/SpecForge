# coding=utf-8
"""The FFN switch between a draft layer's dense MLP and an MoE layer."""

from typing import Callable

from torch import nn
from transformers import PretrainedConfig


def build_ffn(
    config: PretrainedConfig,
    *,
    dense: Callable[[PretrainedConfig], nn.Module],
) -> nn.Module:
    """``dense(config)``, or an MoE FFN when the config names a ``moe_preset``."""

    moe_preset = getattr(config, "moe_preset", None)
    if moe_preset is None:
        return dense(config)
    raise NotImplementedError(
        f"moe_preset={moe_preset!r}: the MoE FFN is not implemented yet"
    )
