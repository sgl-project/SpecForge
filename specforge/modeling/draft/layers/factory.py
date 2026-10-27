"""Explicit constructors for the currently supported draft components."""

from torch import nn
from transformers import PretrainedConfig

from .attention.base import Attention
from .config import AttentionConfig, LayerKernels


def build_attention(
    config: PretrainedConfig,
    layer_idx: int,
    kernels: LayerKernels,
    options: AttentionConfig,
) -> Attention:
    if options.kind == "mla":
        from .attention.mla import MLAAttention

        return MLAAttention(config, layer_idx, kernels, options)
    if (
        options.kind == "mha"
        and config.num_key_value_heads != config.num_attention_heads
    ):
        raise ValueError("MHA requires num_key_value_heads == num_attention_heads")
    from .attention.gqa import GQAAttention

    return GQAAttention(config, layer_idx, kernels, options)


def build_ffn(config: PretrainedConfig, kernels: LayerKernels) -> nn.Module:
    # Return the provider's module directly: adding a wrapper changes checkpoint
    # paths and can change optimizer parameter ordering.
    return kernels.make_mlp(config)
