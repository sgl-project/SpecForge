"""Build attention against the caller's context/cache contract."""

from specforge.config.draft.component_registry import build_component


def build_dflash_attention(spec, *, config, layer_idx, kernels):
    return build_component(
        "attention",
        spec,
        "context_then_draft",
        config=config,
        layer_idx=layer_idx,
        kernels=kernels,
    )


def build_gqa(*, spec, config, layer_idx, kernels):
    from .gqa import Qwen3DFlashAttention

    return Qwen3DFlashAttention(config=config, layer_idx=layer_idx, kernels=kernels)


def build_mla(*, spec, config, layer_idx, kernels):
    from .mla import Qwen3DFlashMLAAttention

    return Qwen3DFlashMLAAttention(config=config, layer_idx=layer_idx, kernels=kernels)


def build_eagle(*, spec, config, attention_backend):
    from .eagle import (
        LlamaAttention,
        LlamaFlashAttention,
        LlamaFlexAttention,
        LlamaUSPFlashAttention,
    )

    constructors = {
        "sdpa": LlamaAttention,
        "flex_attention": LlamaFlexAttention,
        "fa": LlamaFlashAttention,
        "usp": LlamaUSPFlashAttention,
    }
    if attention_backend not in constructors:
        raise ValueError(f"Unknown attention backend {attention_backend}")
    return constructors[attention_backend](config=config)


def build_peagle(*, spec, config, input_size):
    from ...peagle import PEagleAttention

    return PEagleAttention(config, input_size=input_size)


def build_mtp(*, spec, config, layer_idx):
    from ...specialized.qwen3_5_mtp.model import Qwen3MTPAttention

    return Qwen3MTPAttention(config=config, layer_idx=layer_idx)
