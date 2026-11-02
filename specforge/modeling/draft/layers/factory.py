"""Explicit constructors for the currently supported draft components."""

from torch import nn
from transformers import PretrainedConfig

from .attention.base import Attention
from .config import AttentionConfig, LayerKernels, ResidualConfig
from .ffn.moe.layer import build_ffn as build_dense_or_moe_ffn
from .residual.base import Residual
from .residual.standard import StandardResidual


def build_attention(
    config: PretrainedConfig,
    layer_idx: int,
    kernels: LayerKernels,
    options: AttentionConfig,
) -> Attention:
    if options.kind == "mla":
        from .attention.mla import MLAAttention

        return MLAAttention(config, layer_idx, kernels, options)
    if options.kind == "deepseek_v4":
        from .attention.deepseek_v4 import DeepseekV4Attention

        return DeepseekV4Attention(config, layer_idx, kernels, options)
    if (
        options.kind == "mha"
        and config.num_key_value_heads != config.num_attention_heads
    ):
        raise ValueError("MHA requires num_key_value_heads == num_attention_heads")
    from .attention.gqa import GQAAttention

    return GQAAttention(config, layer_idx, kernels, options)


def build_ffn(
    config: PretrainedConfig, layer_idx: int, kernels: LayerKernels
) -> nn.Module:
    """``kernels.make_mlp(config)``, or for an MoE draft (``moe_preset``) the family's MoE layer."""
    # Return the provider's module directly: adding a wrapper changes checkpoint
    # paths and can change optimizer parameter ordering.
    return build_dense_or_moe_ffn(config, layer_idx, dense=kernels.make_mlp)


def build_residual(
    options: ResidualConfig = ResidualConfig(),
    *,
    hidden_size: int | None = None,
    norm_eps: float = 1e-6,
) -> Residual:
    if options.kind == "standard":
        return StandardResidual()
    if hidden_size is None:
        raise ValueError("Hyper-connections require hidden_size")
    if options.kind == "gr":
        from .residual.gr import GatedResidual

        return GatedResidual(
            hidden_size, options.num_streams, options.gate_rank, norm_eps
        )
    from .residual.mhc import ManifoldHyperConnection

    return ManifoldHyperConnection(
        hidden_size,
        options.num_streams,
        norm_eps,
        options.sinkhorn_iters,
        options.sinkhorn_eps,
        pre_from_prev=options.pre_from_prev,
    )


def build_residual_readout(
    options: ResidualConfig, *, hidden_size: int, norm_eps: float
) -> nn.Module:
    """The model-level read of the residual streams before the final norm."""
    if options.kind == "gr":
        from .residual.gr import GatedResidual

        return GatedResidual(
            hidden_size, options.num_streams, options.gate_rank, norm_eps, readout=True
        )
    if options.kind == "mhc" and not options.pre_from_prev:
        from .residual.mhc import HyperConnectionHead

        return HyperConnectionHead(
            hidden_size, options.num_streams, norm_eps, options.sinkhorn_eps
        )
    return nn.Identity()
