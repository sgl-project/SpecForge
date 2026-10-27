"""Explicit constructors for the currently supported draft components."""

from torch import nn
from transformers import PretrainedConfig

from .attention.base import Attention
from .config import AttentionConfig, LayerKernels, ResidualConfig
from .residual.base import Residual
from .residual.standard import StandardResidual


def build_attention(
    config: PretrainedConfig,
    layer_idx: int,
    kernels: LayerKernels,
    options: AttentionConfig,
) -> Attention:
    if options.kind in ("qsa", "dsa"):
        raise NotImplementedError(
            f"{options.kind.upper()} has sparse interfaces but no implementation yet"
        )
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
    )
