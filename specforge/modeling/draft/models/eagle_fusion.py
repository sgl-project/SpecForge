"""EAGLE input computation owned by models, not the shared decoder loop."""

import torch
from torch import Tensor, nn

from specforge.config.draft._validation import boolean, positive_int
from specforge.config.draft.architecture.eagle_fusion import AuxiliaryFeatureSpec
from specforge.config.draft.norm import NormSpec

from ..layers import build_component, component_builder
from ..layers.attention.backends import PreparedAttention
from ..layers.attention.kv import AttentionKV
from ..layers.inputs import LayerInputs


class AuxiliaryFeatureProjection(nn.Module):
    """Optionally normalize each target feature separately before projection."""

    def __init__(
        self,
        config: AuxiliaryFeatureSpec,
        *,
        hidden_size: int,
        normalize_features: bool,
        norm: NormSpec,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if type(config) is not AuxiliaryFeatureSpec:
            raise TypeError("auxiliary projection requires AuxiliaryFeatureSpec")
        positive_int("hidden_size", hidden_size)
        boolean("normalize_features", normalize_features)
        if normalize_features:
            component_builder("norm", norm.type)
        self.config = config
        self.fc_norm = (
            nn.ModuleList(
                [
                    build_component(
                        "norm",
                        norm,
                        hidden_size=config.hidden_size,
                        device=device,
                        dtype=dtype,
                    )
                    for _ in config.target_layer_ids
                ]
            )
            if normalize_features
            else None
        )
        self.fc = nn.Linear(
            config.input_width, hidden_size, bias=False, device=device, dtype=dtype
        )

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3 or features.shape[-1] != self.config.input_width:
            raise ValueError(
                "auxiliary features must be [batch, sequence, concatenated_width]"
            )
        if self.fc_norm is not None:
            features = torch.cat(
                [
                    norm(chunk)
                    for norm, chunk in zip(
                        self.fc_norm, features.split(self.config.hidden_size, dim=-1)
                    )
                ],
                dim=-1,
            )
        return self.fc(features)


class EagleDecoderStage(nn.Module):
    """First-layer fusion: [token embedding, projected features] -> H.

    Own the two input norms and residual choice while reusing constructed
    attention/FFN components. Later layers use the ordinary DecoderLayer.
    No algorithm-specific callback is needed in DraftBackbone.
    """

    def __init__(
        self,
        attention: nn.Module,
        mlp: nn.Module,
        *,
        hidden_size: int,
        norm: NormSpec,
        norm_before_residual: bool,
        device=None,
        dtype=None,
    ):
        super().__init__()
        positive_int("hidden_size", hidden_size)
        boolean("norm_before_residual", norm_before_residual)
        if not isinstance(attention, nn.Module) or not isinstance(mlp, nn.Module):
            raise TypeError("attention and mlp must be constructed modules")
        self.hidden_size = hidden_size
        self.norm_before_residual = norm_before_residual
        self.self_attn = attention
        self.mlp = mlp
        options = dict(hidden_size=hidden_size, device=device, dtype=dtype)
        self.input_layernorm = build_component("norm", norm, **options)
        self.hidden_norm = build_component("norm", norm, **options)
        self.post_attention_layernorm = build_component("norm", norm, **options)

    def forward(
        self,
        hidden: Tensor,
        inputs: LayerInputs,
        *,
        kv_prefix: AttentionKV | None = None,
        output_kv: bool = False,
    ) -> Tensor | tuple[Tensor, AttentionKV]:
        if hidden.ndim != 3 or hidden.shape[-1] != 2 * self.hidden_size:
            raise ValueError(
                "EAGLE first stage requires [batch, sequence, 2 * hidden_size]"
            )
        if not isinstance(inputs, LayerInputs):
            raise TypeError("EAGLE first stage requires LayerInputs")
        if not isinstance(inputs.attention, PreparedAttention):
            raise ValueError("EAGLE first stage requires prepared attention")
        if inputs.target_context is not None:
            raise ValueError(
                "EAGLE fuses features into queries, not a target KV context"
            )
        embeds, features = hidden.split(self.hidden_size, dim=-1)
        residual = features
        embeds = self.input_layernorm(embeds)
        features = self.hidden_norm(features)
        if self.norm_before_residual:
            residual = features
        kv_options = dict(kv_prefix=kv_prefix, output_kv=True) if output_kv else {}
        if kv_prefix is not None and not output_kv:
            raise ValueError("a K/V prefix requires output_kv=True")
        update = self.self_attn(
            torch.cat((embeds, features), dim=-1),
            attention=inputs.attention,
            position_embeddings=inputs.position_embeddings,
            **kv_options,
        )
        if output_kv:
            update, current_kv = update
        if update.shape != residual.shape:
            raise ValueError(
                "first attention must project the fused stream to hidden_size"
            )
        hidden = residual + update
        update = self.mlp(self.post_attention_layernorm(hidden))
        if update.shape != hidden.shape:
            raise ValueError("mlp output must preserve the residual shape")
        hidden = hidden + update
        return (hidden, current_kv) if output_kv else hidden
