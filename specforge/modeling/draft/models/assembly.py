"""Assemble ordinary decoder stacks without taking over algorithm execution."""

from dataclasses import replace

from torch import nn

from specforge.config.draft import DraftConfig
from specforge.config.draft.architecture.features import TargetFeatureSpec
from specforge.config.draft.residual import StandardResidualSpec

from ..backbone import DraftBackbone
from ..layers import DecoderLayer, build_component, component_builder
from ..layers.attention.backends import AttentionPreparationScope
from ..layers.rope import build_rotary
from .target_features import TargetFeatureFusion
from .tokens import build_token_modules


class DecoderBackbone(DraftBackbone):
    """Prepare attention once per forward for the assembled decoder stages."""

    def begin(self, hidden, layer_inputs):
        if len(layer_inputs) != self.num_layers:
            raise ValueError("layer_inputs must contain one entry per layer")
        with AttentionPreparationScope() as scope:
            prepared = [
                (
                    replace(
                        inputs,
                        attention=layer.self_attn.prepare(
                            inputs.attention_layout, scope, batch_size=hidden.shape[0]
                        ),
                        attention_layout=None,
                    )
                    if inputs.attention_layout is not None
                    else inputs
                )
                for layer, inputs in zip(self.layers, layer_inputs)
            ]
        return super().begin(hidden, prepared)


def build_decoder_backbone(
    config: DraftConfig, *, backend: str, device=None, dtype=None
) -> DraftBackbone:
    """Build an H→H, single-stream stack with independent parameters per layer.

    Algorithm models still own fusion, inter-layer execution, and heads. Models
    whose attention input width differs from H must assemble that stage explicitly.
    Repeated names reuse configuration nodes, never module instances or weights.
    """
    if type(config) is not DraftConfig:
        raise TypeError("decoder assembly requires a resolved DraftConfig")
    if type(config.residual_config) is not StandardResidualSpec:
        raise ValueError("decoder assembly currently requires standard residuals")
    if any(
        config.spec_config.attention_input_width(config.hidden_size, index)
        != config.hidden_size
        for index in range(config.num_hidden_layers)
    ):
        raise ValueError(
            "model must assemble its nonstandard attention input fusion explicitly"
        )
    # Resolve every implementation before the first parameter is allocated.
    for kind, nodes in (
        ("attention", config.attention_configs.values()),
        ("mlp", config.mlp_configs.values()),
        ("conv", config.conv_configs.values()),
        ("norm", (config.norm_config,)),
        ("residual", (config.residual_config,)),
    ):
        for node in nodes:
            component_builder(kind, node.type)
    definitions = [config.layer_configs[name] for name in config.layer_types]
    attentions = [config.attention_configs[layer.attention] for layer in definitions]
    ropes = [node.effective_rope(config.rope_parameters) for node in attentions]
    rotary = [
        (
            None
            if node.rotary_input_dim is None
            else build_rotary(rope, head_dim=node.rotary_input_dim)
        )
        for node, rope in zip(attentions, ropes)
    ]
    protocol = config.spec_config.protocol
    block_size = None if protocol is None else protocol.num_model_slots
    options = dict(hidden_size=config.hidden_size, device=device, dtype=dtype)
    residual = build_component("residual", config.residual_config)
    layers = []
    for index, (layer, attention, rope) in enumerate(
        zip(definitions, attentions, ropes)
    ):
        conv = None
        if layer.sublayer_conv is not None:
            conv = tuple(
                build_component(
                    "conv",
                    config.conv_configs[layer.sublayer_conv],
                    block_size=block_size,
                    **options,
                )
                for _ in range(2)
            )
        layers.append(
            DecoderLayer(
                build_component(
                    "attention",
                    attention,
                    input_size=config.hidden_size,
                    norm=config.norm_config,
                    rope=rope,
                    backend=backend,
                    **options,
                ),
                build_component("mlp", config.mlp_configs[layer.mlp], **options),
                norm=config.norm_config,
                attention_connection=residual.build_connection(
                    layer_index=index, sublayer="attention", **options
                ),
                mlp_connection=residual.build_connection(
                    layer_index=index, sublayer="mlp", **options
                ),
                sublayer_conv=conv,
                **options,
            )
        )
    return DecoderBackbone(
        layers,
        final_norm=build_component("norm", config.norm_config, **options),
        rotary=rotary,
        residual=residual,
    )


def build_target_context_modules(
    config: DraftConfig,
    target_features: TargetFeatureSpec,
    *,
    backend: str,
    device=None,
    dtype=None,
) -> tuple[DraftBackbone, TargetFeatureFusion, nn.Module, nn.Module]:
    """Build backbone, context fusion, embedding and head; the model owns initialization."""
    backbone = build_decoder_backbone(
        config, backend=backend, device=device, dtype=dtype
    )
    fusion = TargetFeatureFusion(
        target_features,
        hidden_size=config.hidden_size,
        norm=config.norm_config,
        device=device,
        dtype=dtype,
    )
    embedding, head = build_token_modules(config, device=device, dtype=dtype)
    return backbone, fusion, embedding, head
