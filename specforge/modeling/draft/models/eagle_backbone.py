"""Assemble EAGLE's fused first stage and ordinary later decoder layers."""

from torch import nn

from specforge.config.draft import DraftConfig
from specforge.config.draft.architecture import Eagle3Spec, PEagleSpec
from specforge.config.draft.residual import StandardResidualSpec

from ..backbone import DraftBackbone
from ..layers import DecoderLayer, build_component, component_builder
from ..layers.rope import build_rotary
from .eagle_fusion import EagleDecoderStage


def build_eagle_backbone(
    config: DraftConfig, *, backend: str, device=None, dtype=None
) -> DraftBackbone:
    """Consume concatenated token embeddings and already projected features.

    EAGLE3's norm_output=False leaves normalization to the model's logits path;
    norm_output=True and P-EAGLE normalize here. Feature projection, learned
    feature masking, vocabulary mappings and the LM head remain model-owned.
    """
    if type(config) is not DraftConfig:
        raise TypeError("EAGLE assembly requires a resolved DraftConfig")
    spec = config.spec_config
    if type(spec) not in (Eagle3Spec, PEagleSpec):
        raise ValueError("EAGLE assembly requires Eagle3Spec or PEagleSpec")
    if type(config.residual_config) is not StandardResidualSpec:
        raise ValueError("EAGLE assembly currently requires standard residuals")
    if backend != "eager":
        raise ValueError("EAGLE assembly currently supports only the eager backend")
    for kind, nodes in (
        ("attention", config.attention_configs.values()),
        ("mlp", config.mlp_configs.values()),
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
    options = dict(hidden_size=config.hidden_size, device=device, dtype=dtype)
    layers = []
    for index, (definition, attention, rope) in enumerate(
        zip(definitions, attentions, ropes)
    ):
        parts = (
            build_component(
                "attention",
                attention,
                input_size=spec.attention_input_width(config.hidden_size, index),
                norm=config.norm_config,
                rope=rope,
                backend=backend,
                **options,
            ),
            build_component("mlp", config.mlp_configs[definition.mlp], **options),
        )
        layers.append(
            EagleDecoderStage(
                *parts,
                norm=config.norm_config,
                norm_before_residual=(
                    spec.norm_before_residual if type(spec) is PEagleSpec else False
                ),
                **options,
            )
            if index == 0
            else DecoderLayer(*parts, norm=config.norm_config, **options)
        )
    normalize_output = type(spec) is PEagleSpec or spec.norm_output
    return DraftBackbone(
        layers,
        rotary=rotary,
        final_norm=(
            build_component("norm", config.norm_config, **options)
            if normalize_output
            else nn.Identity()
        ),
        residual=build_component("residual", config.residual_config),
    )
