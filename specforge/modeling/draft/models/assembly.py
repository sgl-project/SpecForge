"""Assemble ordinary decoder stacks without taking over algorithm execution."""

from specforge.config.draft import DraftConfig
from specforge.config.draft.residual import StandardResidualSpec

from ..backbone import DraftBackbone
from ..layers import DecoderLayer, build_component, component_builder
from ..layers.rope import build_rotary


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
    if backend not in ("eager", "sdpa"):
        raise ValueError("decoder assembly supports eager and sdpa backends")
    input_width = getattr(config.spec_config, "attention_input_width", None)
    if input_width is not None and any(
        input_width(config.hidden_size, index) != config.hidden_size
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
    options = dict(hidden_size=config.hidden_size, device=device, dtype=dtype)
    layers = []
    for layer, attention, rope in zip(definitions, attentions, ropes):
        conv = None
        if layer.sublayer_conv is not None:
            conv = tuple(
                build_component(
                    "conv",
                    config.conv_configs[layer.sublayer_conv],
                    block_size=config.spec_config.protocol.num_model_slots,
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
                sublayer_conv=conv,
                **options,
            )
        )
    return DraftBackbone(
        layers,
        final_norm=build_component("norm", config.norm_config, **options),
        rotary=rotary,
        residual=build_component("residual", config.residual_config),
    )
