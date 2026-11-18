"""Dense Qwen3.5 MTP configuration, validation and merge-back layout."""

from specforge.checkpoint.native import register_native_target
from specforge.config.draft.attention import GQASpec
from specforge.config.draft.codec.components import decode_component
from specforge.config.draft.mlp import DenseFFNSpec
from specforge.config.draft.norm import RMSNormSpec
from specforge.config.draft.residual import StandardResidualSpec
from specforge.config.draft.rope import DefaultRoPESpec, MultiAxisRoPESpec


def validate_target(config, target):
    """Check the computation of Qwen3.5's native MTP before copying its weights."""
    if config.architecture != "MTPDraftModel" or config.num_hidden_layers != 1:
        raise ValueError("native Qwen3.5 MTP requires one MTP layer")
    layer = config.layer_configs[config.layer_types[0]]
    attention = config.attention_configs[layer.attention]
    mlp = config.mlp_configs[layer.mlp]
    if not (
        type(attention) is GQASpec
        and type(mlp) is DenseFFNSpec
        and type(config.norm_config) is RMSNormSpec
        and config.norm_config.weight_offset == 1
        and config.norm_config.has_weight
        and type(config.residual_config) is StandardResidualSpec
        and layer.sublayer_conv is None
        and attention.qk_norm
        and attention.qk_norm_position == "before_rope"
        and attention.qk_norm_scope == "head"
        and attention.qk_norm_config in (None, config.norm_config)
        and not attention.attention_bias
        and attention.sliding_window is None
    ):
        raise ValueError("draft components disagree with native Qwen3.5 MTP")
    rope = attention.rope_parameters or config.rope_parameters
    if type(rope) not in (DefaultRoPESpec, MultiAxisRoPESpec) or rope.interleaved:
        raise ValueError(
            "native Qwen3.5 MTP requires default or multi-axis half-split RoPE"
        )
    if config.target_bindings != ("embed_tokens", "lm_head"):
        raise ValueError("native Qwen3.5 MTP requires both target bindings")
    fields = _fields(config)
    for name, value in fields.items():
        if name in ("rope_theta", "partial_rotary_factor"):
            continue  # These may be recorded inside the target's RoPE node.
        actual = target.get(
            name, {"is_causal": True, "attention_dropout": 0.0}.get(name)
        )
        if actual is None:
            raise ValueError(f"target {name} must be explicit")
        if actual != value:
            raise ValueError(f"target {name} disagrees with draft")
    if target.get("rope_parameters") is None and target.get("rope_scaling") is None:
        raise ValueError("target must state rope_parameters or rope_scaling")
    for key in ("rope_parameters", "rope_scaling"):
        if target.get(key) is None:
            continue
        values = dict(target[key])
        for option in ("use_fope", "video_rope", "mrope_interleaved_glm"):
            if values.pop(option, False):
                raise ValueError(f"target {key}.{option} is unsupported")
        kind = values.pop("type", "default")
        values.setdefault("rope_type", kind)
        if "mrope_section" in values and values["rope_type"] == "default":
            values["rope_type"] = "mrope"
        if values["rope_type"] == "mrope":
            values.setdefault("mrope_interleaved", False)
        for name in ("rope_theta", "partial_rotary_factor"):
            value = values.get(name, target.get(name))
            if value is None:
                raise ValueError(f"target {key}.{name} must be explicit")
            if name in target and target[name] != value:
                raise ValueError(f"target {key}.{name} disagrees with top-level {name}")
            values[name] = value
        values.setdefault("interleaved", False)
        if decode_component("rope", values) != rope:
            raise ValueError(f"target {key} disagrees with draft")


def _fields(config):
    layer = config.layer_configs[config.layer_types[0]]
    attention = config.attention_configs[layer.attention]
    mlp = config.mlp_configs[layer.mlp]
    rope = attention.rope_parameters or config.rope_parameters
    return {
        "hidden_size": config.hidden_size,
        "vocab_size": config.vocab_size,
        "intermediate_size": mlp.intermediate_size,
        "hidden_act": mlp.hidden_act,
        "rms_norm_eps": config.norm_config.eps,
        "num_attention_heads": attention.num_attention_heads,
        "num_key_value_heads": attention.num_key_value_heads,
        "head_dim": attention.head_dim,
        "max_position_embeddings": config.max_position_embeddings,
        "partial_rotary_factor": rope.partial_rotary_factor,
        "attn_output_gate": attention.output_gate,
        "attention_dropout": attention.attention_dropout,
        "is_causal": attention.is_causal,
        "rope_theta": rope.rope_theta,
    }


def export(config, state, source, payload, output, *, copy_target=False):
    from ..writer import write_native

    weights = {}
    for name, tensor in state.items():
        for old, new in (
            ("backbone.layers.", "mtp.layers."),
            ("backbone.final_norm.", "mtp.norm."),
            ("fusion.", "mtp."),
        ):
            if name.startswith(old):
                weights[new + name[len(old) :]] = tensor
                break
        else:
            raise ValueError(f"unexpected native Qwen3.5 MTP tensor: {name}")
    write_native(
        config,
        weights,
        source,
        payload,
        output,
        prefix="mtp.",
        binding_names={name: f"mtp.{name}.weight" for name in config.target_bindings},
        optional_weights={"mtp.embed_tokens.weight", "mtp.lm_head.weight"},
        copy_target=copy_target,
    )


register_native_target(
    "qwen3_5_text",
    "MTPDraftModel",
    source_prefix="mtp",
    validate=validate_target,
    export=export,
)
