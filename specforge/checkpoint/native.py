"""Pure native-target contracts shared by initialization and merge-back."""

from specforge.config.draft.attention import GQASpec
from specforge.config.draft.codec.components import decode_component
from specforge.config.draft.mlp import DenseFFNSpec
from specforge.config.draft.norm import RMSNormSpec
from specforge.config.draft.residual import StandardResidualSpec

_VALIDATORS = {}


def register_native_target(model_type, source_prefix):
    """Register a whole-native-model copy contract, not arbitrary tensor copies."""

    def register(validate):
        key = (model_type, source_prefix)
        if key in _VALIDATORS:
            raise ValueError(f"duplicate native target contract: {key}")
        _VALIDATORS[key] = validate
        return validate

    return register


def validate_target_initialization(config, target, from_target):
    # Only a root destination declares a complete native model initialization.
    entry = (from_target or {}).get("")
    source = entry.get("from") if isinstance(entry, dict) else entry
    if not isinstance(source, str):
        return
    text = target.get("text_config", target)
    validate = _VALIDATORS.get((text.get("model_type"), source))
    if validate is not None:
        validate(config, text)


@register_native_target("qwen3_5_text", "mtp")
def validate_qwen3_5_mtp(config, target):
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
    fields = {
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
        "is_causal": attention.is_causal,
        "rope_theta": rope.rope_theta,
    }
    for name, value in fields.items():
        if name in target and target[name] != value:
            raise ValueError(f"target {name} disagrees with draft")
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
        values.setdefault("rope_theta", target.get("rope_theta", rope.rope_theta))
        values.setdefault("partial_rotary_factor", fields["partial_rotary_factor"])
        values.setdefault("interleaved", False)
        if decode_component("rope", values) != rope:
            raise ValueError(f"target {key} disagrees with draft")
