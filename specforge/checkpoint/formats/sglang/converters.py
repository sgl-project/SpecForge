"""Translate supported computation; reject values the serving format cannot hold."""

from dataclasses import asdict, replace

from specforge.config.draft.attention import GQASpec, MHASpec, MLASpec
from specforge.config.draft.conv import GroupedCausalConvSpec
from specforge.config.draft.logits import MuPLogitTransformSpec, NoLogitTransformSpec
from specforge.config.draft.mlp import DenseFFNSpec
from specforge.config.draft.norm import RMSNormSpec
from specforge.config.draft.residual import StandardResidualSpec
from specforge.config.draft.rope import (
    DefaultRoPESpec,
    LinearRoPESpec,
    Llama3RoPESpec,
    MultiAxisRoPESpec,
    YaRNSpec,
)

from . import register_converter


def require(condition, message):
    if not condition:
        raise ValueError(f"SGLang export: {message}")


def common(config, *, eagle=False, native=False):
    require(
        config.target_logit_transform is None
        or type(config.target_logit_transform)
        in (NoLogitTransformSpec, MuPLogitTransformSpec),
        f"target logit transform {type(config.target_logit_transform).__name__}",
    )
    require(
        type(config.residual_config) is StandardResidualSpec,
        f"residual kind {config.residual_config.type}",
    )
    require(
        type(config.norm_config) is RMSNormSpec, f"norm kind {config.norm_config.type}"
    )
    require(config.norm_config.has_weight, "weightless RMSNorm is unsupported")
    require(
        not config.tie_word_embeddings, "tied draft token tables are not representable"
    )
    layers = [config.layer_configs[name] for name in config.layer_types]
    attentions = [config.attention_configs[layer.attention] for layer in layers]
    a = attentions[0]
    require(type(a) in (GQASpec, MHASpec, MLASpec), f"attention kind {a.type}")
    for attention in attentions:
        require(
            type(attention) is type(a)
            and replace(attention, sliding_window=None)
            == replace(a, sliding_window=None),
            "per-layer attention values differ (including is_causal); pinned SGLang holds these once",
        )
    windows = {attention.sliding_window for attention in attentions} - {None}
    require(len(windows) <= 1, "per-layer sliding windows differ")
    require(
        not windows or min(windows) > 1,
        "sliding window must exceed 1",
    )
    mlps = [config.mlp_configs[layer.mlp] for layer in layers]
    require(
        all(type(m) is DenseFFNSpec and m == mlps[0] for m in mlps),
        "requires uniform dense MLP nodes",
    )
    require(mlps[0].hidden_act == "silu", f"MLP activation {mlps[0].hidden_act}")
    require(a.attention_dropout == 0, "attention_dropout must be zero for serving")
    rope = a.rope_parameters or config.rope_parameters
    require(
        type(rope) in (DefaultRoPESpec, LinearRoPESpec, Llama3RoPESpec, YaRNSpec)
        or ((eagle or native) and type(rope) is MultiAxisRoPESpec),
        f"RoPE kind {type(rope).__name__}",
    )
    require(
        not (eagle or native) or not rope.interleaved,
        "EAGLE3/native MTP require half-split rotary pairing",
    )
    scaling = asdict(rope) | {"rope_type": rope.rope_type}
    scaling.pop("interleaved")
    if type(rope) is LinearRoPESpec:
        scaling["factor"] = float(rope.factor)
    if type(rope) is DefaultRoPESpec:
        scaling["use_fope"] = False
    if type(rope) is MultiAxisRoPESpec:
        scaling.update(
            rope_type="default",
            mrope_section=list(rope.mrope_section),
            video_rope=False,
            mrope_interleaved_glm=False,
        )
    if type(rope) is YaRNSpec:
        # SGLang multiplies its mscale ratio by attn_factor. Equal mscale
        # numerator/denominator makes that ratio one; retain MLA score scaling.
        scaling.pop("attention_factor")
        scaling.update(
            mscale=rope.mscale_all_dim,
            attn_factor=rope.attention_factor,
            extrapolation_factor=1.0,
        )
    result = {
        name: getattr(config, name)
        for name in (
            "hidden_size",
            "vocab_size",
            "draft_vocab_size",
            "max_position_embeddings",
            "initializer_range",
            "num_hidden_layers",
        )
    }
    result.update(
        model_type="llama" if eagle else "qwen3",
        quantization_config=None,
        tie_word_embeddings=False,
        intermediate_size=mlps[0].intermediate_size,
        hidden_act="silu",
        rms_norm_eps=config.norm_config.eps,
        use_gemma_norm=config.norm_config.weight_offset == 1,
        num_attention_heads=a.num_attention_heads,
        attention_bias=a.attention_bias,
        attention_dropout=0.0,
        is_causal=a.is_causal,
        layer_types=[
            "full_attention" if v.sliding_window is None else "sliding_attention"
            for v in attentions
        ],
        sliding_window=next(iter(windows), None),
        use_sliding_window=bool(windows),
        max_window_layers=config.num_hidden_layers,
        rope_theta=rope.rope_theta,
        partial_rotary_factor=rope.partial_rotary_factor,
        rope_scaling=scaling,
        rope_parameters=dict(scaling),
        rope_is_neox_style=not rope.interleaved,
        is_neox_style=not rope.interleaved,
        attention_sink_bias=False,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
    )
    if isinstance(a, GQASpec):
        require(a.qk_norm == (not eagle), "qk_norm differs from serving implementation")
        if a.qk_norm:
            require(
                a.qk_norm_position == "before_rope" and a.qk_norm_scope == "head",
                "QK normalization must be per-head and before RoPE",
            )
            require(
                (a.qk_norm_config or config.norm_config) == config.norm_config,
                "QK normalization must use the model's weighted RMSNorm settings",
            )
        require(native or not a.output_gate, "attention output_gate is unsupported")
        require(
            not eagle or (a.is_causal and not windows and not a.attention_bias),
            "EAGLE3 requires causal full attention without biases",
        )
        require(
            not eagle or config.norm_config.weight_offset == 0,
            "EAGLE3 uses ordinary RMSNorm",
        )
        result.update(num_key_value_heads=a.num_key_value_heads, head_dim=a.head_dim)
    else:
        require(
            not eagle and not native, "MLA is only supported for DFlash-family exports"
        )
        require(
            not a.attention_bias and rope.interleaved,
            "MLA requires no bias and interleaved rotary input",
        )
        require(
            type(rope) in (DefaultRoPESpec, YaRNSpec)
            and rope.partial_rotary_factor == 1,
            "MLA requires full default/YaRN RoPE",
        )
        require(config.norm_config.weight_offset == 0, "MLA uses ordinary RMSNorm")
        require(
            type(rope) is not YaRNSpec or rope.truncate,
            "MLA YaRN requires truncate=True",
        )
        result.update(
            {
                name: getattr(a, name)
                for name in (
                    "q_lora_rank",
                    "kv_lora_rank",
                    "qk_nope_head_dim",
                    "qk_rope_head_dim",
                    "v_head_dim",
                )
            }
        )
        result.update(
            num_key_value_heads=a.num_attention_heads,
            head_dim=a.qk_head_dim,
            rope_interleave=rope.interleaved,
        )
    convs = [
        config.conv_configs[layer.sublayer_conv] if layer.sublayer_conv else None
        for layer in layers
    ]
    require(all(c == convs[0] for c in convs), "per-layer convolutions differ")
    conv = convs[0]
    require(
        conv is None or type(conv) is GroupedCausalConvSpec,
        "unsupported convolution kind",
    )
    require(
        not (eagle or native) or conv is None,
        "convolution is not supported by this architecture",
    )
    return result, a, conv


def context(config):
    require(
        config.spec_config.target_features.hidden_size == config.hidden_size,
        "DFlash-family target feature width must match draft hidden_size for SGLang",
    )
    require(
        config.target_bindings == ("embed_tokens", "lm_head"),
        "DFlash-family target_bindings must contain embed_tokens and lm_head",
    )
    result, a, conv = common(config)
    spec = config.spec_config
    protocol = spec.protocol
    window = result["sliding_window"]
    require(
        a.is_causal or window is None or window >= protocol.num_model_slots,
        "a non-causal sliding window must be at least the block size",
    )
    result["block_size"] = protocol.num_model_slots
    result["num_target_layers"] = None
    result["has_embed_tokens"] = False
    result["dflash_config"] = dict(
        block_size=protocol.num_model_slots,
        target_layer_ids=list(spec.target_features.target_layer_ids),
        num_target_layers=None,
        mask_token_id=spec.target_features.mask_token_id,
        mask_token="<|MASK|>",
        attention_mode=a.type,
        conv_kernel_size=conv.kernel_size if conv else 0,
        conv_group_size=conv.group_size if conv else 0,
        selector_rank=0,
        selector_top_k=0,
        output_multiplier=1.0,
        final_logit_softcapping=None,
        projector_type=None,
        attention_sink_bias=False,
        attention_value_scale=None,
    )
    return result


@register_converter("DFlashDraftModel")
def dflash(config):
    result = context(config)
    result["architectures"] = ["DFlashDraftModel"]
    return result


@register_converter("DFlash2DraftModel")
def dflash2(config):
    require(
        type(config.target_logit_transform) is not MuPLogitTransformSpec,
        f"pinned SGLang does not apply the target muP multiplier to {config.architecture} logits",
    )
    result = context(config)
    result["architectures"] = ["DFlash2DraftModel"]
    spec = config.spec_config
    result["dflash_config"].update(
        {
            name: getattr(spec, name)
            for name in (
                "selector_rank",
                "selector_top_k",
                "output_multiplier",
                "final_logit_softcapping",
            )
        }
    )
    return result


@register_converter("DominoDraftModel")
def domino(config):
    require(
        type(config.target_logit_transform) is not MuPLogitTransformSpec,
        f"pinned SGLang does not apply the target muP multiplier to {config.architecture} logits",
    )
    result = context(config)
    spec = config.spec_config
    result["architectures"] = ["DFlashDraftModel"]
    values = dict(
        projector_type="domino",
        shift_label=spec.block.shift_label,
        pure_draft_prefix_len=spec.pure_draft_prefix_len,
        gru_hidden_dim=spec.gru.hidden_size,
        emb_dim=spec.gru.projection_size,
    )
    result.update(values)
    result["dflash_config"].update(values)
    return result


@register_converter("DSparkDraftModel")
def dspark(config):
    require(
        config.norm_config.weight_offset == 0,
        "pinned SGLang DSpark does not support offset (1 + w) RMSNorm",
    )
    result = context(config)
    spec = config.spec_config
    require(spec.markov is not None, "pinned DSpark requires a Markov head")
    result["architectures"] = ["DSparkDraftModel"]
    result["dspark_config"] = {}
    result.update(
        {
            "dspark_" + name: None
            for name in (
                "block_size",
                "markov_rank",
                "markov_head_type",
                "noise_token_id",
                "target_layer_ids",
            )
        }
    )
    values = dict(
        markov_rank=spec.markov.rank,
        markov_head_type=spec.markov.type,
        enable_confidence_head=spec.confidence is not None,
        confidence_head_with_markov=(
            spec.confidence.with_markov if spec.confidence is not None else False
        ),
        gamma=spec.protocol.num_draft_tokens,
        sample_from_anchor=True,
        projector_type="dspark",
    )
    result.update(values)
    result["dflash_config"].update(values)
    return result


@register_converter("Eagle3DraftModel")
def eagle3(config):
    require(
        config.target_bindings == ("embed_tokens",),
        "EAGLE3 requires only embed_tokens bound",
    )
    result, attention, _ = common(config, eagle=True)
    require(
        config.hidden_size % attention.num_attention_heads == 0,
        "EAGLE3 hidden_size must be divisible by num_attention_heads",
    )
    spec = config.spec_config
    result.update(
        architectures=["LlamaForCausalLMEagle3"],
        target_hidden_size=spec.target_features.hidden_size,
        num_aux_hidden_states=len(spec.target_features.target_layer_ids),
        fc_norm=spec.normalize_features,
        use_aux_norm=False,
        norm_before_fc=False,
        norm_output=spec.norm_output,
        bias=False,
        mlp_bias=False,
        qkv_bias=False,
        original_max_position_embeddings=None,
        eagle_config=dict(
            eagle_aux_hidden_state_layer_ids=list(
                spec.target_features.target_layer_ids
            ),
            use_aux_hidden_state=True,
            norm_before_fc=False,
        ),
    )
    return result


@register_converter("MTPDraftModel")
def mtp(config):
    require(
        config.target_bindings == ("embed_tokens", "lm_head"),
        "native MTP requires both target bindings",
    )
    require(
        config.num_hidden_layers == 1
        and config.spec_config.target_features.hidden_size == config.hidden_size,
        "native Qwen3.5 MTP requires one layer and equal feature width",
    )
    result, a, _ = common(config, native=True)
    require(
        config.norm_config.weight_offset == 1
        and not a.attention_bias
        and a.sliding_window is None,
        "native Qwen3.5 requires offset RMSNorm and full attention without biases",
    )
    require(
        type(a.rope_parameters or config.rope_parameters)
        in (DefaultRoPESpec, MultiAxisRoPESpec),
        "native Qwen3.5 supports default or multi-axis RoPE only",
    )
    result.update(attn_output_gate=a.output_gate)
    return result


def unsupported(config):
    raise ValueError(
        f"SGLang export: pinned SGLang has no serving implementation for {config.architecture}"
    )


for name in ("DSpineDraftModel", "DMTPDraftModel", "PEagleDraftModel"):
    register_converter(name)(unsupported)


def serving_weights(config, state):
    result = {}
    for name, value in state.items():
        for prefix, replacement in (
            ("backbone.layers.", "layers."),
            ("backbone.final_norm.", "norm."),
            ("fusion.", ""),
            ("gru_head.", ""),
            ("vocabulary.", ""),
        ):
            if name.startswith(prefix):
                name = replacement + name[len(prefix) :]
                break
        require(name not in result, f"weight mapping collision: {name}")
        result[name] = value
    return result
