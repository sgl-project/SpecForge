"""Shared field decoding for legacy target-context drafts; no algorithm dispatch."""

from ..._validation import positive_int
from ...architecture.features import TargetFeatureSpec
from ...attention import GQASpec, MHASpec, MLASpec
from ...layer import LayerSpec
from ...mlp import DenseFFNSpec
from ...norm import RMSNormSpec
from ...residual import StandardResidualSpec
from ..target import resolve_target_logits
from .rules import (
    default_target_layer_ids,
    resolve_alias,
    resolve_attention_layout,
    resolve_rope,
)

_REQUIRED_FIELDS = {
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
}
# Remaining Qwen3Config defaults used by the existing draft loader.
_DEFAULTS = {
    "num_key_value_heads": 32,
    "head_dim": 128,
    "hidden_act": "silu",
    "max_position_embeddings": 32768,
    "initializer_range": 0.02,
    "rms_norm_eps": 1e-6,
    "attention_bias": False,
    "attention_dropout": 0.0,
}
_HF_METADATA = frozenset(
    {
        "model_type",
        "auto_map",
        "transformers_version",
        "dtype",
        "torch_dtype",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "use_cache",
        "_name_or_path",
        "_commit_hash",
        "return_dict",
        "output_attentions",
        "output_hidden_states",
        "id2label",
        "label2id",
        "problem_type",
        "is_encoder_decoder",
        "chunk_size_feed_forward",
        "_attn_implementation",
        "attn_implementation",
        "_attn_implementation_internal",
        "_attn_implementation_autoset",
    }
)
_MLA_FIELDS = frozenset(
    {
        "q_lora_rank",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "rope_interleave",
    }
)
_TOP_FIELDS = (
    set(_DEFAULTS)
    | _REQUIRED_FIELDS
    | _HF_METADATA
    | _MLA_FIELDS
    | {
        "architectures",
        "draft_vocab_size",
        "dflash_config",
        "block_size",
        "num_target_layers",
        "tie_word_embeddings",
        "layer_types",
        "is_causal",
        "use_sliding_window",
        "sliding_window",
        "max_window_layers",
        "rope_theta",
        "rope_scaling",
        "rope_parameters",
        "partial_rotary_factor",
    }
)
_COMMON_FIELDS = {
    "block_size",
    "target_layer_ids",
    "mask_token_id",
    "attention_mode",
    "projector_type",
}


def _check_fields(payload, allowed, path):
    unknown = payload.keys() - allowed
    if unknown:
        raise ValueError(f"{path}: unknown fields {sorted(unknown, key=str)}")


def validate_dimensions(payload):
    missing = _REQUIRED_FIELDS - payload.keys()
    if missing:
        raise ValueError(f"legacy configuration is missing fields {sorted(missing)}")
    for name in sorted(_REQUIRED_FIELDS):
        positive_int(name, payload[name])


def _attention(mode, values, causal, window):
    common = dict(
        rope_parameters=None,
        num_attention_heads=values["num_attention_heads"],
        attention_bias=values["attention_bias"],
        attention_dropout=values["attention_dropout"],
        is_causal=causal,
        sliding_window=window,
    )
    if mode == "mla":
        required = _MLA_FIELDS - {"q_lora_rank", "rope_interleave"}
        missing = required - values.keys()
        if missing:
            raise ValueError(f"legacy MLA is missing fields {sorted(missing)}")
        return MLASpec(
            **common,
            **{name: values[name] for name in required},
            q_lora_rank=values.get("q_lora_rank"),
            rope_interleave=values.get("rope_interleave", True),
        )
    if _MLA_FIELDS & values.keys():
        raise ValueError("MLA projection fields require attention_mode='mla'")
    schema = MHASpec if mode == "mha" else GQASpec
    kv_heads = values["num_key_value_heads"]
    return schema(
        **common,
        num_key_value_heads=(
            values["num_attention_heads"] if kv_heads is None else kv_heads
        ),
        head_dim=values["head_dim"],
        qk_norm=True,
        output_gate=False,
    )


def read_context(payload, method, target_config):
    validate_dimensions(payload)
    mode = method.get("attention_mode", "gqa")
    if type(mode) is not str or mode.lower() not in {"gqa", "mha", "mla"}:
        raise ValueError("attention_mode must be gqa, mha, or mla")
    mode = mode.lower()
    values = {**_DEFAULTS, **payload}
    depth = values["num_hidden_layers"]
    layers, settings = resolve_attention_layout(values, depth)
    rope = resolve_rope(values)
    # Both embeddings and the head were already supplied by the target in the
    # existing training path. Resolving logits also requires an explicit target.
    logits = resolve_target_logits(target_config)
    target_depth = resolve_alias(
        "num_target_layers",
        payload,
        {"num_target_layers": target_config.get("num_hidden_layers")},
    )
    positive_int("num_target_layers", target_depth)
    feature_ids = method.get("target_layer_ids")
    if "target_layer_ids" not in method:
        feature_ids = default_target_layer_ids(target_depth, depth)
    if not isinstance(feature_ids, (list, tuple)):
        raise ValueError("target_layer_ids must be an array")
    features = TargetFeatureSpec(
        hidden_size=target_config["hidden_size"],
        mask_token_id=method.get("mask_token_id"), target_layer_ids=tuple(feature_ids)
    )
    fields = dict(
        hidden_size=values["hidden_size"],
        vocab_size=values["vocab_size"],
        draft_vocab_size=(
            values["vocab_size"]
            if values.get("draft_vocab_size") is None
            else values["draft_vocab_size"]
        ),
        max_position_embeddings=values["max_position_embeddings"],
        initializer_range=values["initializer_range"],
        tie_word_embeddings=False,  # Existing AutoDraftConfig forces untied weights.
        attention_configs={
            name: _attention(mode, values, *setting)
            for name, setting in settings.items()
        },
        mlp_configs={
            "default": DenseFFNSpec(
                intermediate_size=values["intermediate_size"],
                hidden_act=values["hidden_act"],
            )
        },
        conv_configs={},
        layer_configs={
            name: LayerSpec(
                attention=name,
                mlp="default",
                sublayer_conv=None,
            )
            for name in settings
        },
        layer_types=layers,
        norm_config=RMSNormSpec(eps=values["rms_norm_eps"], weight_offset=0),
        residual_config=StandardResidualSpec(),
        rope_parameters=rope,
        target_bindings=("embed_tokens", "lm_head"),
        target_logit_transform=logits,
    )
    block_size = resolve_alias("block_size", payload, method, ignore_none=True)
    return fields, features, block_size
