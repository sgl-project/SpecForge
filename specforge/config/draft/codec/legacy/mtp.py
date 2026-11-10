"""Resolve the existing Qwen3.5 MTP implementation, not the target's network."""

from collections.abc import Mapping

from ..._validation import boolean, positive_int
from ...architecture import MTPSpec
from ...architecture.features import FinalHiddenFeatureSpec
from ...attention import GQASpec
from ...layer import LayerSpec
from ...logits import MuPLogitTransformSpec
from ...mlp import DenseFFNSpec
from ...model import DraftConfig
from ...norm import RMSNormSpec
from ...residual import StandardResidualSpec
from ...rope import DefaultRoPESpec, MultiAxisRoPESpec
from ..target import _text_config, resolve_target_logits, validate_target
from .context import (
    _DEFAULTS,
    _HF_METADATA,
    _REQUIRED_FIELDS,
    _check_fields,
    validate_dimensions,
)
from .rules import reject_ignored_settings, resolve_alias

_FIELDS = (
    set(_DEFAULTS)
    | _REQUIRED_FIELDS
    | _HF_METADATA
    | {
        "architectures",
        "mtp_config",
        "attn_output_gate",
        "draft_vocab_size",
        "tie_word_embeddings",
        "mlp_bias",
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
_ROPE_FIELDS = {
    "type",
    "rope_type",
    "rope_theta",
    "partial_rotary_factor",
    "factor",
    "original_max_position_embeddings",
    "low_freq_factor",
    "high_freq_factor",
    "attention_factor",
    "beta_fast",
    "beta_slow",
    "mscale",
    "mscale_all_dim",
    "truncate",
    "mrope_section",
    "mrope_interleaved",
    "short_factor",
    "long_factor",
}


def _rope(payload, target):
    sources = []
    fraction = payload.get("partial_rotary_factor", 1.0)
    for name in ("rope_parameters", "rope_scaling"):
        value = payload.get(name)
        if value is None:
            continue
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be an object or null")
        _check_fields(value, _ROPE_FIELDS, name)
        alias = {"rope_type": value["type"]} if "type" in value else {}
        if resolve_alias("rope_type", value, alias, default="default") != "default":
            raise ValueError("legacy MTP supports only unscaled default RoPE")
        resolve_alias(
            "partial_rotary_factor", value, {"partial_rotary_factor": fraction}
        )
        sources.append(value)
    theta = resolve_alias("rope_theta", *sources, payload, default=10000.0)
    rope = DefaultRoPESpec(
        interleaved=False, rope_theta=theta, partial_rotary_factor=fraction
    )
    target_rope = target.get("rope_parameters") or {}
    if (
        target_rope.get("mrope_section") is not None
        and target_rope.get("rope_type", "default") in {"default", "mrope"}
        and target_rope.get("rope_theta", target.get("rope_theta")) == theta
        and target_rope.get(
            "partial_rotary_factor", target.get("partial_rotary_factor", 1.0)
        )
        == fraction
    ):
        # Equal text positions make multi-axis RoPE identical to the MTP rotation.
        return MultiAxisRoPESpec(
            interleaved=False,
            rope_theta=theta,
            partial_rotary_factor=fraction,
            mrope_section=tuple(target_rope["mrope_section"]),
            mrope_interleaved=target_rope.get("mrope_interleaved", False),
        )
    return rope


def decode_mtp(payload, *, target_config):
    _check_fields(payload, _FIELDS, "legacy MTP")
    validate_dimensions(payload)
    reject_ignored_settings(
        payload,
        mlp_bias=False,
        is_causal=True,
        num_hidden_layers=1,
        use_sliding_window=False,
        layer_types=["full_attention"],
    )
    target = _text_config(target_config)
    values = {**_DEFAULTS, **payload}
    # AutoDraftConfig validates this field, but the MTP head always uses vocab_size.
    positive_int(
        "draft_vocab_size",
        (
            values["vocab_size"]
            if values.get("draft_vocab_size") is None
            else values["draft_vocab_size"]
        ),
    )
    method = payload.get("mtp_config")
    if method is None:
        method = {}
    if not isinstance(method, Mapping):
        raise ValueError("mtp_config must be an object or null")
    _check_fields(method, {"share_lm_head"}, "mtp_config")
    share_head = method.get("share_lm_head", True)
    boolean("share_lm_head", share_head)
    logits = resolve_target_logits(target) if share_head else None
    if isinstance(logits, MuPLogitTransformSpec) and target.get("tie_word_embeddings"):
        raise ValueError("legacy MTP cannot fold muP into a tied embedding/LM head")
    heads = values["num_attention_heads"]
    config = DraftConfig(
        architecture="MTPDraftModel",
        hidden_size=values["hidden_size"],
        vocab_size=values["vocab_size"],
        draft_vocab_size=values["vocab_size"],
        max_position_embeddings=values["max_position_embeddings"],
        initializer_range=values["initializer_range"],
        tie_word_embeddings=False,
        attention_configs={
            "default": GQASpec(
                qk_norm_position="before_rope",
                qk_norm_scope="head",
                qk_norm_config=None,
                rope_parameters=None,
                num_attention_heads=heads,
                num_key_value_heads=(
                    heads
                    if values["num_key_value_heads"] is None
                    else values["num_key_value_heads"]
                ),
                head_dim=values["head_dim"],
                qk_norm=True,
                attention_bias=values["attention_bias"],
                output_gate=values.get("attn_output_gate", False),
                is_causal=True,
                sliding_window=None,
                attention_dropout=values["attention_dropout"],
            )
        },
        mlp_configs={
            "default": DenseFFNSpec(
                intermediate_size=values["intermediate_size"],
                hidden_act=values["hidden_act"],
            )
        },
        conv_configs={},
        layer_configs={"default": LayerSpec(attention="default", mlp="default")},
        # Qwen3_5MTPModel constructs one layer, independent of the HF depth field.
        layer_types=("default",),
        norm_config=RMSNormSpec(
            has_weight=True, eps=values["rms_norm_eps"], weight_offset=1
        ),
        residual_config=StandardResidualSpec(),
        rope_parameters=_rope(payload, target),
        spec_config=MTPSpec(
            target_features=FinalHiddenFeatureSpec(hidden_size=values["hidden_size"])
        ),
        target_bindings=(
            ("embed_tokens", "lm_head") if share_head else ("embed_tokens",)
        ),
        target_logit_transform=logits,
    )
    validate_target(config, target)
    return config
