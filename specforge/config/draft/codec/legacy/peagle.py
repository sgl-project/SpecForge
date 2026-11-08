"""Decode P-EAGLE's effective Llama settings without importing Transformers."""

from collections.abc import Mapping

from ..._validation import positive_int
from ...architecture import PEagleSpec
from ...architecture.eagle_fusion import AuxiliaryFeatureSpec
from ...attention import GQASpec
from ...layer import LayerSpec
from ...mlp import DenseFFNSpec
from ...model import DraftConfig
from ...norm import RMSNormSpec
from ...residual import StandardResidualSpec
from ...rope import DefaultRoPESpec
from ..target import _text_config, validate_target
from .context import _HF_METADATA, _check_fields
from .rules import resolve_alias

_DEFAULTS = {
    "vocab_size": 32000,
    "hidden_size": 4096,
    "intermediate_size": 11008,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "hidden_act": "silu",
    "max_position_embeddings": 2048,
    "initializer_range": 0.02,
    "rms_norm_eps": 1e-6,
}
_FIELDS = (
    set(_DEFAULTS)
    | _HF_METADATA
    | {
        "architectures",
        "num_key_value_heads",
        "head_dim",
        "draft_vocab_size",
        "target_hidden_size",
        "eagle_config",
        "norm_before_residual",
        "pretraining_tp",
        "tie_word_embeddings",
        "attention_bias",
        "attention_dropout",
        "mlp_bias",
        "layer_types",
        "sliding_window",
        "use_sliding_window",
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
    "factor",
    "partial_rotary_factor",
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


def _rope(payload):
    sources = []
    for name in ("rope_parameters", "rope_scaling"):
        value = payload.get(name)
        if value is not None:
            if not isinstance(value, Mapping):
                raise ValueError(f"{name} must be an object or null")
            _check_fields(value, _ROPE_FIELDS, name)
            sources.append(value)
    # PEagleDraftModel reads theta but constructs unscaled, full-dimensional
    # LlamaRotaryEmbedding. Scaling fields were accepted but never executed.
    theta = resolve_alias("rope_theta", *sources, payload, default=10000.0)
    return DefaultRoPESpec(rope_theta=theta, partial_rotary_factor=1.0)


def decode_peagle(payload, *, target_config):
    _check_fields(payload, _FIELDS, "legacy P-EAGLE")
    target = _text_config(target_config)
    values = {**_DEFAULTS, **payload}
    depth = values["num_hidden_layers"]
    heads = values["num_attention_heads"]
    positive_int("num_hidden_layers", depth)
    positive_int("num_attention_heads", heads)
    positive_int("hidden_size", values["hidden_size"])
    tp = values.get("pretraining_tp", 1)
    positive_int("pretraining_tp", tp)
    if tp != 1:
        raise ValueError("legacy P-EAGLE pretraining_tp != 1 is not supported")
    method = payload.get("eagle_config")
    if method is None:
        method = {}
    if not isinstance(method, Mapping):
        raise ValueError("eagle_config must be an object or null")
    _check_fields(method, {"eagle_aux_hidden_state_layer_ids"}, "eagle_config")
    layers = method.get("eagle_aux_hidden_state_layer_ids")
    if layers is None:
        count = target.get("num_hidden_layers")
        positive_int("target_config.num_hidden_layers", count)
        layers = (1, count // 2 - 1, count - 4)
    if not isinstance(layers, (list, tuple)):
        raise ValueError("eagle_aux_hidden_state_layer_ids must be an array")
    norm_before = values.get("norm_before_residual")
    config = DraftConfig(
        architecture="PEagleDraftModel",
        hidden_size=values["hidden_size"],
        vocab_size=values["vocab_size"],
        draft_vocab_size=values.get("draft_vocab_size", values["vocab_size"]),
        max_position_embeddings=values["max_position_embeddings"],
        initializer_range=values["initializer_range"],
        tie_word_embeddings=False,
        attention_configs={
            "default": GQASpec(
                rope_parameters=None,
                num_attention_heads=heads,
                num_key_value_heads=(
                    heads
                    if values.get("num_key_value_heads") is None
                    else values["num_key_value_heads"]
                ),
                head_dim=(
                    values["hidden_size"] // heads
                    if values.get("head_dim") is None
                    else values["head_dim"]
                ),
                qk_norm=False,
                attention_bias=False,
                output_gate=False,
                is_causal=True,
                sliding_window=None,
                attention_dropout=0.0,
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
        layer_types=("default",) * depth,
        norm_config=RMSNormSpec(eps=values["rms_norm_eps"], weight_offset=0),
        residual_config=StandardResidualSpec(),
        rope_parameters=_rope(payload),
        spec_config=PEagleSpec(
            target_features=AuxiliaryFeatureSpec(
                target_layer_ids=tuple(layers),
                hidden_size=values.get("target_hidden_size", values["hidden_size"]),
            ),
            norm_before_residual=False if norm_before is None else norm_before,
        ),
        target_bindings=("embed_tokens",),
        target_logit_transform=None,
    )
    validate_target(config, target)
    return config
