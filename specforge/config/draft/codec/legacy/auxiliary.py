"""Shared effective Llama fields and auxiliary selection; no architecture dispatch."""

from collections.abc import Mapping

from ..._validation import boolean, positive_int
from ...architecture.eagle_fusion import AuxiliaryFeatureSpec
from ...attention import GQASpec
from ...layer import LayerSpec
from ...mlp import DenseFFNSpec
from ...norm import RMSNormSpec
from ...residual import StandardResidualSpec
from ..target import _text_config
from .context import _HF_METADATA, _REQUIRED_FIELDS, _check_fields, validate_dimensions

_DEFAULTS = {
    "hidden_act": "silu",
    "max_position_embeddings": 2048,
    "initializer_range": 0.02,
    "rms_norm_eps": 1e-6,
}
_FIELDS = (
    set(_DEFAULTS)
    | _REQUIRED_FIELDS
    | _HF_METADATA
    | {
        "architectures",
        "num_key_value_heads",
        "head_dim",
        "draft_vocab_size",
        "target_hidden_size",
        "eagle_config",
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


def read_auxiliary(payload, target_config, *, num_layers=None, capture_layers=None):
    validate_dimensions(payload)
    target = _text_config(target_config)
    values = {**_DEFAULTS, **payload}
    depth = values["num_hidden_layers"] if num_layers is None else num_layers
    heads = values["num_attention_heads"]
    positive_int("num_hidden_layers", depth)
    tp = values.get("pretraining_tp", 1)
    positive_int("pretraining_tp", tp)
    if tp != 1:
        raise ValueError("legacy auxiliary draft pretraining_tp != 1 is not supported")
    method = payload.get("eagle_config")
    if method is None:
        method = {}
    if not isinstance(method, Mapping):
        raise ValueError("eagle_config must be an object or null")
    _check_fields(
        method,
        {"eagle_aux_hidden_state_layer_ids", "use_aux_hidden_state"},
        "eagle_config",
    )
    if "use_aux_hidden_state" in method:
        boolean("use_aux_hidden_state", method["use_aux_hidden_state"])
    layers = method.get("eagle_aux_hidden_state_layer_ids")
    for name, value in (
        ("eagle_aux_hidden_state_layer_ids", layers),
        ("capture_layers", capture_layers),
    ):
        if value is not None and not isinstance(value, (list, tuple)):
            raise ValueError(f"{name} must be an array")
    if capture_layers is not None:
        if layers is not None and tuple(layers) != tuple(capture_layers):
            raise ValueError("recorded capture_layers conflict with eagle_config")
        layers = capture_layers
    if layers is None:
        count = target.get("num_hidden_layers")
        positive_int("target_config.num_hidden_layers", count)
        layers = (1, count // 2 - 1, count - 4)
    if not isinstance(layers, (list, tuple)):
        raise ValueError("eagle_aux_hidden_state_layer_ids must be an array")
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
        target_bindings=(),
        target_logit_transform=None,
    )
    features = AuxiliaryFeatureSpec(
        target_layer_ids=tuple(layers),
        hidden_size=values.get("target_hidden_size", values["hidden_size"]),
    )
    return fields, features
